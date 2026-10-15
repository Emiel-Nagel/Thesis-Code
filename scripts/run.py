import torch
from snntorch.surrogate import fast_sigmoid
from pathlib import Path
# from tqdm import tqdm

from thesis_code import SRNN, datasets, runs
from thesis_code.network_components.layers import SynapticRecurrentLayer, StandardRecurrentLayer, OutputLayer
from thesis_code.decay_sampling import tau_to_beta, sample_heterogeneous_decays_uniform, sample_heterogeneous_decays_normal
from thesis_code.connectivity import sample_ee_pv_som_neurons, save_neurons, build_recurrent_and_pruned_matrices, save_matrices

from thesis_code.recording import Recorder
import thesis_code.training as tr

from thesis_code.logging import Logger

def setup(seed: int, logger: Logger) -> torch.device:
    flushing = torch.set_flush_denormal(True)
    logger.log(f"Floating point flushing is set to {flushing}")

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.log(f"Using device: {device}")
    return device

def build_srnn(
        rec_layer_class: SynapticRecurrentLayer | StandardRecurrentLayer,
        forw_matrices: list[torch.Tensor],
        rec_matrices: list[torch.Tensor],
        betas: list[torch.Tensor],
        out_layer: OutputLayer,
        device: torch.device
    ) -> SRNN:
    return SRNN(
        rec_layers=[
            rec_layer_class(
                forward_matrix=fm.clone().to(device),
                recurrent_matrix=rm.clone().to(device),
                beta=beta.clone().to(device),
                spike_grad=fast_sigmoid(slope=25),
                refractory_period = 5,
            ) for fm, rm, beta in zip(forw_matrices, rec_matrices, betas)],
        out_layer=out_layer
    ).to(device)

def train(
        net: SRNN,
        logger: Logger,
        device: torch.device,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        lr: float,
        n_epochs: int,
        regularizer: tr.Regularizer | None,
        recorder: Recorder | None,
    ) -> tuple[SRNN, torch.optim.Optimizer, Recorder]:
        acc_before = tr.test_net(net, testloader)
        logger.log(f"Test Accuracy before training is: {acc_before}")

        net, optimizer, recorder = tr.train_net(
            net=net,
            logger=logger,
            device=device,
            trainloader=trainloader,
            lr=lr,
            n_epochs=n_epochs,
            regularizer=regularizer,
            recorder=recorder
        )

        acc_after = tr.test_net(net, testloader)
        logger.log(f"Test Accuracy after training is: {acc_after}")

        return net, optimizer, recorder

def run(run_name: str, seed: int, cfg: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_name=run_name, seed=seed)
    device = setup(seed, logger)

    train_cfg = cfg["training"]
    dt = train_cfg["dt"]
    SHD_trainloader, SHD_testloader, input_dim, classes = datasets.get_SHD_dataset(
        batch_size=train_cfg["batch_size"],
        device=device,
        time_window=dt*1e6,
    )

    reg_cfg = train_cfg["regularization"]
    regularizer = tr.Regularizer(
        lam_lower=reg_cfg["lam_lower"],
        v_lower=reg_cfg["v_lower"],
        lam_uppers=reg_cfg["lam_uppers"],
        v_uppers=reg_cfg["v_uppers"],
        L=reg_cfg["L"]
    ) if reg_cfg["apply"] else None

    ns_hidden = cfg["srnn"]["ns_hidden"]
    layers = [input_dim] + ns_hidden
    neurons = sample_ee_pv_som_neurons(
        ns_hidden=cfg["srnn"]["ns_hidden"],
        percent_pv=cfg["srnn"]["percent_pv"],
        percent_som=cfg["srnn"]["percent_som"],
    )
    save_neurons(output_dir, neurons)
    forw_matrices = [torch.ones(n_in, n_out) for n_in, n_out in zip(layers[:-1], layers[1:])]
    # rec_matrices, pruned_matrices = build_recurrent_and_pruned_matrices(neurons)
    rec_matrices = [torch.ones(n, n) for n in ns_hidden]
    pruned_matrices = [torch.ones(n, n) for n in ns_hidden]
    save_matrices(output_dir, rec_matrices, 
        names=[f"recurrent_mask_{i}" for i, _ in enumerate(rec_matrices)], 
        is_control=False
    )

    decay_cfg = cfg["srnn"]["decays"]
    match decay_cfg["beta_sampling_mode"]:
        case "single_value":
            betas = decay_cfg["tau_single_values"]
        case "normal":
            betas = [sample_heterogeneous_decays_normal(dt, n, tau_mean=m*dt, tau_std=s*dt, device=device)
                     for n, m, s in zip(ns_hidden, decay_cfg["tau_means"], decay_cfg["tau_stds"])]
        case "uniform":
            betas = [sample_heterogeneous_decays_uniform(dt, n, tau_upper=tu*dt, tau_lower=tl*dt, device=device)
                     for n, tu, tl in zip(ns_hidden, decay_cfg["tau_uppers"], decay_cfg["tau_lowers"])]

    recorder = Recorder(
        num_hidden_layers=len(ns_hidden),
        options=cfg.get("recording", {}),
    )

    logger.log("\n\n---------------")
    logger.log("Now testing net")
    rec_layer_class = SynapticRecurrentLayer if cfg["srnn"]["use_synapses"] else StandardRecurrentLayer
    logger.log(f"Using neuron class {rec_layer_class}")
    net = build_srnn(
        rec_layer_class=rec_layer_class,
        forw_matrices=forw_matrices,
        rec_matrices=rec_matrices,
        betas=betas,
        out_layer=OutputLayer(ns_hidden[-1], len(classes), beta=tau_to_beta(dt, decay_cfg["tau_out"]*dt)),
        device=device,
    )
    net, optimizer, recorder = train(net, logger, device, SHD_trainloader, SHD_testloader, train_cfg["lr"], train_cfg["n_epochs"], regularizer, recorder)
    runs.save_net(net, optimizer, epoch=train_cfg["n_epochs"], save_path=output_dir / "net.checkpoint.pt")
    recorder.save(output_dir / "data_net")

    if cfg["srnn"]["add_control_net"]:
        recorder.clear()
        save_matrices(output_dir, pruned_matrices, 
            names=[f"recurrent_mask_{i}" for i, _ in enumerate(pruned_matrices)], 
            is_control=True
        )
            
        logger.log("\n\n---------------")
        logger.log("Now testing control net")
        net_control = build_srnn(
            rec_layer_class=rec_layer_class,
            forw_matrices=forw_matrices,
            rec_matrices=pruned_matrices,
            betas=betas,
            out_layer=OutputLayer(ns_hidden[-1], len(classes), beta=tau_to_beta(dt, decay_cfg["tau_out"]*dt)),
            device=device,
        )
        net_control, optimizer_control, recorder = train(net_control, logger, device, SHD_trainloader, SHD_testloader, train_cfg["lr"], train_cfg["n_epochs"], regularizer, recorder)
        runs.save_net(net_control, optimizer_control, epoch=train_cfg["n_epochs"], save_path=output_dir / "net_control.checkpoint.pt")
        recorder.save(output_dir / "data_net_control")

    logger.write_to_file(output_dir)

if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    cfg = runs.load_config()
    run_name = f"{runs.generate_run_id()}-{cfg["run_name"]}"
    output_dir = runs.create_output_dir(run_name)

    n_trials = cfg["training"]["n_trials"]
    seeds = range(n_trials)
    output_subdirs = [output_dir / f"seed_{i}" for i in seeds]

    datasets.download_SHD_dataset(
        time_window=cfg["training"]["dt"]*1e6,
        re_download=cfg["training"]["dataset"]["re_download"],
    )

    ctx = mp.get_context("spawn")       # required for CUDA
    # lock = ctx.RLock()
    with ProcessPoolExecutor(
        max_workers=3,
        mp_context=ctx,
        # initializer=lambda lock: tqdm.set_lock(lock),
        # initargs=(lock,),
    ) as pool:
        results = list(pool.map(run, [run_name] * len(seeds), seeds, [cfg] * len(seeds), output_subdirs))

    runs.push_output(output_dir)