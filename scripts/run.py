import torch
from snntorch.surrogate import fast_sigmoid
from pathlib import Path

from thesis_code import SRNN, datasets, runs
from thesis_code.network_components.layers import SynapticRecurrentLayer, StandardRecurrentLayer, OutputLayer
from thesis_code.decay_sampling import tau_to_beta, sample_heterogeneous_decays_uniform, sample_heterogeneous_decays_normal
from thesis_code.connectivity import sample_ee_pv_som_neurons, save_neurons, build_recurrent_and_pruned_matrices, save_matrices, sample_exc_inh_recurrent_matrices

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

def build_srnns(
    dt: float,
    input_dim: int,
    output_dim: int,
    ns_hidden: list[int],
    percent_pv: float,
    percent_som: float,
    use_synapses: bool,
    use_daleslaw: bool,
    use_connectivity: bool,
    refractory_period: int,
    decay_cfg: dict,
    device: torch.device,
    output_dir: Path,
    add_control_net: bool,
) -> tuple[SRNN, SRNN | None]:
    layers = [input_dim] + ns_hidden
    print(f"Using layers: {layers}")

    if use_daleslaw:
        if use_connectivity:
            neurons = sample_ee_pv_som_neurons(
                ns_hidden=ns_hidden,
                percent_pv=percent_pv,
                percent_som=percent_som,
            )
            save_neurons(output_dir, neurons)
            rec_matrices, pruned_matrices = build_recurrent_and_pruned_matrices(neurons)
        else:
            rec_matrices = sample_exc_inh_recurrent_matrices(ns_hidden, percent_inh=percent_pv + percent_som)
            pruned_matrices = sample_exc_inh_recurrent_matrices(ns_hidden, percent_inh=percent_pv + percent_som)
    else:
        rec_matrices = [torch.ones(n, n) for n in ns_hidden]
        pruned_matrices = [torch.ones(n, n) for n in ns_hidden]

    forw_matrices = [torch.ones(n_in, n_out) for n_in, n_out in zip(layers[:-1], layers[1:])]

    print("Forward matrix shapes are: " + "".join(str(tuple(m.shape)) for m in forw_matrices))
    print("Recurrent matrix shapes are: " + "".join(str(tuple(m.shape)) for m in rec_matrices))
    print("Pruned matrix shapes are: " + "".join(str(tuple(m.shape)) for m in pruned_matrices))

    save_matrices(output_dir, rec_matrices, 
        names=[f"recurrent_mask_{i}" for i, _ in enumerate(rec_matrices)], 
        is_control=False
    )
    if add_control_net:
        save_matrices(output_dir, pruned_matrices, 
            names=[f"recurrent_mask_{i}" for i, _ in enumerate(pruned_matrices)], 
            is_control=True
        )

    match decay_cfg["beta_sampling_mode"]:
        case "single_value":
            betas = decay_cfg["tau_single_values"]
        case "normal":
            betas = [sample_heterogeneous_decays_normal(dt, n, tau_mean=m*dt, tau_std=s*dt, device=device)
                     for n, m, s in zip(ns_hidden, decay_cfg["tau_means"], decay_cfg["tau_stds"])]
        case "uniform":
            betas = [sample_heterogeneous_decays_uniform(dt, n, tau_upper=tu*dt, tau_lower=tl*dt, device=device)
                     for n, tu, tl in zip(ns_hidden, decay_cfg["tau_uppers"], decay_cfg["tau_lowers"])]

    rec_layer_class = SynapticRecurrentLayer if use_synapses else StandardRecurrentLayer

    def build_srnn(rec_matrices: list[torch.Tensor]) -> SRNN:
        return SRNN(
            rec_layers=[
                rec_layer_class(
                    forward_matrix=fm.clone().to(device),
                    recurrent_matrix=rm.clone().to(device),
                    beta=beta.clone().to(device),
                    spike_grad=fast_sigmoid(slope=25),
                    refractory_period=refractory_period,
                    use_daleslaw=use_daleslaw,
                ) for fm, rm, beta in zip(forw_matrices, rec_matrices, betas)],
            out_layer=OutputLayer(ns_hidden[-1], output_dim, beta=tau_to_beta(dt, decay_cfg["tau_out"]*dt))
        ).to(device)

    net = build_srnn(rec_matrices=rec_matrices)
    net_control = build_srnn(rec_matrices=pruned_matrices) if add_control_net else None
    return net, net_control

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
        acc_before, _ = tr.test_net(net, testloader)
        logger.log(f"Test Accuracy before training is: {acc_before}")

        net, optimizer, recorder = tr.train_net(
            net=net,
            logger=logger,
            device=device,
            trainloader=trainloader,
            testloader=testloader,
            lr=lr,
            n_epochs=n_epochs,
            regularizer=regularizer,
            recorder=recorder
        )

        acc_after, _ = tr.test_net(net, testloader)
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

    net, net_control = build_srnns(
        dt=dt,
        input_dim=input_dim,
        output_dim=len(classes),
        ns_hidden=ns_hidden,
        percent_pv=cfg["srnn"]["percent_pv"],
        percent_som=cfg["srnn"]["percent_som"],
        use_synapses=cfg["srnn"]["use_synapses"],
        use_daleslaw=cfg["srnn"]["use_daleslaw"],
        use_connectivity=cfg["srnn"]["use_connectivity"],
        refractory_period=cfg["srnn"]["refractory_period"],
        decay_cfg=cfg["srnn"]["decays"],

        device=device,
        output_dir=output_dir,
        add_control_net=cfg["srnn"]["add_control_net"],
    )

    recorder = Recorder(
        num_hidden_layers=len(ns_hidden),
        options=cfg.get("recording", {}),
    )

    logger.log("\n\n---------------")
    logger.log("Now testing net")

    net, optimizer, recorder = train(net, logger, device, SHD_trainloader, SHD_testloader, train_cfg["lr"], train_cfg["n_epochs"], regularizer, recorder)
    runs.save_net(net, optimizer, epoch=train_cfg["n_epochs"], save_path=output_dir / "net.checkpoint.pt")
    recorder.save(output_dir / "data_net")

    if net_control is not None:
        recorder.clear()   
        logger.log("\n\n---------------")
        logger.log("Now testing control net")

        net_control, optimizer_control, recorder = train(net_control, logger, device, SHD_trainloader, SHD_testloader, train_cfg["lr"], train_cfg["n_epochs"], regularizer, recorder)
        runs.save_net(net_control, optimizer_control, epoch=train_cfg["n_epochs"], save_path=output_dir / "net_control.checkpoint.pt")
        recorder.save(output_dir / "data_net_control")

    logger.write_to_file(output_dir)

if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    for config_path, cfg, run_name, output_dir in runs.iter_scheduled_runs():
        print(f"Now attempting run {run_name}")
        n_trials = cfg["training"]["n_trials"]
        seeds = range(n_trials)
        output_subdirs = [output_dir / f"seed_{i}" for i in seeds]

        datasets.download_SHD_dataset(
            time_window=cfg["training"]["dt"]*1e6,
            re_download=cfg["training"]["dataset"]["re_download"],
        )

        ctx = mp.get_context("spawn")       # required for CUDA
        with ProcessPoolExecutor(
            max_workers=cfg["training"]["n_workers"],
            mp_context=ctx,
        ) as pool:
            results = list(pool.map(run, [run_name] * len(seeds), seeds, [cfg] * len(seeds), output_subdirs))

        runs.push_output(output_dir)
        print(f"Finished run {run_name} successfully")
        runs.remove_config(config_path)