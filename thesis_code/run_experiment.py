import torch
from snntorch.surrogate import fast_sigmoid
from pathlib import Path

from thesis_code import SRNN, datasets, runs
from thesis_code.network_components.layers import OutputLayer, RecurrentLayerConfig, LAYER_CLASSES
from thesis_code.parameter_sampling import tau_to_beta, sample_values
from thesis_code.neuron_emulation.neurons import sample_ee_pv_som_neurons, save_neurons
from thesis_code.neuron_emulation.decays import get_membrane_decays_ee_pv_som
from thesis_code.neuron_emulation.connectivity import build_recurrent_and_pruned_matrices, save_matrices, sample_exc_inh_recurrent_matrices, create_forward_matrix_output

from thesis_code.recording import Recorder
import thesis_code.training as tr

from thesis_code.run_logging import Logger
from thesis_code.configs import Config

def setup(output_dir: Path, seed: int, cfg: Config) -> tuple[Path, Logger, torch.device]:
    output_dir = runs.create_seed_dir(output_dir, seed)
    logger = Logger(run_name=cfg.run_name, seed=seed)

    flushing = torch.set_flush_denormal(True)
    logger.log(f"Floating point flushing is set to {flushing}")

    torch.manual_seed(seed)
    torch.set_num_threads(4)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.log(f"Using device: {device}")
    return output_dir, logger, device

def build_srnns(
    dt: float,
    input_dim: int,
    output_dim: int,
    hidden_layer_cfgs: list[RecurrentLayerConfig],
    use_daleslaw: bool,
    use_connectivity: bool,
    refractory_period: int,
    tau_out: float,
    device: torch.device,
    output_dir: Path,
    add_control_net: bool,
) -> tuple[SRNN, SRNN | None]:    
    ns_hidden = [hl_cfg.n_neurons for hl_cfg in hidden_layer_cfgs]
    percent_pv = hidden_layer_cfgs[0].percent_pv
    percent_som = hidden_layer_cfgs[0].percent_som
    neurons = None
    if use_daleslaw:
        if use_connectivity:
            neurons = sample_ee_pv_som_neurons(ns_hidden, percent_pv, percent_som)
            save_neurons(output_dir, neurons)
            rec_matrices, pruned_matrices = build_recurrent_and_pruned_matrices(neurons)
        else:
            rec_matrices = sample_exc_inh_recurrent_matrices(ns_hidden, percent_inh=percent_pv + percent_som)
            pruned_matrices = sample_exc_inh_recurrent_matrices(ns_hidden, percent_inh=percent_pv + percent_som)
    else:
        rec_matrices = [torch.ones(n, n) for n in ns_hidden]
        pruned_matrices = [torch.ones(n, n) for n in ns_hidden]

    save_matrices(output_dir, rec_matrices, 
        names=[f"recurrent_mask_{i}" for i, _ in enumerate(rec_matrices)], 
        is_control=False
    )
    if add_control_net:
        save_matrices(output_dir, pruned_matrices, 
            names=[f"recurrent_mask_{i}" for i, _ in enumerate(pruned_matrices)], 
            is_control=True
        )

    def _build_srnn(rec_matrices: list[torch.Tensor]) -> SRNN:
        rec_layers = []
        n_prev = input_dim
        for hl_cfg, rec_matrix in zip(hidden_layer_cfgs, rec_matrices):
            n = hl_cfg.n_neurons

            if neurons is not None:
                beta, beta_min, beta_max = get_membrane_decays_ee_pv_som(neurons[0], dt, device)
            else:
                beta = sample_values(hl_cfg.beta, dt, n, device)
                beta_min, beta_max = 0.0, 1.0

            rec_layers.append(
                LAYER_CLASSES[hl_cfg.rec_layer_type](
                    forward_matrix=torch.ones(n_prev, n, device=device),
                    recurrent_matrix=rec_matrix.to(device),
                    beta=beta,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    beta_w=sample_values(hl_cfg.beta_w, dt, n, device) if hl_cfg.beta_w is not None else None,
                    a=sample_values(hl_cfg.a, dt, n, device) if hl_cfg.a is not None else None,
                    b=sample_values(hl_cfg.b, dt, n, device) if hl_cfg.b is not None else None,
                    spike_grad=fast_sigmoid(slope=25),
                    threshold=torch.ones(n, device=device),
                    refractory_period=refractory_period,
                    use_daleslaw=use_daleslaw,
                )
            )
            n_prev = n
        return SRNN(rec_layers, out_layer=OutputLayer(create_forward_matrix_output(neurons[0], output_dim, device=device), beta=tau_to_beta(dt, tau_out*dt))).to(device)
        return SRNN(rec_layers, out_layer=OutputLayer(n_prev, output_dim, beta=tau_to_beta(dt, tau_out*dt))).to(device)     # this is only for one test case

    net = _build_srnn(rec_matrices=rec_matrices)
    net_control = _build_srnn(rec_matrices=pruned_matrices) if add_control_net else None
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

def run(output_dir: Path, seed: int, cfg: Config) -> None:
    output_dir, logger, device = setup(output_dir, seed, cfg)

    train_cfg = cfg.training
    dt = train_cfg.dt
    SHD_trainloader, SHD_testloader, input_dim, classes = datasets.get_SHD_dataset(
        batch_size=train_cfg.batch_size,
        device=device,
        n_timepoints=train_cfg.dataset.n_timepoints,
    )

    reg_cfg = train_cfg.regularization
    regularizer = tr.Regularizer(
        lam_lower=reg_cfg.lam_lower,
        v_lower=reg_cfg.v_lower,
        lam_uppers=reg_cfg.lam_uppers,
        v_uppers=reg_cfg.v_uppers,
        L=reg_cfg.L,
    ) if reg_cfg.apply else None

    srnn_cfg = cfg.srnn

    net, net_control = build_srnns(
        dt=dt,
        input_dim=input_dim,
        output_dim=len(classes),
        hidden_layer_cfgs=srnn_cfg.hidden_layers,
        use_daleslaw=srnn_cfg.use_daleslaw,
        use_connectivity=srnn_cfg.use_connectivity,
        refractory_period=srnn_cfg.refractory_period,
        tau_out=srnn_cfg.tau_out,

        device=device,
        output_dir=output_dir,
        add_control_net=srnn_cfg.add_control_net,
    )

    recorder = Recorder(
        num_hidden_layers=len(srnn_cfg.hidden_layers),
        rec_cfg=cfg.recording,
    )

    logger.log("\n\n---------------")
    logger.log("Now testing net")

    net, optimizer, recorder = train(net, logger, device, SHD_trainloader, SHD_testloader, train_cfg.lr, train_cfg.n_epochs, regularizer, recorder)
    runs.save_net(net, optimizer, epoch=train_cfg.n_epochs, save_path=output_dir / "net.checkpoint.pt")
    recorder.save(output_dir / "data_net")

    if net_control is not None:
        recorder.clear()   
        logger.log("\n\n---------------")
        logger.log("Now testing control net")

        net_control, optimizer_control, recorder = train(net_control, logger, device, SHD_trainloader, SHD_testloader, train_cfg.lr, train_cfg.n_epochs, regularizer, recorder)
        runs.save_net(net_control, optimizer_control, epoch=train_cfg.n_epochs, save_path=output_dir / "net_control.checkpoint.pt")
        recorder.save(output_dir / "data_net_control")

    logger.write_to_file(output_dir)