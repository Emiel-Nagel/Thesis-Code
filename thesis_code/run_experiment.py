import torch
# from snntorch.surrogate import fast_sigmoid
from pathlib import Path

from thesis_code import SRNN, task_datasets, runs
# from thesis_code.network_components.layers import OutputLayer, RecurrentLayerConfig, LAYER_CLASSES
# from thesis_code.parameter_sampling import tau_to_beta, sample_values
# from thesis_code.neuron_emulation.neurons import sample_ee_pv_som_neurons, save_neurons
# from thesis_code.neuron_emulation.decays import get_membrane_decays_ee_pv_som
# from thesis_code.neuron_emulation.connectivity import build_recurrent_and_pruned_matrices, save_matrices, sample_exc_inh_recurrent_matrices, create_forward_matrix_output

from thesis_code.build_net import build_srnns

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
    SHD_trainloader, SHD_testloader, input_dim, classes = task_datasets.get_SHD_dataset(
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
        recurrent_layer_cfgs=srnn_cfg.recurrent_layers,
        # use_daleslaw=srnn_cfg.use_daleslaw,
        # use_connectivity=srnn_cfg.use_connectivity,
        # refractory_period=srnn_cfg.refractory_period,
        tau_out=srnn_cfg.tau_out,

        device=device,
        output_dir=output_dir,
        add_control_net=srnn_cfg.add_control_net,
    )

    recorder = Recorder(
        num_hidden_layers=len(srnn_cfg.recurrent_layers),
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