import torch
from snntorch.surrogate import fast_sigmoid

from thesis_code import SRNN, datasets, runs
from thesis_code.network_components.layers import SynapticRecurrentLayer, StandardRecurrentLayer, OutputLayer
from thesis_code.decay_sampling import tau_to_beta, sample_heterogeneous_decays_uniform, sample_heterogeneous_decays_normal
from thesis_code.connectivity import sample_ee_pv_som_neurons, build_recurrent_matrix, build_recurrent_and_pruned_matrices

from thesis_code.recording import Recorder, SpikeRecorder, PerformanceRecorder, GradientRecorder
import thesis_code.training as tr
from thesis_code.plotting import plot_performance, plot_gradients, InteractiveSpikePlot

from matplotlib import pyplot as plt

def setup(seed: int) -> torch.device:
    flushing = torch.set_flush_denormal(True)
    print(f"Floating point flushing is set to {flushing}")

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
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
            ) for fm, rm, beta in zip(forw_matrices, rec_matrices, betas)],
        out_layer=out_layer
    ).to(device)

def train(
        net: SRNN,
        device: torch.device,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        lr: int,
        n_epochs: int,
        regularizer: tr.Regularizer | None,
        recorder: Recorder | None,
    ) -> tuple[SRNN, Recorder]:
        acc_before = tr.test_net(net, testloader)
        print(f"Test Accuracy before training is: {acc_before}")

        net, recorder = tr.train_net(
            net=net,
            device=device,
            trainloader=trainloader,
            lr=1e-3,
            n_epochs=20,
            regularizer=regularizer,
            perf_rec=perf_rec,
            # grad_rec=GradientRecorder(),
            # spk_rec=SpikeRecorder(num_hidden_layers=len(ns_hidden)),
        )

        acc_after = tr.test_net(net, testloader)
        print(f"Test Accuracy after training is: {acc_after}")

        return net, recorder

def run(seed: int, cfg: dict, recorder: Recorder) -> tuple[..., Recorder]:
    device = setup(seed)
    train_cfg = cfg["training"]
    dt = train_cfg["dt"]
    SHD_trainloader, SHD_testloader, input_dim, classes = datasets.get_SHD_dataset(
        batch_size=train_cfg["batch_size"],
        device=device,
        time_window=dt*1e6,
        re_download=train_cfg["dataset"]["re_download"],
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
    forw_matrices = [torch.ones(n_in, n_out) for n_in, n_out in zip(layers[:-1], layers[1:])]
    rec_matrices, pruned_matrices = build_recurrent_and_pruned_matrices(neurons)

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

    print("\n\n---------------")
    print("Now testing net")
    net = build_srnn(
        layer_class=SynapticRecurrentLayer if cfg["srnn"]["use_synapses"] else StandardRecurrentLayer,
        forw_matrices=forw_matrices,
        rec_matrices=rec_matrices,
        betas=betas,
        out_layer=OutputLayer(ns_hidden[-1], len(classes), beta=tau_to_beta(dt, decay_cfg["tau_out"]*dt)),
        device=device,
    )
    net, recorder = train(net, device, SHD_trainloader, SHD_testloader, train_cfg["lr"], train_cfg["n_epochs"],
                          regularizer, recorder)

    if cfg["srnn"]["add_control_net"]:
        print("\n\n---------------")
        print("Now testing control net")
        net_control = build_srnn(
            layer_class=SynapticRecurrentLayer if cfg["srnn"]["use_synapses"] else StandardRecurrentLayer,
            forw_matrices=forw_matrices,
            rec_matrices=pruned_matrices,
            betas=betas,
            out_layer=OutputLayer(ns_hidden[-1], len(classes), beta=tau_to_beta(dt, decay_cfg["tau_out"]*dt)),
            device=device,
        )
        net_control, recorder = train(net_control, device, SHD_trainloader, SHD_testloader, train_cfg["lr"], train_cfg["n_epochs"],
                                      regularizer, recorder)

    # add code to send data to github repo

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    cfg = run.get_config()
    n_trials = cfg["training"]["n_trials"]
    seeds = range(n_trials)

    ctx = mp.get_context("spawn")       # required for CUDA
    with ProcessPoolExecutor(max_workers=n_trials, mp_context=ctx) as pool:
        results = list(pool.map(run, seeds, [cfg] * len(seeds)))