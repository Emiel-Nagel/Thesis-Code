import torch
from snntorch.surrogate import fast_sigmoid

from thesis_code import SRNN, datasets, runs
from thesis_code.network_components.layers import SynapticRecurrentLayer, StandardRecurrentLayer, OutputLayer
from thesis_code.decay_sampling import tau_to_beta, sample_heterogeneous_decays_uniform, sample_heterogeneous_decays_normal
from thesis_code.connectivity import sample_ee_pv_som_neurons, build_recurrent_matrix, build_recurrent_and_pruned_matrices

from thesis_code.recording import SpikeRecorder, PerformanceRecorder, GradientRecorder
import thesis_code.training as tr
from thesis_code.plotting import plot_performance, plot_gradients, InteractiveSpikePlot

from matplotlib import pyplot as plt

def setup() -> torch.device:
    flushing = torch.set_flush_denormal(True)
    print(f"Floating point flushing is set to {flushing}")

    torch.manual_seed(42)
    torch.set_num_threads(4)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    return device

def build_srnn(layers: list, rec_matrices: list[torch.Tensor], betas: list[torch.Tensor], device: torch.device) -> SRNN:
    return SRNN(
        rec_layers=[
            StandardRecurrentLayer(
                forward_matrix=torch.ones(n_prev, n).to(device),
                recurrent_matrix=rec_matrix.to(device),
                beta=beta.clone().to(device),
                spike_grad=fast_sigmoid(slope=25),
            ) for n_prev, n, rec_matrix, beta in zip(layers[:-1], layers[1:], rec_matrices, betas)],
        out_layer=OutputLayer(
            n_in=layers[-1],
            n_out=len(classes),
            beta=tau_to_beta(dt, 20 * dt),
        ),
    )

def run(cfg: dict) -> ...:
    device = setup()
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

    layer_class = SynapticRecurrentLayer if cfg["srnn"]["use_synapses"] else StandardRecurrentLayer

    add_control_net = cfg["srnn"]["add_control_net"]
    rec_layers, rec_layers_control = []
    for n_in, n_out, rm, pm, beta in zip(layers[:-1], layers[1:], rec_matrices, pruned_matrices, betas):
        fm = torch.ones(n_in, n_out)
        rec_layers.append(layer_class(fm.clone().to(device), rm.to(device), beta.clone().to(device), spike_grad=fast_sigmoid(slope=25)))
        if add_control_net:
            rec_layers_control.append(layer_class(fm.clone().to(device), pm.to(device), beta.clone().to(device), spike_grad=fast_sigmoid(slope=25)))

    net = SRNN(
        rec_layers=rec_layers,
        out_layer=OutputLayer(ns_hidden[-1], len(classes), beta=tau_to_beta(dt, decay_cfg["tau_out"]*dt)),
    ).to(device)

    
    acc_before = tr.test_net(net, testloader=SHD_testloader)
    print(f"Test Accuracy before training is: {acc_before}")

    net, perf_rec, _, _ = tr.train_net(
        net=net,
        device=device,
        trainloader=SHD_trainloader,
        lr=train_cfg["lr"],
        n_epochs=train_cfg["n_epochs"],
        regularizer=regularizer,
        perf_rec=perf_rec,
        # grad_rec=GradientRecorder(),
        # spk_rec=SpikeRecorder(num_hidden_layers=len(ns_hidden)),
    )

    acc_after = tr.test_net(net, testloader=SHD_testloader)
    print(f"Test Accuracy after training is: {acc_after}")

    if add_control_net:
        net_control = SRNN(
            rec_layers=rec_layers_control,
            out_layer=OutputLayer(ns_hidden[-1], len(classes), beta=tau_to_beta(dt, decay_cfg["tau_out"]*dt)),
        ).to(device)



    layers = [input_dim] + ns_hidden
    net = models.StandardSRNN(
        forward_matrices=[torch.ones(n_in, n_out) for n_in, n_out in zip(layers[:-1], layers[1:])],
        recurrent_matrices=rec_matrices_pv_som_ee,
        betas=[sample_heterogeneous_decays_normal(dt, n, tau_mean=10 * dt, tau_std=1 * dt, device=device) for n in ns_hidden],
        beta_out=tau_to_beta(dt, 20 * dt),
        n_classes=len(classes),
    ).to(device)

    compute_loss_reg = get_compute_loss_reg_fn(
        lam_lower=100.0,
        v_lower=1e-2,               # min n spikes
        lam_uppers=[0.5, 0.5],
        v_uppers=[50, 50],          # max n spikes per layer
        L=2,
    )

    accuracy = test_net(net, SHD_testloader)
    print(f"Test Accuracy before training is: {accuracy}")

    net, loss_hist, acc_hist, spike_hist, max_grad_rec, avg_grad_rec = train_net(net, SHD_trainloader,
        loss_fn=lambda mem_outs, targets, hidden_spks: compute_loss(mem_outs, targets) + compute_loss_reg(hidden_spks),
        lr=1e-3,
        n_epochs=50,
        max_iters=None,
    )

    accuracy = test_net(net, SHD_testloader)
    print(f"Test Accuracy after training is: {accuracy}")

    # add code to send data to github repo

    plot_performance(loss_hist, acc_hist)
    plot_gradients(max_grad_rec, avg_grad_rec)

    InteractiveSpikePlot(spike_hist, iteration_i_start=0, batch_item_i=0).draw()
    plt.show()

if __name__ == "__main__":
    cfg = run.get_config()
    run(cfg)

import sys

lr = sys.argv[1]