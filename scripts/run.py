import torch
from thesis_code.datasets import get_SHD_dataset
from thesis_code.connectivity import sample_ee_pv_som_neurons, build_recurrent_matrices
from _deprecated.networks import models
from thesis_code.decay_sampling import sample_heterogeneous_decays_normal, tau_to_beta
from thesis_code.plotting import plot_performance, plot_gradients, InteractiveSpikePlot
from thesis_code.training import compute_loss, get_compute_loss_reg_fn, train_net, test_net

from matplotlib import pyplot as plt

flushing = torch.set_flush_denormal(True)
print(f"Floating point flushing is set to {flushing}")

torch.manual_seed(42)
torch.set_num_threads(4)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def run():
    dt = 0.001     # Timesteps are in milliseconds
    batch_size = 128
    SHD_trainloader, SHD_testloader, input_dim, classes = get_SHD_dataset(batch_size, device, time_window=1000, re_download=False)
    # SHD_trainloader, SHD_testloader, input_dim, classes = get_SHD_dataset(batch_size, device)         # use this after first download

    ns_hidden=[256, 256]
    neurons = sample_ee_pv_som_neurons(ns_hidden, percent_pv=25.0, percent_som=25.0)
    rec_matrices_pv_som_ee = build_recurrent_matrices(neurons)

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
    run()