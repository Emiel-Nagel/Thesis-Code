import torch
from snntorch.surrogate import fast_sigmoid

from thesis_code import SRNN, datasets
from thesis_code.network_components.layers import StandardRecurrentLayer, OutputLayer
from thesis_code.decay_sampling import sample_heterogeneous_decays_normal, tau_to_beta
from thesis_code.connectivity import sample_ee_pv_som_neurons, build_recurrent_matrix, build_recurrent_and_pruned_matrices

from thesis_code.recording import SpikeRecorder, PerformanceRecorder, GradientRecorder
import thesis_code.training as tr
from thesis_code.plotting import plot_performance, plot_gradients, InteractiveSpikePlot

flushing = torch.set_flush_denormal(True)
print(f"Floating point flushing is set to {flushing}")

torch.manual_seed(42)
torch.set_num_threads(4)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

dt = 0.001     # Timesteps are in milliseconds
batch_size = 128
SHD_trainloader, SHD_testloader, input_dim, classes = datasets.get_SHD_dataset(batch_size, device, time_window=1000)

regularizer = tr.Regularizer(lam_lower=100.0, v_lower=1e-2, lam_uppers=[0.5, 0.5], v_uppers=[50, 50], L=2)

ns_hidden=[256]
layers = [input_dim] + ns_hidden
n_trials = 1

def build_srnn(rec_matrices: list[torch.Tensor], betas: list[torch.Tensor]) -> SRNN:
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

neurons = sample_ee_pv_som_neurons(ns_hidden, percent_pv=25.0, percent_som=25.0)
rec_matrices, _ = build_recurrent_and_pruned_matrices(neurons)
betas = [sample_heterogeneous_decays_normal(dt, n, tau_mean=10 * dt, tau_std=1 * dt, device=device) for n in ns_hidden]
net = build_srnn(rec_matrices, betas).to(device)

if __name__ == "__main__":
    perf_rec = PerformanceRecorder()
    # grad_rec=GradientRecorder(),
    # spk_rec=SpikeRecorder(num_hidden_layers=len(ns_hidden)),

    for trial in range(n_trials):
        acc_before = tr.test_net(net, testloader=SHD_testloader)
        print(f"Test Accuracy before training is: {acc_before}")

        net, perf_rec, _, _ = tr.train_net(
            net=net,
            device=device,
            trainloader=SHD_trainloader,
            lr=1e-3,
            n_epochs=20,
            regularizer=regularizer,
            perf_rec=perf_rec,
            # grad_rec=GradientRecorder(),
            # spk_rec=SpikeRecorder(num_hidden_layers=len(ns_hidden)),
        )

        acc_after = tr.test_net(net, testloader=SHD_testloader)
        print(f"Test Accuracy after training is: {acc_after}")

    plot_performance(perf_rec)