import torch
from snntorch.surrogate import fast_sigmoid
from tqdm.auto import tqdm
import psutil, os

from thesis_code import SRNN
from thesis_code.network_components.layers import ALIFRecurrentLayer, OutputLayer
from thesis_code.parameter_sampling import tau_to_beta, sample_heterogeneous_normal
from thesis_code.neuron_emulation.connectivity import sample_ee_pv_som_neurons, build_recurrent_and_pruned_matrices, create_forward_matrix_output
from thesis_code.training import compute_loss, measure_accuracy, Regularizer

def sample_data_and_targets(batch_size: int, input_size: int, output_size: int, n_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.randint(0, output_size, (batch_size,), device=device)
    targets_as_binary = (targets.unsqueeze(-1) >> torch.arange(input_size - 1, -1, -1, device=device)) & 1

    raw_vectors = targets_as_binary[None, :, :].broadcast_to(n_steps, batch_size, input_size) * 0.6 + 0.2
    data = torch.bernoulli(raw_vectors).float()
    return data, targets

def test_net(n_steps: int, n_epochs: int, batch_size: int, lr: float, dt: float = 0.001) -> None:
    flushing = torch.set_flush_denormal(True)
    print(f"Floating point flushing is set to {flushing}")

    torch.manual_seed(0)
    torch.set_num_threads(4)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    regularizer = Regularizer(
        lam_lower = 100.0,
        v_lower = 1e-2,
        lam_uppers = [0.5],
        v_uppers = [50],
        L = 2,
    )

    input_dim = 4
    hidden_dim = 32
    output_dim = 16

    neurons = sample_ee_pv_som_neurons([hidden_dim], percent_pv=25, percent_som=25)
    rec_matrices, _ = build_recurrent_and_pruned_matrices(neurons)
    forward_matrix_output = create_forward_matrix_output(neurons[0], output_dim, device=device)
    
    net = SRNN(
        rec_layers=[
            ALIFRecurrentLayer(
                forward_matrix=torch.ones(input_dim, hidden_dim, device=device),
                recurrent_matrix=rec_matrices[0].to(device),
                beta=sample_heterogeneous_normal(dt, hidden_dim, tau_mean=10, tau_std=1, device=device),
                beta_w=sample_heterogeneous_normal(dt, hidden_dim, tau_mean=30, tau_std=5, device=device),
                a=sample_heterogeneous_normal(dt, hidden_dim, tau_mean=3, tau_std=2, device=device),
                b=sample_heterogeneous_normal(dt, hidden_dim, tau_mean=1, tau_std=1, device=device),
                spike_grad=fast_sigmoid(slope=25),
                threshold=torch.ones(hidden_dim, device=device),
                refractory_period=0,
                use_daleslaw=True,
            )   
        ],
        out_layer=OutputLayer(
            forward_matrix=forward_matrix_output,
            beta=tau_to_beta(dt, 20*dt),
        )
    ).to(device)

    net.train()
    progress_bar = tqdm(range(n_epochs), total=n_epochs, position=0, desc=f"Epoch Progress"
                            , dynamic_ncols=True, bar_format="{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {bar}",)
    status_bar = tqdm(total=0, position=1, bar_format="{desc}", dynamic_ncols=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    process = psutil.Process(os.getpid())

    for epoch in progress_bar:
        data, targets = sample_data_and_targets(batch_size=batch_size, input_size=input_dim, output_size=output_dim, n_steps=n_steps, device=device)

        net.reset()
        spk_outs, mem_outs, hidden_spks, hidden_mems = net.forward_sequence(data)

        loss_val = compute_loss(mem_outs, targets) + regularizer(hidden_spks)
        optimizer.zero_grad()
        loss_val.backward()

        optimizer.step()
        net.apply_daleslaw()

        acc = measure_accuracy(mem_outs.detach(), targets)

        status_bar.set_description_str(
            f"loss={loss_val.item():.2f}, acc={acc * 100:.2f}% "
            f"time_elapsed={progress_bar.format_dict['elapsed']:.1f}s, used_memory={process.memory_info().rss / 1e9:.2f} GB"
        )

if __name__ == "__main__":
    test_net(
        n_steps=100,
        n_epochs=100,
        batch_size=128,
        lr=0.001,
    )