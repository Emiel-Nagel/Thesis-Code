import torch
from enum import Enum
import numpy as np
from pathlib import Path

class Neurons(Enum):
    EXCITATORY = 1
    PV = 2
    SOM = 3

connectivity_rules = {
    "EXCITATORY": {
        "EXCITATORY": 1,
        "PV": 1,
        "SOM": 1,
    },
    "PV": {
        "EXCITATORY": -1,
        "PV": -1,
        "SOM": 0,
    },
    "SOM": {
        "EXCITATORY": -1,
        "PV": -1,
        "SOM": 0,      # should be 0
    },
}

type_order = list(Neurons)  # [EXCITATORY, PV, SOM]
type_matrix = torch.tensor(
    [[connectivity_rules[a.name][b.name] for b in type_order] for a in type_order],
    dtype=torch.int8,
)

def sample_ee_pv_som_neurons(ns_hidden: list[int], percent_pv: float, percent_som: float) -> list[list[Neurons]]:
    percent_excitatory = 100 - (percent_pv + percent_som)
    assert percent_excitatory > 0, \
        f"'percent_pv' + 'percent_som' = {percent_pv + percent_som} ,which is > 100%, \
        there are less than zero excitatory neurons left"

    choices = list(Neurons)
    probs = torch.tensor([percent_excitatory / 100.0, percent_pv / 100.0, percent_som / 100.0], dtype=torch.float)
    neurons = [
        [choices[i] for i in torch.multinomial(probs, num_samples=n, replacement=True).tolist()]
        for n in ns_hidden
    ]
    return [sort_neuron_layer(l) for l in neurons]

def save_neurons(output_dir: Path, neurons: list[list[Neurons]]) -> None:
    npy_array = np.array([[n.value for n in row] for row in neurons], dtype=np.int8)
    np.save(output_dir / "neurons.npy", npy_array)

def load_neurons(work_dir: Path) -> list[list[Neurons]]:
    npy_array = np.load(work_dir / "neurons.npy")
    return [[Neurons(v) for v in row] for row in npy_array]

def sort_neuron_layer(layer: list[Neurons]) -> list[Neurons]:
    return sorted(layer, key=lambda n: n.value)

def build_recurrent_matrix(layer: list[Neurons]) -> torch.Tensor:
    # layer = sort_neuron_layer(layer)
    type_idx = torch.tensor([type_order.index(n) for n in layer], dtype=torch.long)
    return type_matrix[type_idx][:, type_idx]

def prune_recurrent_matrix(rec_matrix: torch.Tensor) -> torch.Tensor:
    """
    Scrambles all inhibitory and nonexistent connections, but keeps excitatory connections the same.
    """
    pruned_matrix = rec_matrix.clone()
    mask = (rec_matrix == 0) | (rec_matrix == -1)
    vals = rec_matrix[mask]
    perm = torch.randperm(vals.numel(), device=rec_matrix.device)
    pruned_matrix[mask] = vals[perm]
    return pruned_matrix

def build_pruned_matrix(rec_matrix: torch.Tensor) -> torch.Tensor:
    """
    Constructs a new pruned matrix where all connections are scrambled.
    """
    n_neurons = rec_matrix.numel()
    n_zero = int((rec_matrix == 0).sum())
    n_inh = int((rec_matrix == -1).sum())

    perm = torch.randperm(n_neurons)
    pruned_matrix = torch.ones(n_neurons, dtype=rec_matrix.dtype)

    pruned_matrix[perm[:n_zero]] = 0
    pruned_matrix[perm[n_zero:n_zero + n_inh]] = -1

    return pruned_matrix.view_as(rec_matrix)

def remove_self_connections(rec_matrix: torch.Tensor) -> torch.Tensor:
    return rec_matrix.fill_diagonal_(0.0)

def build_recurrent_and_pruned_matrices(neurons: list[list[Neurons]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    rec_matrices, pruned_matrices = [], []
    for layer in neurons:
        rec_matrix = build_recurrent_matrix(layer)
        rec_matrices.append(rec_matrix)

        pruned_matrix = prune_recurrent_matrix(rec_matrix)
        pruned_matrices.append(pruned_matrix)

        print(f"shapes are equal = {rec_matrix.shape == pruned_matrix.shape}")
        print(f"number of connections = {np.prod(rec_matrix.shape)} and {np.prod(pruned_matrix.shape)}")
        print(f"number of alive connections = {rec_matrix.count_nonzero()} and {pruned_matrix.count_nonzero()}")

        for v in (-1, 0, 1):
            assert (rec_matrix == v).sum() == (pruned_matrix == v).sum(), f"mismatch for value {v}"
        
    return rec_matrices, pruned_matrices

def _filename(is_control: bool) -> str:
    return "control_net_weight_masks.npz" if is_control else "net_weights_masks.npz"

def save_matrices(output_dir: Path, matrices: list[torch.Tensor], names: list[str], is_control: bool = False) -> None:
    if len(set(names)) != len(names):
        raise ValueError("names must be unique")
    
    out_dict = {n: m.detach().cpu().numpy() for n, m in zip(names, matrices, strict=True)}
    np.savez_compressed(output_dir / _filename(is_control), **out_dict)

def load_matrices(work_dir: Path, is_control: bool = False) -> dict[str, np.ndarray]:
    with np.load(work_dir / _filename(is_control)) as data:
        return dict(data.items())

def sample_exc_inh_recurrent_matrices(ns_hidden: list[int], percent_inh: float) -> list[torch.Tensor]:
    def _build_matrix(n_neurons: int) -> torch.Tensor:
        n_inh = round((percent_inh / 100.0) * n_neurons)
        n_exc = n_neurons - n_inh
        one_dim_tensor = torch.cat([torch.ones(n_exc), -torch.ones(n_inh)])
        return one_dim_tensor.unsqueeze(1).expand(-1, n_neurons).clone()

    return [_build_matrix(n_neurons) for n_neurons in ns_hidden]