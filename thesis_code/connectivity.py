import torch
from enum import Enum
import numpy as np

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
    return [
        [choices[i] for i in torch.multinomial(probs, num_samples=n, replacement=True).tolist()]
        for n in ns_hidden
    ]

def build_recurrent_matrix(layer: list[Neurons]) -> torch.Tensor:
    type_idx = torch.tensor([type_order.index(n) for n in layer], dtype=torch.long)
    return type_matrix[type_idx][:, type_idx]

def build_pruned_matrix(rec_matrix: torch.Tensor) -> torch.Tensor:
    n_neurons = rec_matrix.numel()
    n_zero = int((rec_matrix == 0).sum())
    n_inh = int((rec_matrix == -1).sum())

    perm = torch.randperm(n_neurons)
    pruned_matrix = torch.ones(n_neurons, dtype=rec_matrix.dtype)

    pruned_matrix[perm[:n_zero]] = 0
    pruned_matrix[perm[n_zero:n_zero + n_inh]] = -1

    return pruned_matrix.view_as(rec_matrix)

def build_recurrent_and_pruned_matrices(neurons: list[list[Neurons]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    rec_matrices, pruned_matrices = [], []
    for layer in neurons:
        rec_matrix = build_recurrent_matrix(layer)
        rec_matrices.append(rec_matrix)

        pruned_matrix = build_pruned_matrix(rec_matrix)
        pruned_matrices.append(pruned_matrix)

        print(f"shapes are equal = {rec_matrix.shape == pruned_matrix.shape}")
        print(f"number of connections = {np.prod(rec_matrix.shape)} and {np.prod(pruned_matrix.shape)}")
        print(f"number of alive connections = {rec_matrix.count_nonzero()} and {pruned_matrix.count_nonzero()}")

        for v in (-1, 0, 1):
            assert (rec_matrix == v).sum() == (pruned_matrix == v).sum(), f"mismatch for value {v}"
        
    return rec_matrices, pruned_matrices