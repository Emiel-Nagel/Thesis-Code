import torch
from torch import Tensor
from enum import Enum

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
        "SOM": -1,
    },
}

type_order = list(Neurons)  # [EXCITATORY, PV, SOM]
type_matrix = torch.tensor(
    [[connectivity_rules[a.name][b.name] for b in type_order] for a in type_order],
    dtype=torch.float,
)

def sample_ee_pv_som_neurons(ns_hidden: list[int], percent_pv: float, percent_som: float) -> list[list[Neurons]]:
    percent_excitatory = 100 - (percent_pv + percent_som)
    assert percent_excitatory > 0, \
        f"'percent_pv' + 'percent_som' = {percent_pv + percent_som} ,which is > 100%, \
        there are less than zero excitatory neurons left"

    choices = list(Neurons)
    probs = torch.tensor([percent_excitatory / 100.0, percent_pv / 100.0, percent_som / 100.0])
    return [
        [choices[i] for i in torch.multinomial(probs, num_samples=n, replacement=True).tolist()]
        for n in ns_hidden
    ]

def build_recurrent_matrices(neurons: list[list[Neurons]]) -> list[Tensor]:
    rec_matrices = []
    for layer in neurons:
        type_idx = torch.tensor([type_order.index(n) for n in layer], dtype=torch.long)
        rec_matrix = type_matrix[type_idx][:, type_idx]
        rec_matrices.append(rec_matrix)
    return rec_matrices
