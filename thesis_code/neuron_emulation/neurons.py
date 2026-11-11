import torch
import numpy as np
from pathlib import Path

from enum import Enum

class Neurons(Enum):
    EXCITATORY = 1
    PV = 2
    SOM = 3

def save_neurons(output_dir: Path, neurons: list[list[Neurons]]) -> None:
    npy_array = np.array([[n.value for n in row] for row in neurons], dtype=np.int8)
    np.save(output_dir / "neurons.npy", npy_array)

def load_neurons(work_dir: Path) -> list[list[Neurons]]:
    npy_array = np.load(work_dir / "neurons.npy")
    return [[Neurons(v) for v in row] for row in npy_array]

def sort_neuron_layer(layer: list[Neurons]) -> list[Neurons]:
    return sorted(layer, key=lambda n: n.value)

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