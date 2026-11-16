import torch
import numpy as np
from pathlib import Path
from typing import Literal

from enum import Enum

class Neurons(Enum):
    EXCITATORY = 1
    PV = 2
    SOM = 3
    INHIBITORY = 4

neuron_map = {member.name: member.value for member in Neurons}

def get_neuron_range(layer: list[Neurons], neuron_type: Literal["Excitatory", "PV", "SOM"]) -> tuple[int, int]:
    neurons: np.ndarray = np.array([n.value for n in layer], dtype=np.int8)
    key = neuron_map[neuron_type]
    return (
        np.min(np.where(neurons == key)),
        np.max(np.where(neurons == key))
    )

def get_neuron_ranges(layer: list[Neurons]) -> dict[str, tuple[int, int]]:
    return {k: get_neuron_range(layer, k) for k in neuron_map.keys()}

def save_neurons(output_dir: Path, neurons: list[list[Neurons]]) -> None:
    npy_array = np.array([[n.value for n in row] for row in neurons], dtype=np.int8)
    np.save(output_dir / "neurons.npy", npy_array)

def load_neurons(work_dir: Path) -> list[list[Neurons]]:
    npy_array = np.load(work_dir / "neurons.npy")
    return [[Neurons(v) for v in row] for row in npy_array]

def sort_neuron_layer(layer: list[Neurons]) -> list[Neurons]:
    return sorted(layer, key=lambda n: n.value)

def sample_exc_pv_som_neurons(n_neurons: int, p_pv: float, p_som: float) -> list[Neurons]:
    n_pv = round(p_pv * n_neurons)
    n_som = round(p_som * n_neurons)
    n_exc = n_neurons - n_pv - n_som
    assert n_exc > 0, f"p_pv + p_som = {p_pv + p_som}, which is >= 1.0, so there would be no excitatory neurons left"
    return [Neurons.EXCITATORY]*n_exc + [Neurons.PV]*n_pv + [Neurons.SOM]*n_som

def sample_exc_inh_neurons(n_neurons: int, p_inh: float) -> list[Neurons]:
    n_inh = round(p_inh * n_neurons)
    n_exc = n_neurons - n_inh
    assert n_exc > 0, f"p_inh = {p_inh}, which is >= 1.0, so there would be no excitatory neurons left"
    return [Neurons.EXCITATORY]*n_exc + [Neurons.INHIBITORY]*n_inh

def sample_exc_neurons(n_neurons: int) -> list[Neurons]:
    return [Neurons.EXCITATORY]*n_neurons