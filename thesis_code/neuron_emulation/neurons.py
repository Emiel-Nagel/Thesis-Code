import numpy as np
from pathlib import Path
from typing import Literal
import json

from enum import Enum

class Neuron(Enum):
    EXCITATORY = 1
    PV = 2
    SOM = 3
    INHIBITORY = 4

def sample_exc_pv_som_neurons(n_neurons: int, p_pv: float, p_som: float) -> list[Neuron]:
    n_pv = round(p_pv * n_neurons)
    n_som = round(p_som * n_neurons)
    n_exc = n_neurons - n_pv - n_som
    assert n_exc > 0, f"p_pv + p_som = {p_pv + p_som}, which is >= 1.0, so there would be no excitatory neurons left"
    return [Neuron.EXCITATORY]*n_exc + [Neuron.PV]*n_pv + [Neuron.SOM]*n_som

def sample_exc_inh_neurons(n_neurons: int, p_inh: float) -> list[Neuron]:
    n_inh = round(p_inh * n_neurons)
    n_exc = n_neurons - n_inh
    assert n_exc > 0, f"p_inh = {p_inh}, which is >= 1.0, so there would be no excitatory neurons left"
    return [Neuron.EXCITATORY]*n_exc + [Neuron.INHIBITORY]*n_inh

def sample_exc_neurons(n_neurons: int) -> list[Neuron]:
    return [Neuron.EXCITATORY]*n_neurons

def sort_neuron_layer(layer: list[Neuron]) -> list[Neuron]:
    return sorted(layer, key=lambda n: n.value)

def get_neuron_map(neurons: list[Neuron]) -> dict[str, int]:
    return {member.name: member.value for member in set(neurons)}

def get_neuron_range(neurons: list[Neuron] | np.ndarray, neuron_value: Literal[1, 2, 3, 4]) -> tuple[int, int]:
    if not isinstance(neurons, np.ndarray):
        neurons = np.array([n.value for n in neurons], dtype=np.int8)
    
    return (
        np.min(np.where(neurons == neuron_value)),
        np.max(np.where(neurons == neuron_value))
    )

def get_neuron_ranges(neurons: list[Neuron] | np.ndarray, neuron_map: dict[str, int]) -> dict[str, tuple[int, int]]:
    if not isinstance(neurons, np.ndarray):
        neurons = np.array([n.value for n in neurons], dtype=np.int8)
    
    build_range = lambda value: (
        np.min(np.where(neurons == value)),
        np.max(np.where(neurons == value))
    )
    return {k: build_range(v) for k, v in neuron_map.items()}

def save_neurons(output_dir: Path, neurons: list[Neuron]) -> None:
    np.savez_compressed(
        output_dir / "neurons.npz",
        neurons=np.array([n.value for n in neurons], dtype=np.int8),
        neuron_map=json.dumps(get_neuron_map(neurons))
    )

def load_neurons(work_dir: Path) -> tuple[np.ndarray, dict[str, int], dict[str, tuple[int, int]]]:
    data = np.load(work_dir / "neurons.npz")
    neurons = data["neurons"]
    neuron_map = json.loads(str(data["neuron_map"]))
    neuron_ranges = get_neuron_ranges(neurons, neuron_map)
    return neurons, neuron_map, neuron_ranges