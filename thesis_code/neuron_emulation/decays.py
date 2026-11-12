import torch
from scipy import stats

from ..parameter_sampling import tau_to_beta, sample_heterogeneous_uniform

from .neurons import Neurons

"""
Decay ranges based on:
https://neuroelectro.org/datatable/neuron/46626.html

Excitatory:     10.5 + 5.1 SD
PV:             4.0  + 1.8 SD
SOM:            11.8 + 4.6 SD
"""

def _get_uniform_bounds(mean: float, std: float, confidence: float = 0.80) -> tuple[float, float]:
    low, high = stats.norm.interval(confidence, loc=mean, scale=std)
    low = max(low, 1e-8)
    return (low, high)

decay_ranges: dict[str, tuple[float, float]] = {
    "EXCITATORY":   _get_uniform_bounds(mean=10.5, std=5.1),
    "PV":           _get_uniform_bounds(mean= 4.0, std=1.8),
    "SOM":          _get_uniform_bounds(mean=11.8, std=4.6)
}

def get_membrane_decays_ee_pv_som(layer: list[Neurons], dt: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau_lower_bounds = torch.tensor([decay_ranges[neuron.name][0] for neuron in layer], device=device)
    tau_upper_bounds = torch.tensor([decay_ranges[neuron.name][1] for neuron in layer], device=device)
    taus = sample_heterogeneous_uniform(len(layer), tau_upper_bounds, tau_lower_bounds, device)
    return tau_to_beta(dt, taus), tau_to_beta(dt, tau_lower_bounds), tau_to_beta(dt, tau_upper_bounds)