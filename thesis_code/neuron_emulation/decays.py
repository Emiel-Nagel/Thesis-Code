import torch
import torch.nn as nn
from scipy import stats

from ..parameter_sampling import tau_to_beta, sample_heterogeneous_uniform

from .neurons import Neuron

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

def get_membrane_decays_ee_pv_som(neurons: list[Neuron], dt: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau_lower_bounds = torch.tensor([decay_ranges[neuron.name][0] for neuron in neurons], device=device)
    tau_upper_bounds = torch.tensor([decay_ranges[neuron.name][1] for neuron in neurons], device=device)
    taus = sample_heterogeneous_uniform(len(neurons), tau_upper_bounds, tau_lower_bounds, device)
    return tau_to_beta(dt, taus), tau_to_beta(dt, tau_lower_bounds), tau_to_beta(dt, tau_upper_bounds)

class FloatingBeta(nn.Module):
    def __init__(self, taus: torch.Tensor, dt: float = 0.001) -> None:
        super().__init__()
        self.register_buffer("taus", taus)
        self.dt = dt
        self.register_parameter("log_tau_scale", nn.Parameter(torch.zeros(()), requires_grad=True))     # having log scale means it can't go negative, which is necessary for the decay to be valid. Inits at 1.0

    def forward(self) -> torch.Tensor:
        return tau_to_beta(self.dt, self.taus * self.log_tau_scale.exp())

def get_floating_beta_ee_pv_som(neurons: list[Neuron], dt: float, device: torch.device) -> FloatingBeta:
    tau_lower_bounds = torch.tensor([decay_ranges[neuron.name][0] for neuron in neurons], device=device)
    tau_upper_bounds = torch.tensor([decay_ranges[neuron.name][1] for neuron in neurons], device=device)
    taus = sample_heterogeneous_uniform(len(neurons), tau_upper_bounds, tau_lower_bounds, device)
    return FloatingBeta(taus, dt=dt).to(device)