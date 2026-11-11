import torch
import torch.nn as nn
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

def get_uniform_bounds(mean: float, std: float, confidence: float = 0.80) -> tuple[float, float]:
    low, high = stats.norm.interval(confidence, loc=mean, scale=std)
    low = max(low, 1e-8)
    return (low, high)

decay_ranges: dict[str, tuple[float, float]] = {
    "EXCITATORY":   get_uniform_bounds(mean=10.5, std=5.1),
    "PV":           get_uniform_bounds(mean= 4.0, std=1.8),
    "SOM":          get_uniform_bounds(mean=11.8, std=4.6)
}

class LayerBetas(nn.Module):
    def __init__(self, betas: torch.Tensor, upper_bounds: torch.Tensor, lower_bounds: torch.Tensor) -> None:
        super().__init__()
        self.betas = nn.Parameter(betas)
        self.register_buffer("lower_bounds", lower_bounds, persistent=False)
        self.register_buffer("upper_bounds", upper_bounds, persistent=False)

    def forward(self):
        return self.betas

    @torch.no_grad()
    def clamp_betas(self) -> None:
        self.betas.clamp_(self.lower_bounds, self.upper_bounds)

def get_membrane_decays(layer: list[Neurons], dt: float, device: torch.device) -> LayerBetas:
    tau_lower_bounds = torch.tensor([decay_ranges[neuron.name][0] for neuron in layer], device=device)
    tau_upper_bounds = torch.tensor([decay_ranges[neuron.name][1] for neuron in layer], device=device)
    taus = sample_heterogeneous_uniform(len(layer), tau_upper_bounds, tau_lower_bounds, device)
    return LayerBetas(
        betas=tau_to_beta(dt, taus),
        upper_bounds=tau_to_beta(dt, tau_upper_bounds),
        lower_bounds=tau_to_beta(dt, tau_lower_bounds),
    )