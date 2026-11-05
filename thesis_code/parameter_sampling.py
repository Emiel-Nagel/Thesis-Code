import torch
from typing import Optional, Literal
from dataclasses import dataclass, field

@dataclass
class SamplingConfig:
    sampling_mode: Literal["single_value", "normal", "uniform"]
    is_decay: bool

    # mode "single_value"
    tau_single_value: Optional[float] = None  # ms
    # mode "normal"
    tau_mean: Optional[float] = None  # ms
    tau_std: Optional[float] = None  # ms
    # mode "uniform"
    tau_upper: Optional[float] = None  # ms
    tau_lower: Optional[float] = None  # ms

    def __post_init__(self) -> None:
        required = {
            "single_value": {"tau_single_values"},
            "normal": {"tau_means", "tau_stds"},
            "uniform": {"tau_uppers", "tau_lowers"},
        }[self.sampling_mode]
        all_params = {"tau_single_values", "tau_means", "tau_stds", "tau_uppers", "tau_lowers"}

        set_params = {p for p in all_params if getattr(self, p) is not None}
        if set_params != required:
            raise ValueError(
                f"sampling_mode={self.sampling_mode!r} requires exactly {sorted(required)}, "
                f"but got {sorted(set_params)}"
            )

def tau_to_beta(dt: float, tau: float | torch.Tensor) -> torch.Tensor:
    """
    Computes 'beta = exp(-dt / tau)', implemented from:
    Friedemann Zenke, Tim P. Vogels; The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks. Neural Comput 2021; 33 (4): 899-925. doi: https://doi.org/10.1162/neco_a_01367
    """
    return torch.exp(-dt / torch.as_tensor(tau))

def sample_heterogeneous_uniform(n_neurons: int, tau_upper: float, tau_lower: float, device: torch.device) -> torch.Tensor:
    return torch.rand(n_neurons, device=device) * (tau_upper - tau_lower) + tau_lower

def sample_heterogeneous_normal(dt: float, n_neurons: int, tau_mean: float, tau_std: float, device: torch.device) -> torch.Tensor:
    return (torch.randn(n_neurons, device=device) * tau_std + tau_mean).clamp(min=dt)        # ensure no negative time constants get sampled

def sample_values(sampling_cfg: SamplingConfig, dt: float, n_neurons: int, device: torch.device) -> torch.Tensor:
    match sampling_cfg.sampling_mode:
        case "single_value":
            sampled = sampling_cfg.tau_single_value
        case "normal":
            sampled = sample_heterogeneous_normal(dt, n_neurons, tau_mean=sampling_cfg.tau_mean*dt, tau_std=sampling_cfg.tau_std*dt, device=device)
        case "uniform":
            sampled = sample_heterogeneous_uniform(dt, n_neurons, tau_upper=sampling_cfg.tau_upper*dt, tau_lower=sampling_cfg.tau_lower*dt, device=device)
    return tau_to_beta(dt, sampled) if sampling_cfg.is_decay else sampled