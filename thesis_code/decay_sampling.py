import torch

def tau_to_beta(dt: float, tau: float | torch.Tensor) -> torch.Tensor:
    """
    Computes 'beta = exp(-dt / tau)', implemented from:
    Friedemann Zenke, Tim P. Vogels; The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks. Neural Comput 2021; 33 (4): 899-925. doi: https://doi.org/10.1162/neco_a_01367
    """
    return torch.exp(-dt / torch.as_tensor(tau))

def sample_heterogeneous_decays_uniform(dt: float, n_neurons: int, tau_upper: float, tau_lower: float, device=None) -> torch.Tensor:
    taus = torch.rand(n_neurons, device=device) * (tau_upper - tau_lower) + tau_lower
    return tau_to_beta(dt, taus)

def sample_heterogeneous_decays_normal(dt: float, n_neurons: int, tau_mean: float, tau_std: float, device=None) -> torch.Tensor:
    taus = (torch.randn(n_neurons, device=device) * tau_std + tau_mean).clamp(min=dt)        # ensure no negative time constants get sampled
    return tau_to_beta(dt, taus)