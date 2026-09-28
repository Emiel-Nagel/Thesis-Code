import torch.nn as nn
import torch

class KaimingLinear(nn.Linear):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__(n_in, n_out, bias=False)
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

class MaskedLinear(nn.Module):
    def __init__(self, con_matrix: torch.Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(*con_matrix.shape, bias=False)      # clamp to higher than 0 + small value, cannot become negative
        self.register_buffer('con_matrix', con_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.linear.weight * self.con_matrix.T, self.linear.bias)

class MaskedKaimingLinear(nn.Module):
    def __init__(self, con_matrix: torch.Tensor) -> None:
        super().__init__()
        self.linear = KaimingLinear(*con_matrix.shape)
        self.register_buffer('con_matrix', con_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.linear.weight * self.con_matrix.T, self.linear.bias)

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