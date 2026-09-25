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

def sample_heterogeneous_decays(n_neurons: int, upper_bound:float=0.96, lower_bound:float=0.69, device=None) -> torch.Tensor:
    decay_range = upper_bound - lower_bound
    return torch.rand(n_neurons, dtype=torch.float, device=device) * decay_range + lower_bound