import torch.nn as nn
import torch

class KaimingLinear(nn.Linear):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__(n_in, n_out, bias=False)
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        # maybe scale initial weights by 1/sqrt(density)

class Mask(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask.to(torch.float32))

    def forward(self, w):
        return w * self.mask

class MaskedLinear(nn.Module):
    def __init__(self, con_matrix: torch.Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(*con_matrix.shape, bias=False)      # clamp to higher than 0 + small value, cannot become negative
        # self.register_buffer('con_matrix', con_matrix)
        nn.utils.parametrize.register_parametrization(self.linear, "weight", Mask(con_matrix.T.contiguous()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class MaskedKaimingLinear(nn.Module):
    def __init__(self, con_matrix: torch.Tensor) -> None:
        super().__init__()
        self.linear = KaimingLinear(*con_matrix.shape)
        # self.register_buffer('con_matrix', con_matrix)
        nn.utils.parametrize.register_parametrization(self.linear, "weight", Mask(con_matrix.T.contiguous()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)