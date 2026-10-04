import torch.nn as nn
import torch

class MaskedLinear(nn.Module):
    def __init__(self, con_matrix: torch.Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(*con_matrix.shape, bias=False)      # clamp to higher than 0 + small value, cannot become negative
        self.register_buffer('con_matrix_t', con_matrix.T.contiguous())
        self._masked_weight = None

    def reset(self) -> None:
        self._masked_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._masked_weight is None:
            self._masked_weight = self.linear.weight * self.con_matrix_t    # we only compute this on the first forward, so it's reused until reset() is called after a learning step
        return nn.functional.linear(x, self._masked_weight)

class MaskedKaimingLinear(MaskedLinear):
    def __init__(self, con_matrix: torch.Tensor) -> None:
        super().__init__(con_matrix)
        nn.init.kaiming_uniform_(self.linear.weight, mode="fan_in", nonlinearity="relu")