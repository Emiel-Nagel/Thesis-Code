import torch.nn as nn
from torch import Tensor

import snntorch as snn

class MaskedLinear(nn.Module):
    def __init__(self, con_matrix: Tensor) -> None:
        self.linear = nn.Linear(*con_matrix.shape)
        self.register_buffer('con_matrix', con_matrix)

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.linear.weight * self.mask, self.linear.bias)

class SparseRLeaky(snn.RLeaky):
    def __init__(self, *args, rec_matrix: Tensor, **kwargs) -> None:
        assert rec_matrix.shape[0] == rec_matrix.shape[1], \
            f"expected a square matrix, got shape {tuple(rec_matrix.shape)}"
        
        super().__init__(*args, **kwargs)
        self.n_neurons = rec_matrix.shape[0]
        self.recurrent = MaskedLinear(rec_matrix)
        if not self.learn_recurrent:
            for param in self.recurrent.parameters():
                param.requires_grad = False