import torch
import torch.nn as nn
from typing import Any

from ..neurons import MaskedRSynaptic
from ..linears import MaskedKaimingLinear

class SynapticRecurrentLayer(nn.Module):
    def __init__(self,
        forward_matrix: torch.Tensor,
        recurrent_matrix: torch.Tensor,
        alpha: torch.Tensor | float,
        beta: torch.Tensor | float,
        spike_grad: Any,
        fully_learnable: bool = True,
    ) -> None:
        self.n_neurons = recurrent_matrix.shape[0]

        assert forward_matrix.shape[1] == self.n_neurons, \
            f"forward_matrix outputs {forward_matrix.shape[1]} neurons, recurrent_matrix has {self.n_neurons}"

        super().__init__()
        self.weights = MaskedKaimingLinear(forward_matrix)
        self.neurons = MaskedRSynaptic(
            alpha=alpha,
            beta=beta,
            linear_features=self.n_neurons,
            init_hidden=False,
            spike_grad=spike_grad,
            learn_alpha=fully_learnable,
            learn_beta=fully_learnable,
            rec_matrix=recurrent_matrix,
        )
        self.spk, self.syn, self.mem = self.neurons.init_rsynaptic()

    def reset(self) -> None:
        self.spk, self.syn, self.mem = self.neurons.init_rsynaptic()
        self.neurons.reset_recurrent_weights()
        self.weights.reset()

    def get_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weights.get_weights(), self.neurons.get_recurrent_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weights(x)
        self.spk, self.syn, self.mem = self.neurons(x, self.spk, self.syn, self.mem)
        return self.spk

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weights(x)
        spks = []

        for t in range(x.size(0)):
            self.spk, self.syn, self.mem = self.neurons(x[t], self.spk, self.syn, self.mem)
            spks.append(self.spk)

        return torch.stack(spks, dim=0)