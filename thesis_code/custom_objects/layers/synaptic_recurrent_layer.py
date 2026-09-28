import torch
import torch.nn as nn

from ..neurons import MaskedRSynaptic
from ..building_blocks import MaskedKaimingLinear

class SynapticRecurrentLayer(nn.Module):
    def __init__(self,
        forward_matrix: torch.Tensor,
        recurrent_matrix: torch.Tensor,
        alpha: torch.Tensor | float,
        beta: torch.Tensor | float,
        fully_learnable: bool = True,
        record: bool = True,
    ) -> None:
        n_neurons = recurrent_matrix.shape[0]

        assert forward_matrix.shape[1] == n_neurons, \
            f"forward_matrix outputs {forward_matrix.shape[1]} neurons, recurrent_matrix has {n_neurons}"

        super().__init__()
        self.weights = MaskedKaimingLinear(forward_matrix)
        self.neurons = MaskedRSynaptic(
            alpha=alpha,
            beta=beta,
            linear_features=n_neurons,
            init_hidden=False,
            learn_alpha=fully_learnable,
            learn_beta=fully_learnable,
            rec_matrix=recurrent_matrix,
        )
        self.spk, self.syn, self.mem = self.neurons.init_rsynaptic()
        self.spk_rec, self.mem_rec = [], []
        self.record = record

    def reset(self) -> None:
        self.spk, self.syn, self.mem = self.neurons.init_rsynaptic()
        self.spk_rec, self.mem_rec = [], []

    def get_recordings(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.record, "recording is disabled (record=False)"
        assert self.spk_rec, "no recordings yet; run at least one forward step"
        return torch.stack(self.spk_rec), torch.stack(self.mem_rec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weights(x)
        self.spk, self.syn, self.mem = self.neurons(x, self.spk, self.syn, self.mem)
        if self.record:
            self.spk_rec.append(self.spk)
            self.mem_rec.append(self.mem)
        return self.spk