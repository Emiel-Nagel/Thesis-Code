import torch
from snntorch.surrogate import fast_sigmoid
from typing import Any

from .srnn_base import SRNN
from ...network_components.layers import SynapticRecurrentLayer

class SynapticSRNN(SRNN):
    def __init__(self, 
            forward_matrices: list[torch.Tensor],
            recurrent_matrices: list[torch.Tensor],
            alphas: list[torch.Tensor | float],
            betas: list[torch.Tensor | float],
            beta_out: float,
            n_classes: int,
            spike_grad: Any = fast_sigmoid(slope=25),
            fully_learnable: bool = True,
            record: bool = True,
    ) -> None:
        super().__init__(
            forward_matrices,
            recurrent_matrices,
            alphas=alphas,
            betas=betas,
            beta_out=beta_out,
            n_classes=n_classes,
            record=record,
        )
        
        for fm, rm, alpha, beta in zip(forward_matrices, recurrent_matrices, alphas, betas):
            assert fm.shape[-1] == rm.shape[0], \
                f"forward matrix outputs {fm.shape[-1]} neurons, recurrent matrix has {rm.shape[0]}"

            self.rec_layers.append(SynapticRecurrentLayer(
                forward_matrix=fm,
                recurrent_matrix=rm,
                alpha=alpha,
                beta=beta,
                spike_grad=spike_grad,
                fully_learnable=fully_learnable,
                record=record,
            ))

    # def forward_net(self, data: torch.Tensor) -> tuple[list, list, list[list]]:
    #     spk_outs, mem_outs = [], []
    #     hidden_spks = [[] for _ in range(len(self.rec_layers))]

    #     for step in range(data.size(0)):
    #         x = data[step]
    #         for i, layer in enumerate(self.rec_layers):
    #             x = layer(x)
    #             hidden_spks[i].append(x)

    #         spk_out, mem_out = self.out_layer(x)
    #         spk_outs.append(spk_out)
    #         mem_outs.append(mem_out)

    #     return spk_outs, mem_outs, hidden_spks