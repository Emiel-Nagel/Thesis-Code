import torch
import torch.nn as nn

from .network_components.layers import OutputLayer, StandardRecurrentLayer, SynapticRecurrentLayer

class SRNN(nn.Module):
    def __init__(self,
            rec_layers: list[StandardRecurrentLayer | SynapticRecurrentLayer] = [],
            out_layer: OutputLayer | None = None,
        ) -> None:
        super().__init__()

        self.rec_layers = nn.ModuleList(rec_layers)
        self.out_layer = out_layer

    def reset(self) -> None:
        for layer in self.rec_layers:
            layer.reset()
        self.out_layer.reset()

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        self.reset()

        spk_outs, mem_outs = [], []
        hidden_spks = [[] for _ in range(len(self.rec_layers))]

        for step in range(data.size(0)):
            x = data[step]
            for i, layer in enumerate(self.rec_layers):
                x = layer(x)
                hidden_spks[i].append(x)

            spk_out, mem_out = self.out_layer(x)
            spk_outs.append(spk_out)
            mem_outs.append(mem_out)

        return torch.stack(spk_outs, dim=0), torch.stack(mem_outs, dim=0), [torch.stack(spks_h, dim=0) for spks_h in hidden_spks]

    def forward_sequence(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        self.reset()
        
        hidden_spks = []

        x_seq = data
        for layer in self.rec_layers:
            x_seq = layer.forward_sequence(x_seq)
            hidden_spks.append(x_seq)
        spk_outs, mem_outs = self.out_layer.forward_sequence(x_seq)

        return spk_outs, mem_outs, hidden_spks
