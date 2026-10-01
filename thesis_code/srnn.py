import torch
import torch.nn as nn

from .network_components.layers import OutputLayer, StandardRecurrentLayer, SynapticRecurrentLayer

class SRNN(nn.Module):
    def __init__(self,
            rec_layers: list[StandardRecurrentLayer | SynapticRecurrentLayer],
            beta_out: float,
            n_classes: int,
        ) -> None:
        super().__init__()

        self.rec_layers = nn.ModuleList(rec_layers)
        self.out_layer = OutputLayer(rec_layers[-1].n_neurons, n_classes, beta_out)

    def reset(self) -> None:
        for layer in self.rec_layers:
            layer.reset()
        self.out_layer.reset()

    def forward(self, data: torch.Tensor, separate_timesteps: bool = True) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        self.reset()

        spk_outs, mem_outs = [], []
        hidden_spks = [[] for _ in range(len(self.rec_layers))]

        with nn.utils.parametrize.cached():
            if separate_timesteps:
                for step in range(data.size(0)):
                    x = data[step]
                    for i, layer in enumerate(self.rec_layers):
                        x = layer(x)
                        hidden_spks[i].append(x)

                    spk_out, mem_out = self.out_layer(x)
                    spk_outs.append(spk_out)
                    mem_outs.append(mem_out)

            else:
                x = data
                for i, layer in enumerate(self.rec_layers):
                    x = layer.forward_sequence(x)
                    hidden_spks.append(x)
                spk_outs, mem_outs = [], []
                for t in range(x.size(0)):                   # OutputLayer unchanged for now
                    s, m = self.out_layer(x[t])
                    spk_outs.append(s); mem_outs.append(m)

        return torch.stack(spk_outs, dim=0), torch.stack(mem_outs, dim=0), [torch.stack(spks_h, dim=0) for spks_h in hidden_spks]