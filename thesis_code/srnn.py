import torch
import torch.nn as nn

from dataclasses import dataclass

from .neuron_emulation.decays import LayerBetas

from .network_components.linears import DaleslawLinear
from .network_components.layers import OutputLayer, StandardRecurrentLayer, SynapticRecurrentLayer, RecurrentLayerConfig

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

    def get_weights(self) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        w_rec = [layer.get_weights() for layer in self.rec_layers]
        w_rec.append((self.out_layer.get_weights(), None))
        return w_rec

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        spk_outs, mem_outs = [], []
        hidden_spks = [[] for _ in range(len(self.rec_layers))]
        hidden_mems = [[] for _ in range(len(self.rec_layers))]

        for step in range(data.size(0)):
            x = data[step]
            for i, layer in enumerate(self.rec_layers):
                x, m = layer(x)
                hidden_spks[i].append(x)
                hidden_mems[i].append(m)

            spk_out, mem_out = self.out_layer(x)
            spk_outs.append(spk_out)
            mem_outs.append(mem_out)

        return torch.stack(spk_outs, dim=0), torch.stack(mem_outs, dim=0), [torch.stack(spks_h, dim=0) for spks_h in hidden_spks], [torch.stack(mem_h, dim=0) for mem_h in hidden_mems]

    @torch.compile
    def forward_sequence(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:        
        hidden_spks = []
        hidden_mems = []

        x_seq = data
        for layer in self.rec_layers:
            x_seq, m_seq = layer.forward_sequence(x_seq)
            hidden_spks.append(x_seq)
            hidden_mems.append(m_seq)
        spk_outs, mem_outs = self.out_layer.forward_sequence(x_seq)

        return spk_outs, mem_outs, hidden_spks, hidden_mems

    def apply_daleslaw(self) -> None:
        for m in self.modules():
            if isinstance(m, DaleslawLinear):
                m.apply_daleslaw()

    def constrain_betas(self) -> None:
        for m in self.modules():
            if isinstance(m, LayerBetas):
                m.clamp_betas()

@dataclass
class SRNNConfig:
    hidden_layers: list[RecurrentLayerConfig]
    add_control_net: bool
    use_daleslaw: bool
    use_connectivity: bool
    refractory_period: int
    tau_out: float  # ms