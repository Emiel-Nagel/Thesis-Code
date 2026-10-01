import torch
import torch.nn as nn

from ...network_components.layers import OutputLayer, StandardRecurrentLayer, SynapticRecurrentLayer
# from ...recording import NeuronRecorder

class SRNN(nn.Module):
    def __init__(self,
            rec_layers: list[StandardRecurrentLayer, SynapticRecurrentLayer],
                 
            forward_matrices: list[torch.Tensor],
            recurrent_matrices: list[torch.Tensor],
            alphas: list[torch.Tensor | float] | None,
            betas: list[torch.Tensor | float],
            beta_out: float,
            n_classes: int,
            # record: bool,
        ) -> None:
        super().__init__()

        self.rec_layers = nn.ModuleList(rec_layers)
        # self.rec_layers = nn.ModuleList()

        # if alphas is not None:
        #     assert len(forward_matrices) == len(recurrent_matrices) == len(alphas) == len(betas), \
        #         "unequal number forward- and recurrent matrices, alphas and betas"
        # else:
        #     assert len(forward_matrices) == len(recurrent_matrices) == len(betas), \
        #         "unequal number forward- and recurrent matrices and betas"

        # for fm_prev, fm_next in zip(forward_matrices, forward_matrices[1:]):
        #     assert fm_prev.shape[1] == fm_next.shape[0], \
        #         f"layer output {fm_prev.shape[1]} doesn't match next layer input {fm_next.shape[0]}"

        self.out_layer = OutputLayer(forward_matrices[-1].shape[-1], n_classes, beta_out)
        # self.recorder = NeuronRecorder(num_layers=len(recurrent_matrices))
        # self.record = record

    def reset(self) -> None:
        for layer in self.rec_layers:
            layer.reset()
        self.out_layer.reset()

    # def record_layers(self) -> None:
    #     for i, layer in enumerate(self.rec_layers):
    #         self.recorder.add_layer_recordings(i, *layer.get_recordings())
    #     self.recorder.increment_iterations()

    # def get_recorder(self) -> NeuronRecorder:
    #     assert self.record, "recording is disabled (record=False)"
    #     return self.recorder

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
                    x = layer.forward_sequence(x)   # write this function
                    hidden_spks.append(x)
                spk_outs, mem_outs = [], []
                for t in range(x.size(0)):                   # OutputLayer unchanged for now
                    s, m = self.out_layer(x[t])
                    spk_outs.append(s); mem_outs.append(m)

        return torch.stack(spk_outs, dim=0), torch.stack(mem_outs, dim=0), [torch.stack(spks_h, dim=0) for spks_h in hidden_spks]

    # def forward_sequence(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    #     self.reset()

    #     spk_outs, mem_outs = [], []
    #     hidden_spks = [[] for _ in range(len(self.rec_layers))]

    #     with nn.utils.parametrize.cached():
    #         x = data
    #         for i, layer in enumerate(self.rec_layers):
    #             x = layer.forward_sequence(x)   # write this function
    #             hidden_spks.append(x)
    #         spk_outs, mem_outs = [], []
    #         for t in range(x.size(0)):                   # OutputLayer unchanged for now
    #             s, m = self.out_layer(x[t])
    #             spk_outs.append(s); mem_outs.append(m)

    #     return torch.stack(spk_outs, dim=0), torch.stack(mem_outs, dim=0), [torch.stack(spks_h, dim=0) for spks_h in hidden_spks]

    # def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    #     self.reset()

    #     # with nn.utils.parametrize.cached():
    #     spk_outs, mem_outs, hidden_spks = self.forward_net(data)

    #     # if self.record:
    #     #     self.record_layers()

    #     return torch.stack(spk_outs, dim=0), torch.stack(mem_outs, dim=0), [torch.stack(spks_h, dim=0) for spks_h in hidden_spks]

    # def forward_net(self, data: torch.Tensor) -> tuple[list, list, list[list]]:
    #     raise NotImplementedError

    # def forward_sequence(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    #     self.reset()

    #     spk_outs, mem_outs, hidden_spks = self.forward_sequence_net(data)

    #     if self.record:
    #         self.record_layers()

    #     return ...

    # def forward_net_sequence(self, data: torch.Tensor) -> tuple[list, list, list[list]]:
    #     raise NotImplementedError


