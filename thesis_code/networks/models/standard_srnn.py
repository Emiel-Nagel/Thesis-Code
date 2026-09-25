import torch
import torch.nn as nn

from ...custom_objects.layers import StandardRecurrentLayer, OutputLayer
from ...custom_objects import Recorder

class StandardSRNN(nn.Module):
    def __init__(self, 
            forward_matrices: list[torch.Tensor],
            recurrent_matrices: list[torch.Tensor],
            n_classes: int,
            beta_upper_bound: float = 0.96,
            beta_lower_bound: float = 0.69,
            fully_learnable: bool = True,
            record: bool = True,
            device = None,
    ) -> None:
        super().__init__()
        modules = [nn.Flatten()]

        assert len(forward_matrices) == len(recurrent_matrices), \
            "unequal number forward- and recurrent matrices"

        for fm_prev, fm_next in zip(forward_matrices, forward_matrices[1:]):
            assert fm_prev.shape[1] == fm_next.shape[0], \
                f"layer output {fm_prev.shape[1]} doesn't match next layer input {fm_next.shape[0]}"
        
        for fm, rm in zip(forward_matrices, recurrent_matrices):
            assert fm.shape[-1] == rm.shape[0], \
                f"forward matrix outputs {fm.shape[-1]} neurons, recurrent matrix has {rm.shape[0]}"

            modules.append(StandardRecurrentLayer(
                forward_matrix=fm,
                recurrent_matrix=rm,
                beta_upper_bound=beta_upper_bound,
                beta_lower_bound=beta_lower_bound,
                fully_learnable=fully_learnable,
                record=record,
                device=device,
            ))

        modules.append(OutputLayer(forward_matrices[-1].shape[-1], n_classes))
        self.net = nn.Sequential(*modules)
        self.recorder = Recorder(num_layers=len(recurrent_matrices))
        self.record = record

    def reset(self) -> None:
        for layer in self.net:
            if isinstance(layer, (StandardRecurrentLayer, OutputLayer)):
                layer.reset()

    def record_layers(self) -> None:
        rec_layers = [l for l in self.net if isinstance(l, StandardRecurrentLayer)]
        for i, layer in enumerate(rec_layers):
            self.recorder.add_layer_recordings(i, *layer.get_recordings())

    def get_recorder(self) -> Recorder:
        assert self.record, "recording is disabled (record=False)"
        return self.recorder

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spk_outs, mem_outs = [], []
        self.reset()

        for step in range(data.size(0)):
            spk_out, mem_out = self.net(data[step])
            spk_outs.append(spk_out)
            mem_outs.append(mem_out)

        if self.record:
            self.record_layers()

        return torch.stack(spk_outs, dim=0), torch.stack(mem_outs, dim=0)