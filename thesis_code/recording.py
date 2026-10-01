import torch
import torch.nn as nn

class Recorder:
    def to_state(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_state(cls, state: dict):
        obj = cls.__new__(cls)        # skips __init__, so no constructor args needed
        obj.__dict__.update(state)    # restores num_layers etc. as well
        return obj

class SpikeRecorder(Recorder):
    def __init__(self, num_hidden_layers: int) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_layers = num_hidden_layers + 2
        self.clear()

    def clear(self) -> None:
        self.recordings = {"Spikes in": []}
        self.recordings.update({f"Spikes hidden {i}": [] for i in range(self.num_hidden_layers)})
        self.recordings["Spikes out"] = []
        self.targets = []
        self.num_iterations = 0

    def _clean(self, data: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return data.detach().to(dtype).cpu().clone()

    def record(self, spk_ins: torch.Tensor, hidden_spks: list[torch.Tensor], spk_outs: torch.Tensor, targets: torch.Tensor) -> None:        
        self.recordings["Spikes in"].append(self._clean(spk_ins[:, 0, :], torch.uint8))
        self.recordings["Spikes out"].append(self._clean(spk_outs[:, 0, :], torch.bool))
        for i, spks in enumerate(hidden_spks):
            self.recordings[f"Spikes hidden {i}"].append(self._clean(spks[:, 0, :], torch.bool))

        self.targets.append(int(targets[0]))
        self.num_iterations += 1

    def get_spikes_and_targets(self, iteration_i: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        recordings = {name: content[iteration_i] for name, content in self.recordings.items()}
        return recordings, self.targets[iteration_i]

    def get_layer_spikes(self, iteration_i: int, layer_i: int) -> torch.Tensor:
        layer_name = list(self.recordings)[layer_i]
        return self.recordings[layer_name][iteration_i]

class PerformanceRecorder(Recorder):
    def __init__(self) -> None:
        self.loss_rec = []
        self.acc_rec = []

    def record(self, loss_val: float, acc: float) -> None:
        self.loss_rec.append(loss_val)
        self.acc_rec.append(acc)

    def get_performance(self) -> tuple[list, list]:
        return self.loss_rec, self.acc_rec

class GradientRecorder(Recorder):
    def __init__(self) -> None:
        self.max_grad_rec = []
        self.avg_grad_rec = []

    def record(self, net: nn.Module) -> None:
        grads = {name: p.grad.detach().clone()
            for name, p in net.named_parameters() if p.grad is not None}
        if len(grads) > 0:
            flat_grads = torch.cat([t.flatten() for t in grads.values()])
            self.max_grad_rec.append(flat_grads.max(dim=0).values.item())
            self.avg_grad_rec.append(flat_grads.mean(dim=0).item())
        else:
            self.max_grad_rec.append(0)
            self.avg_grad_rec.append(0)

    def get_grads(self) -> tuple[list, list]:
        return self.max_grad_rec, self.avg_grad_rec