import torch
import torch.nn as nn
import numpy as np

class RecorderBase:
    def to_state(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_state(cls, state: dict):
        obj = cls.__new__(cls)        # skips __init__, so no constructor args needed
        obj.__dict__.update(state)    # restores num_layers etc. as well
        return obj

class Recorder(RecorderBase):
    def __init__(self)

class SpikeRecorder(RecorderBase):
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

class PerformanceRecorder(RecorderBase):
    def __init__(self) -> None:
        self.loss_rec: list[list[float]] = []   # will store nested lists for per-trial separation
        self.acc_rec: list[list[float]] = []

    def add_trial(self) -> None:
        self.loss_rec.append([])
        self.acc_rec.append([])

    def record(self, loss_val: float, acc: float) -> None:
        self.loss_rec[-1].append(loss_val)
        self.acc_rec[-1].append(acc)

    def get_performance(self) -> tuple[np.ndarray, np.ndarray]:
        loss_rec = np.array(self.loss_rec, dtype=float)
        acc_rec = np.array(self.acc_rec, dtype=float)
        return loss_rec, acc_rec

class GradientRecorder(RecorderBase):
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

class WeightRecorder(Recorder):
    def __init__(self, num_hidden_layers: int) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.clear()

    def clear(self) -> None:
        self.w_rec = {
            i : {"forward": [], "recurrent": []}
            for i in range(self.num_hidden_layers)
        }
        self.w_rec[self.num_hidden_layers + 1] = {"forward": []}

    def record(self, weights: list[tuple[torch.Tensor, torch.Tensor | None]]) -> None:
        for layer_i, (w_forward, w_recurrent) in enumerate(weights):
            self.w_rec[layer_i]["forward"].append(w_forward)
            if w_recurrent is not None:
                self.w_rec[layer_i]["recurrent"].append(w_recurrent)

    def get_weights(self) -> dict[int, dict[str, list[torch.Tensor]]]:
        return self.w_rec

    def get_layer_weights(self, layer_i: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        layer = self.w_rec[layer_i]
        return layer["forward"], layer["recurrent"]