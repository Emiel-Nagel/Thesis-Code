import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Literal, Sequence

class RecorderBase:
    def __init__(self, *rec_names: str) -> None:
        self.rec_names = rec_names
        self.recordings: dict[str, list[float | torch.Tensor]] = {n: [] for n in rec_names}

    def _record(self, **metrics: float | torch.Tensor) -> None:
        if not metrics:
            raise TypeError("_record() requires at least one keyword argument")
        unknown = metrics.keys() - self.recordings.keys()
        if unknown:
            raise KeyError(f"unknown recordings {sorted(unknown)}, expected {self.rec_names}")
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self.recordings[name].append(value)

    def clear(self) -> None:
        for data in self.recordings.values():
            data.clear()

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in self.recordings.items():
            if not data:
                continue
            if isinstance(data[0], torch.Tensor):
                npy_array = torch.stack(data).numpy()
            else:
                npy_array = np.asarray(data, dtype=np.float32)
            np.save(output_dir / name, npy_array)

class GradientRecorder(RecorderBase):
    def __init__(self) -> None:
        super().__init__("max_grad", "avg_grad")

    def record(self, *, net: nn.Module, **_) -> None:
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if grads:
            n = sum(g.numel() for g in grads)
            max_grad = torch.stack([g.max() for g in grads]).max().item()
            avg_grad = (torch.stack([g.sum() for g in grads]).sum() / n).item()
        else:
            max_grad = avg_grad = 0.0
        self._record(max_grad_rec=max_grad, avg_grad_rec=avg_grad)




RecorderOptions = Literal["performance", "gradients", "weights", "spikes"]
rec_map = {
    "performance": PerformanceRecorder,
    "gradients": GradientRecorder,
    "weights": WeightRecorder,
    "spikes": SpikeRecorder,
}
class Recorder:
    def __init__(self, output_dir: Path, num_hidden_layers: int, options: Sequence[RecorderOptions]) -> None:
        self.output_dir = output_dir    # = output_folder/seed_(n)
        self.num_hidden_layers = num_hidden_layers
        self.iter_recorders = [rec_map[option]() for option in options
                               if option in ["gradients"]]
        self.epoch_recorders = [rec_map[option]() for option in options
                               if option in ["performance", "weights", "spikes"]]

    def record_iteration(self, **kwargs) -> None:
        for r in self.iter_recorders:
            r.record(**kwargs)

    def record_epoch(self, **kwargs) -> None:
        for r in self.epoch_recorders:
            r.record(**kwargs)

    def clear(self) -> None:
        for r in self.iter_recorders + self.epoch_recorders:
            r.clear()

    def save_recordings(self) -> None:
        for r in self.iter_recorders + self.epoch_recorders:
            r.save(self.output_dir)




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