import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class RecorderBase:
    def __init__(self, *rec_names: str, subfolder_name: str) -> None:
        self.rec_names = rec_names
        self.recordings: dict[str, list[float | torch.Tensor]] = {n: [] for n in rec_names}
        self.subfolder_name = subfolder_name

    def record(self, **kwargs) -> None:
        raise NotImplementedError

    def _record(self, **metrics: float | torch.Tensor) -> None:
        if not metrics:
            raise TypeError("_record() requires at least one keyword argument")
        unknown = metrics.keys() - self.recordings.keys()
        if unknown:
            raise KeyError(f"unknown recordings {sorted(unknown)}, expected {self.rec_names}")
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().to("cpu", copy=True)
            self.recordings[name].append(value)

    def clear(self) -> None:
        for data in self.recordings.values():
            data.clear()

    def save(self, output_dir: Path) -> None:
        def _save_compressed(path: Path, npy_array: np.ndarray, lengths: np.ndarray | None) -> None:
            meta = {"lengths": lengths} if lengths is not None else {}
            if npy_array.dtype == np.bool_ or (npy_array.dtype == np.uint8 and npy_array.max() <= 1):
                meta["n_cols"] = npy_array.shape[-1]
                npy_array = np.packbits(npy_array.astype(bool), axis=-1)
            np.savez_compressed(path, data=npy_array, **meta)

        output_dir = output_dir / self.subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in self.recordings.items():
            if not data:
                continue
            try:
                lengths = None
                if isinstance(data[0], torch.Tensor):
                    if any(t.shape != data[0].shape for t in data):     # if any tensors have unequal shapes, pad them
                        lengths = np.array([t.shape[0] for t in data])
                        npy_array = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0).numpy()
                    else:
                        npy_array = torch.stack(data).numpy()
                else:
                    npy_array = np.asarray(data, dtype=np.float32)
                _save_compressed(output_dir / name, npy_array, lengths)
            except Exception as e:
                raise RuntimeError(f"Could not save '{name}'") from e

class PerformanceRecorder(RecorderBase):
    def __init__(self) -> None:
        super().__init__("loss", "acc", subfolder_name="performance")

    def record(self, *, loss_val: float, acc: float, **_) -> None:
        self._record(loss=loss_val, acc=acc)

class GradientRecorder(RecorderBase):
    def __init__(self) -> None:
        super().__init__("max_grad", "avg_grad", subfolder_name="gradients")

    def record(self, *, net: nn.Module, **_) -> None:
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if grads:
            n = sum(g.numel() for g in grads)
            max_grad = torch.stack([g.max() for g in grads]).max().item()
            avg_grad = (torch.stack([g.sum() for g in grads]).sum() / n).item()
        else:
            max_grad = avg_grad = 0.0
        self._record(max_grad=max_grad, avg_grad=avg_grad)

class WeightRecorder(RecorderBase):
    def __init__(self, num_hidden_layers: int) -> None:
        rec_names = [n for i in range(num_hidden_layers) for n in (f"weights_forward_{i}", f"weights_recurrent_{i}")]
        rec_names += [f"weights_forward_{num_hidden_layers}"]
        super().__init__(*rec_names, subfolder_name="weights")

    def record(self, *, weights: list[tuple[torch.Tensor, torch.Tensor | None]], **_) -> None:
        metrics: dict[str, torch.Tensor] = {}
        for i, (w_forward, w_recurrent) in enumerate(weights):
            metrics[f"weights_forward_{i}"] = w_forward.to(torch.float16)
            if w_recurrent is not None:
                metrics[f"weights_recurrent_{i}"] = w_recurrent.to(torch.float16)
        self._record(**metrics)

class SpikeRecorder(RecorderBase):
    def __init__(self, num_hidden_layers: int) -> None:
        rec_names = ["spikes_in", "mems_out", "targets"]
        rec_names += [f"spikes_hidden_{i}" for i in range(num_hidden_layers)]
        super().__init__(*rec_names, subfolder_name="spikes")

    def record(self, *, spk_ins: torch.Tensor, hidden_spks: list[torch.Tensor], mem_outs: torch.Tensor, targets: torch.Tensor, **_) -> None:
        """
        Consistently only records one item in the batch, and converts dtypes to save storage space.
        """
        metrics: dict[str, torch.Tensor] = {f"spikes_hidden_{i}": spks[:, 0, :].to(torch.bool) 
                                            for i, spks in enumerate(hidden_spks)}
        self._record(
            spikes_in=spk_ins[:, 0, :].to(torch.uint8),
            mems_out=mem_outs[:, 0, :].to(torch.float16),
            targets=targets[0],
            **metrics,
        )

class Recorder:
    def __init__(self, num_hidden_layers: int, options: dict[str, bool]) -> None:
        self.iter_recorders: list[RecorderBase] = []
        self.epoch_recorders: list[RecorderBase] = []

        if options.get("record_gradients"):
            self.iter_recorders.append(GradientRecorder())
        if options.get("record_performance"):
            self.epoch_recorders.append(PerformanceRecorder())
        if options.get("record_weights"):
            self.epoch_recorders.append(WeightRecorder(num_hidden_layers))
        if options.get("record_spikes"):
            self.epoch_recorders.append(SpikeRecorder(num_hidden_layers))

    def record_iteration(self, **kwargs) -> None:
        for r in self.iter_recorders:
            r.record(**kwargs)

    def record_epoch(self, **kwargs) -> None:
        for r in self.epoch_recorders:
            r.record(**kwargs)

    def clear(self) -> None:
        for r in self.iter_recorders + self.epoch_recorders:
            r.clear()

    def save(self, output_dir: Path) -> None:
        for r in self.iter_recorders + self.epoch_recorders:
            r.save(output_dir)