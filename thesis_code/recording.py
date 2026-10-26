import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

_pool = ThreadPoolExecutor(max_workers=1)

def gpu_to_cpu_async(t: torch.Tensor) -> Future:
    # ensures gpu doesn't have to sync with cpu when copying a tensor to cpu
    t = t.detach()
    return _pool.submit(lambda: t.to("cpu"))  # returns a Future

class RecorderBase:
    def __init__(self, *rec_names: str, data_name: str) -> None:
        self.rec_names = rec_names
        self.recordings: dict[str, list[float | torch.Tensor]] = {n: [] for n in rec_names}
        self.data_name = data_name

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
                value = gpu_to_cpu_async(value)
            self.recordings[name].append(value)

    def _resolve(self) -> None:
        # waits for all tensors to be moved to the cpu, then continues
        for name, data in self.recordings.items():
            self.recordings[name] = [d.result() if isinstance(d, Future) else d for d in data]

    def clear(self) -> None:
        for data in self.recordings.values():
            data.clear()

    def save(self, output_dir: Path) -> None:
        self._resolve()
        try:
            output_data: dict[str, np.ndarray] = {}
            for name, data in self.recordings.items():
                if not data:
                    continue

                if isinstance(data[0], torch.Tensor):
                    if any(t.shape != data[0].shape for t in data):     # if any tensors have unequal shapes, pad them
                        output_data[f"{name}-lengths"] = np.array([t.shape[0] for t in data])
                        npy_array = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0).numpy()
                    else:
                        npy_array = torch.stack(data).numpy()
                else:
                    npy_array = np.asarray(data, dtype=np.float32)

                if npy_array.dtype == np.bool_ or (npy_array.dtype == np.uint8 and npy_array.max() <= 1):
                    output_data[f"{name}-n_cols"] = npy_array.shape[-1]
                    npy_array = np.packbits(npy_array.astype(bool), axis=-1)
                output_data[name] = npy_array

            np.savez_compressed(output_dir / self.data_name, **output_data)
        except Exception as e:
            raise RuntimeError(f"Could not save '{name}'") from e

class PerformanceRecorder(RecorderBase):
    def __init__(self) -> None:
        super().__init__("loss_train", "acc_train", "loss_test", "acc_test", data_name="performance")

    def record(self, *, loss_train: float, acc_train: float, loss_test: float, acc_test: float, **_) -> None:
        self._record(loss_train=loss_train, acc_train=acc_train, loss_test=loss_test, acc_test=acc_test)

class GradientRecorder(RecorderBase):
    def __init__(self) -> None:
        super().__init__("max_grad", "avg_grad", data_name="gradients")

    def record(self, *, net: nn.Module, **_) -> None:
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if grads:
            n = sum(g.numel() for g in grads)
            max_grad = torch.stack(torch._foreach_norm(grads, float("inf"))).max()
            avg_grad = torch.stack(torch._foreach_norm(grads, 1)).sum() / n
        else:
            max_grad = avg_grad = torch.zeros(())
        self._record(max_grad=max_grad, avg_grad=avg_grad)

class WeightRecorder(RecorderBase):
    def __init__(self, num_hidden_layers: int) -> None:
        rec_names = [n for i in range(num_hidden_layers) for n in (f"weights_forward_{i}", f"weights_recurrent_{i}")]
        rec_names += [f"weights_forward_{num_hidden_layers}"]
        super().__init__(*rec_names, data_name="weights")

    def record(self, *, weights: list[tuple[torch.Tensor, torch.Tensor | None]], **_) -> None:
        metrics: dict[str, torch.Tensor] = {}
        for i, (w_forward, w_recurrent) in enumerate(weights):
            metrics[f"weights_forward_{i}"] = w_forward.to(torch.float16, copy=True)
            if w_recurrent is not None:
                metrics[f"weights_recurrent_{i}"] = w_recurrent.to(torch.float16, copy=True)
        self._record(**metrics)

class SpikeRecorder(RecorderBase):
    def __init__(self, num_hidden_layers: int) -> None:
        rec_names = ["spikes_in", "mems_out", "targets"]
        rec_names += [f"spikes_hidden_{i}" for i in range(num_hidden_layers)]
        super().__init__(*rec_names, data_name="spikes")

    def record(self, *, spk_ins: torch.Tensor, hidden_spks: list[torch.Tensor], mem_outs: torch.Tensor, targets: torch.Tensor, **_) -> None:
        """
        Consistently only records one item in the batch, and converts dtypes to save storage space.
        """
        metrics: dict[str, torch.Tensor] = {f"spikes_hidden_{i}": spks[:, 0, :].to(torch.bool) 
                                            for i, spks in enumerate(hidden_spks)}
        self._record(
            spikes_in=spk_ins[:, 0, :].to(torch.uint8, copy=True),
            mems_out=mem_outs[:, 0, :].to(torch.float16, copy=True),
            targets=targets[0].clone(),
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
        output_dir.mkdir(parents=True, exist_ok=True)
        for r in self.iter_recorders + self.epoch_recorders:
            r.save(output_dir)