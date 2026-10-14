import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# class GpuToCpuBuffer:
#     def __init__(self, chunk_size: int = 20000) -> None:
#         self._chunk_size = chunk_size
#         self._pin = torch.cuda.is_available()
#         self._index = 0
#         self._data_chunks: list[torch.Tensor] = []
#         self._lengths = []

#     def _add_new_chunk(self, data: torch.Tensor) -> torch.Tensor:
#         shape = data.shape
#         shape[0] = self._chunk_size
#         self._data_chunks.append(torch.empty((self._chunk_size, *data.shape), dtype=data.dtype, pin_memory=self._pin))

#     def append(self, data: torch.Tensor | float | int) -> None:
#         data = data.detach() if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)

#         if self._data_chunks is []:
#             self._add_new_chunk()

#         last_chunk = self._data_chunks[-1]
#         n_chunks, insert_idx = divmod(self._index, self._chunk_size)
#         remaining_timepoints = last_chunk.shape[0] - insert_idx

#         time_length = data.shape[0]
#         self._lengths.append(time_length)
#         self._index += time_length

#         n_fits, split_idx = divmod(remaining_timepoints, time_length)
#         if n_fits == 0:
#             last_chunk[insert_idx: insert_idx + split_idx].copy_(data[:split_idx], non_blocking=True)
#             self._add_new_chunk()
#             self._data_chunks[-1].copy_(data[split_idx:], non_blocking=True)
#         else:
#             last_chunk[insert_idx: insert_idx + time_length].copy_(data, non_blocking=True)

#     def get(self) -> tuple[np.ndarray, np.ndarray]:
#         if self._pin:
#             torch.cuda.synchronize()

#         all_data = torch.stack(self._data_chunks, dim=0)
#         split_recordings: list[torch.Tensor] = []
#         split_idx = 0
#         for length in self._lengths:
#             split_idx += length
#             split_recordings.append[all_data[split_idx: split_idx + length]]

#         if any(t.shape != split_recordings[0].shape for t in split_recordings):
#             npy_array = nn.utils.rnn.pad_sequence(split_recordings, batch_first=True, padding_value=0).numpy()
#         else:
#             npy_array = torch.stack(split_recordings).numpy()
#         return npy_array, np.array(self._lengths)
    



#     class TorchBuffer:
#         def __init__(self, chunk_size: int = 128) -> None:
#             self._chunk_size = chunk_size
#             self._pin = torch.cuda.is_available()
#             self._index = 0
#             self._data: torch.Tensor = None
#             self._lengths = []

#         def _create_new_chunk(self, data: torch.Tensor) -> torch.Tensor:
#             return torch.empty((self._chunk_size, *data.shape), dtype=data.dtype, device="cpu", pin_memory=self._pin)

#         def _pad_time_axis(self, data: torch.Tensor, k: int) -> torch.Tensor:
#             pad = (0, 0, 0, k)      # should pad the second-to-last dimension
#             return nn.functional.pad(data, pad)

#         def append(self, data: torch.Tensor) -> None:
#             if self._data is None:
#                 self._data = self._create_new_chunk(data)
#             else:
#                 if self._index >= self._data.shape[0]:
#                     self._data = torch.cat((self._data, self._create_new_chunk(data)))

#                 self._lengths.append(data.shape[0])
#                 time_diff = self._data.shape[1] - data.shape[0]     # assumes the second dim represents the time-domain
#                 if time_diff > 0:
#                     data = self._pad_time_axis(data, time_diff)
#                 if time_diff < 0:
#                     self._data = self._pad_time_axis(self._data, time_diff)

#                 self._data[self._index].copy_(data, non_blocking=True)
            
#             self._index += 1

#         def get(self) -> tuple[np.ndarray, np.ndarray]:                
#             return self._data[:, self._index, :].numpy, np.array(self._lengths)




#     def append(self, value: float | torch.Tensor) -> None:
#         if self._data is None:
#             if isinstance(value, torch.Tensor):
#                 self._data = [torch.empty_like(value, dtype=value.dtype, device="cpu", pin_memory=self._pin) for _ in self._chunk_size]
#                 self._data_type = torch.Tensor

#             elif isinstance(value, float):
#                 self._data = []
#                 self._data_type = float

#         assert isinstance(value, self._data_type), \
#             "Recorded value does not match type of previous values"

#         if isinstance(value, float):
#             self._data.append(value)
#         elif isinstance(value, torch.Tensor):
#             self._data[_index]

#         if self._index >= self._data.shape[0]:

#             new_chunk = torch.empty((self._chunk_size, *value.shape), dtype=value.dtype, pin_memory=self._pin)

        



#         self._index += 1




#         torch.cat()



#         if isinstance(value, torch.Tensor):
#             value = value.detach().to("cpu", copy=True)
#         pass

#     def get(self) -> tuple[np.ndarray, np.ndarray | None]:
#         # will also need to prune the last bits


#         if isinstance(self._data[0], torch.Tensor):
#             if any(t.shape != self._data[0].shape for t in self._data):     # if any tensors have unequal shapes, pad them
#                 lengths = np.array([t.shape[0] for t in self._data])
#                 return nn.utils.rnn.pad_sequence(self._data, batch_first=True, padding_value=0).numpy(), lengths
#             else:
#                 return torch.stack(self._data).numpy(), None
#         return np.asarray(self._data, dtype=np.float32), None       
        
        
# torch.cuda.synchronize()



# divmod

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
                value = value.detach().to("cpu", copy=True)
            self.recordings[name].append(value)

    def clear(self) -> None:
        for data in self.recordings.values():
            data.clear()

    def save(self, output_dir: Path) -> None:
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
        super().__init__("loss", "acc", data_name="performance")

    def record(self, *, loss_val: float, acc: float, **_) -> None:
        self._record(loss=loss_val, acc=acc)

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
            metrics[f"weights_forward_{i}"] = w_forward.to(torch.float16)
            if w_recurrent is not None:
                metrics[f"weights_recurrent_{i}"] = w_recurrent.to(torch.float16)
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
        output_dir.mkdir(parents=True, exist_ok=True)
        for r in self.iter_recorders + self.epoch_recorders:
            r.save(output_dir)