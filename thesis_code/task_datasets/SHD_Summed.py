import tonic
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .core import _build_raw_dataset, _build_dataloader, PreparedLoader
from .SHD import _SHD_cache_path, download_SHD_dataset

download_SHD_Summed_dataset = download_SHD_dataset

class SHDSummedLoader(PreparedLoader):
    def __init__(self, loader: DataLoader, device: torch.device, batch_size: int, n_classes: int, seed: int = 0, resample: bool = True) -> None:
        super().__init__(loader, device, batch_size)
        self.n_classes = n_classes
        self.seed = seed
        self.epoch = 0
        self.resample = resample

    def _make_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        N = len(self.data)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + self.epoch)
        i = torch.randperm(N, device=self.device, generator=generator)
        j = torch.randperm(N, device=self.device, generator=generator)
        bad = self.targets[i] == self.targets[j]
        while bad.any():
            j[bad] = torch.randint(N, (int(bad.sum()),), device=self.device, generator=generator)
            bad = self.targets[i] == self.targets[j]
        return i, j

    def __iter__(self):
        i, j = self._make_pairs()
        for index in range(0, len(i) - self.batch_size + 1, self.batch_size):
            s = slice(index, index + self.batch_size)
            data = self.data[i[s]].float() + self.data[j[s]].float()
            targets = F.one_hot(self.targets[i[s]], self.n_classes) | F.one_hot(self.targets[j[s]], self.n_classes)     # construct multi_targets of shape (N, C), it has zeros everywhere, except ones for the correct labels
            yield data.transpose(0, 1), targets.float()

        if self.resample:
            self.epoch += 1
 
def get_SHD_Summed_dataset(batch_size: int, device: torch.device, n_timepoints: int = 200) -> tuple[PreparedLoader, PreparedLoader, int, list]:
    sensor_size = tonic.datasets.SHD.sensor_size
    train_dataset = _build_raw_dataset(train=True, n_timepoints=n_timepoints)
    trainloader = _build_dataloader(train_dataset, _SHD_cache_path(n_timepoints, True))

    test_dataset = _build_raw_dataset(train=False, n_timepoints=n_timepoints)
    testloader = _build_dataloader(test_dataset, _SHD_cache_path(n_timepoints, False))

    return (
        SHDSummedLoader(trainloader, device, batch_size, n_classes=len(train_dataset.classes), resample=True),
        SHDSummedLoader(testloader, device, batch_size, n_classes=len(train_dataset.classes), resample=False),
        sensor_size[0],
        train_dataset.classes,
    )