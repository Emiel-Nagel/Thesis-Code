import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import shutil
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_PATH = DATASETS_DIR / "data"
CACHE_PATH = DATASETS_DIR / "cache"

class PreparedLoader:
    """Wraps a DataLoader and applies the standard device/dtype/shape ops to each batch."""

    def __init__(self, loader: DataLoader, device: torch.device, batch_size: int) -> None:
        self.device = device
        self.data, self.targets = self._load_data(loader)
        self.batch_size = batch_size

    def _load_data(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads all data into one big tensor into memory.
        """
        data_list, targets_list = [], []
        for data, targets in loader:
            data_list.append(data.squeeze(2).clamp_(max=255).to(torch.uint8))
            targets_list.append(targets)
        return torch.cat(data_list).to(self.device), torch.cat(targets_list).long().to(self.device)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size

    def __iter__(self):
        perm = torch.randperm(len(self.data), device=self.device)
        for i in range(0, len(perm) - self.batch_size + 1, self.batch_size):  # drops last incomplete batch
            sample_indices = perm[i : i + self.batch_size]
            yield self.data[sample_indices].float().transpose(0, 1), self.targets[sample_indices]

def _build_dataloader(dataset: Dataset, cache_path: Path) -> DataLoader:
    cached_dataset = DiskCachedDataset(
        dataset,
        cache_path=str(cache_path)
    )
    return DataLoader(
        cached_dataset,
        batch_size=256,
        collate_fn=tonic.collation.PadTensors(batch_first=True),
        shuffle=False,
        drop_last=False,
    )

def _build_raw_dataset(train: bool, n_timepoints: int) -> tonic.datasets.SHD:
    """
    What ToFrame does

    With n_time_bins=n_timepoints, it:

    Splits the time axis of the sample into n_timepoints equal-width bins. start_time=0 makes the first bin start at t = 0. The last bin ends at the sample's final event time.
    Counts the spikes of each neuron in each bin.
    Returns an array of shape (n_timepoints, 1, 700), which is (T, polarity, neurons). Each entry is the number of spikes that neuron fired in that time bin.

    For example, with n_timepoints=100 and a ~1 s recording, each bin is about 10 ms wide. A neuron that fires 3 times in that window gets a value of 3 in that bin.
    """
    return tonic.datasets.SHD(
        save_to=str(DATA_PATH),
        train=train,
        transform=transforms.ToFrame(
            sensor_size=tonic.datasets.SHD.sensor_size,
            n_time_bins=n_timepoints,
            start_time=0,
        ),
    )

def _SHD_cache_path(n_timepoints: int, train: bool) -> Path:
    return CACHE_PATH / "SHD" / f"tw{n_timepoints}" / ("train" if train else "test")

def download_SHD_dataset(n_timepoints: int = 200, re_download: bool = False):
    if re_download:
        shutil.rmtree(DATA_PATH / "SHD", ignore_errors=True)
        shutil.rmtree(CACHE_PATH / "SHD", ignore_errors=True)

    for train in (True, False):
        dataset = _build_raw_dataset(train, n_timepoints)
        cached = DiskCachedDataset(dataset, cache_path=str(_SHD_cache_path(n_timepoints, train)))
        for i in range(len(cached)):    # forces every sample to be computed and written
            cached[i]

def get_SHD_dataset(batch_size: int, device: torch.device, n_timepoints: int = 200) -> tuple[PreparedLoader, PreparedLoader, int, list]:
    sensor_size = tonic.datasets.SHD.sensor_size
    train_dataset = _build_raw_dataset(train=True, n_timepoints=n_timepoints)
    trainloader = _build_dataloader(train_dataset, _SHD_cache_path(n_timepoints, True))

    test_dataset = _build_raw_dataset(train=False, n_timepoints=n_timepoints)
    testloader = _build_dataloader(test_dataset, _SHD_cache_path(n_timepoints, False))

    return (
        PreparedLoader(trainloader, device, batch_size),
        PreparedLoader(testloader, device, batch_size),
        sensor_size[0],
        train_dataset.classes,
    )

@dataclass
class DatasetConfig:
    re_download: bool
    n_timepoints: int