import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_PATH = DATASETS_DIR / "data"
CACHE_PATH = DATASETS_DIR / "cache"

class PreparedLoader:
    """Wraps a DataLoader and applies the standard device/dtype/shape ops to each batch."""

    def __init__(self, loader: DataLoader, device: torch.device | str) -> None:
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for data, targets in self.loader:
            data = data.to(self.device, non_blocking=True).squeeze(2)
            targets = targets.to(self.device, non_blocking=True).long()
            yield data, targets

    def __getattr__(self, name: str):
        if name == "loader":
            raise AttributeError(name)
        return getattr(self.loader, name)

def _build_dataloader(dataset: Dataset, cache_path: Path, batch_size: int, train: bool, pin_memory: bool) -> DataLoader:
    cached_dataset = DiskCachedDataset(
        dataset,
        cache_path=str(cache_path)
    )
    return DataLoader(
        cached_dataset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=train,
        pin_memory=pin_memory,
        drop_last=train,
    )

def _build_raw_dataset(train: bool, time_window: float) -> tonic.datasets.SHD:
    return tonic.datasets.SHD(
        save_to=str(DATA_PATH),
        train=train,
        transform=transforms.ToFrame(
            sensor_size=tonic.datasets.SHD.sensor_size,
            time_window=time_window,
            start_time=0,
        ),
    )

def _SHD_cache_path(time_window: float, train: bool) -> Path:
    return CACHE_PATH / "SHD" / f"tw{round(time_window)}" / ("train" if train else "test")

def download_SHD_dataset(time_window: float = 1000, re_download: bool = False):
    if re_download:
        shutil.rmtree(DATA_PATH / "SHD", ignore_errors=True)
        shutil.rmtree(CACHE_PATH / "SHD", ignore_errors=True)

    for train in (True, False):
        dataset = _build_raw_dataset(train, time_window)
        cached = DiskCachedDataset(dataset, cache_path=str(_SHD_cache_path(time_window, train)))
        for i in range(len(cached)):    # forces every sample to be computed and written
            cached[i]

def get_SHD_dataset(batch_size: int, device: torch.device, time_window: float = 1000) -> tuple[PreparedLoader, PreparedLoader, int, list]:
    sensor_size = tonic.datasets.SHD.sensor_size
    pin_memory = torch.device(device).type == "cuda"
    train_dataset = _build_raw_dataset(train=True, time_window=time_window)
    trainloader = _build_dataloader(train_dataset, _SHD_cache_path(time_window, True), batch_size, train=True, pin_memory=pin_memory)

    test_dataset = _build_raw_dataset(train=False, time_window=time_window)
    testloader = _build_dataloader(test_dataset, _SHD_cache_path(time_window, False), batch_size, train=False, pin_memory=pin_memory)

    return (
        PreparedLoader(trainloader, device),
        PreparedLoader(testloader, device),
        sensor_size[0],
        train_dataset.classes,
    )
