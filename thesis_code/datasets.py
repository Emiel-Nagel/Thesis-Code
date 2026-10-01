import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def _build_dataloader(dataset: Dataset, cache_path: Path, batch_size: int, train: bool) -> DataLoader:
    train_extension = "train" if train else "test"

    cached_dataset = DiskCachedDataset(
        dataset,
        cache_path=str(cache_path / train_extension)
    )
    return DataLoader(
        cached_dataset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=train,
        drop_last=train,
    )

class PreparedLoader:
    """Wraps a DataLoader and applies the standard device/dtype/shape ops to each batch."""

    def __init__(self, loader: DataLoader, device: torch.device | str) -> None:
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for data, targets in self.loader:
            data = data.to(self.device, non_blocking=True).squeeze()
            targets = targets.to(self.device, non_blocking=True).long()
            yield data, targets

    def __getattr__(self, name: str):
        return getattr(self.loader, name)

def get_SHD_dataset(batch_size: int, device: torch.device, time_window: float = 1000, re_download: bool = False) -> tuple[PreparedLoader, PreparedLoader, int, list]:
    data_path = DATASETS_DIR / "data"
    cache_path = DATASETS_DIR / "cache" / "SHD" / f"tw{int(time_window)}"
    
    if re_download:
        shutil.rmtree(data_path / "SHD", ignore_errors=True)
        shutil.rmtree(DATASETS_DIR / "cache" / "SHD", ignore_errors=True)

    sensor_size = tonic.datasets.SHD.sensor_size
    frame_transform=transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=time_window,
        start_time=0,
    )

    train_dataset = tonic.datasets.SHD(
        save_to=str(data_path), 
        train=True,
        transform=frame_transform,
    )
    trainloader = _build_dataloader(train_dataset, cache_path, batch_size, train=True)

    test_dataset = tonic.datasets.SHD(
        save_to=str(data_path), 
        train=False,
        transform=frame_transform,
    )
    testloader = _build_dataloader(test_dataset, cache_path, batch_size, train=False)

    return (
        PreparedLoader(trainloader, device),
        PreparedLoader(testloader, device),
        sensor_size[0],
        train_dataset.classes,
    )
