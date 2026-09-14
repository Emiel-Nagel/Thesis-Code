import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def get_SHD_dataloader(batch_size: int, time_window: float = 10000, train: bool = True, shuffle: bool = True, re_download: bool = False) -> tuple[DataLoader, int, int]:
    train_extension = "train" if train else "test"
    cache_path = DATASETS_DIR / "cache" / "SHD" / train_extension
    data_path = DATASETS_DIR / "data"

    if re_download:
        shutil.rmtree(cache_path, ignore_errors=True)
        shutil.rmtree(DATASETS_DIR / "data" / "SHD", ignore_errors=True)

    sensor_size = tonic.datasets.SHD.sensor_size
    frame_transform=transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=time_window,
        start_time=0,
    )
    dataset = tonic.datasets.SHD(
        save_to=str(data_path), 
        train=train,
        transform=frame_transform,
    )
    cached_dataset = DiskCachedDataset(
        dataset,
        cache_path=str(cache_path)
    )
    dataloader = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=shuffle,
    )
    return dataloader, sensor_size[0], dataset.classes