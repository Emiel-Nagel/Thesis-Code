import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import shutil

def get_SHD_dataloader(batch_size: int, time_window: float = 10000, train: bool = True, shuffle: bool = True, re_download: bool = False) -> tuple[DataLoader, int, int]:
    train_extension = "train" if train else "test"
    if re_download:
        shutil.rmtree(f'./datasets/cache/SHD/{train_extension}', ignore_errors=True)
        shutil.rmtree(f'./datasets/data/SHD', ignore_errors=True)

    sensor_size=tonic.datasets.SHD.sensor_size
    frame_transform = transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=time_window,
        start_time=0,
    )
    dataset = tonic.datasets.SHD(
        save_to='./datasets/data', 
        train=train,
        transform=frame_transform,
    )
    cached_dataset = DiskCachedDataset(
        dataset,
        cache_path=f'./datasets/cache/SHD/{train_extension}'
    )
    dataloader = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=shuffle,
    )
    return dataloader, sensor_size[0], dataset.classes