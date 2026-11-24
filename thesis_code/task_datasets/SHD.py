import tonic
from tonic import DiskCachedDataset
import torch
import shutil
from pathlib import Path

from .core import DATA_PATH, CACHE_PATH, _build_raw_dataset, _build_dataloader, PreparedLoader

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