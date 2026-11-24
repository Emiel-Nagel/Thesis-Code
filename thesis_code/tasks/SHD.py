import torch
from functools import partial

from ..loss_functions import CrossEntropy
from ..task_datasets import SHD, SHD_Summed

from .core import Task

class ClassifySHD(Task):
    def __init__(self, batch_size: int, n_timepoints: int, device: torch.device) -> None:
        trainloader, testloader, input_dim, classes = SHD.get_SHD_dataset(
            batch_size=batch_size, device=device, n_timepoints=n_timepoints,
        )
        super().__init__(
            batch_size=batch_size,
            n_timepoints=n_timepoints,
            device=device,
            trainloader=trainloader,
            testloader=testloader,
            input_dim=input_dim,
            output_dim=len(classes),
            download_dataset=partial(SHD.download_SHD_dataset, n_timepoints, re_download=False),
            compute_loss=CrossEntropy.compute_loss,
            measure_accuracy=CrossEntropy.measure_accuracy,
        )

class ClassifySummedSHD(Task):
    def __init__(self, batch_size: int, n_timepoints: int, device: torch.device) -> None:
        trainloader, testloader, input_dim, classes = SHD_Summed.get_SHD_Summed_dataset(
            batch_size=batch_size, device=device, n_timepoints=n_timepoints,
        )
        super().__init__(
            batch_size=batch_size,
            n_timepoints=n_timepoints,
            device=device,
            trainloader=trainloader,
            testloader=testloader,
            input_dim=input_dim,
            output_dim=len(classes),
            download_dataset=partial(SHD_Summed.download_SHD_dataset, n_timepoints, re_download=False),
            compute_loss=CrossEntropy.compute_loss,
            measure_accuracy=CrossEntropy.measure_accuracy,
        )