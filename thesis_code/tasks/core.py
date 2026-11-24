import torch

from dataclasses import dataclass
from typing import Callable

from ..datasets.core import PreparedLoader

@dataclass
class Task:
    batch_size: int
    n_timepoints: int
    device: torch.device
    trainloader: PreparedLoader
    testloader: PreparedLoader
    input_dim: int
    output_dim: int
    download_dataset: Callable[[], None]
    compute_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    measure_accuracy: Callable[[torch.Tensor, torch.Tensor], float]