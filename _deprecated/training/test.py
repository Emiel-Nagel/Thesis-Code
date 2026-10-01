import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .functions import measure_accuracy

def test_net(net: nn.Module, testloader: DataLoader, max_iters: int = None) -> float:
    with torch.no_grad():
        acc = 0
        net.eval()

        for i, (data, targets) in enumerate(testloader):           
            _, mem_rec, _ = net(data)
            acc += measure_accuracy(mem_rec, targets)
            if max_iters is not None and i == max_iters:
                break

    return acc / len(testloader)