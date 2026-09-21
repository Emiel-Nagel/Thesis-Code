import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from snntorch import utils
from snntorch import functional as SF
from snntorch.functional import LossFunctions

from tqdm.auto import tqdm

def train(net: nn.Module, device: torch.device, trainloader: DataLoader, optimizer: Optimizer, loss_fn: LossFunctions, loss_is_membrane: bool = False, n_epochs: int = 1, max_iters: int = None) -> tuple[nn.Module, list, list]:
    loss_hist, acc_hist = [], []
    net.train()

    for epoch in range(n_epochs):
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}")
        for i, (data, targets) in progress_bar:
            data, targets = data.to(device), targets.to(device)
            data = data.squeeze()
            
            spk_rec, mem_rec = net(data)
            loss_val = loss_fn(mem_rec, targets) if loss_is_membrane else loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())

            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)

            progress_bar.set_postfix(loss=f"{loss_val.item():.2f}", acc=f"{acc * 100:.2f}%")

            if i % 25 == 0:
                tqdm.write(f"Epoch {epoch}, Iteration {i} — Loss: {loss_val.item():.2f}, Acc: {acc*100:.2f}%")

            if max_iters is not None and i == max_iters:
                break

    return net, loss_hist, acc_hist

def test(net: nn.Module, device: torch.device, testloader: DataLoader, max_iters: int = None) -> float:
    with torch.no_grad():
        total, acc = 0, 0
        net.eval()

        for i, (data, targets) in enumerate(testloader):
            data, targets = data.to(device), targets.to(device)
            utils.reset(net)
            spk_rec, _ = net(data)
            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)
            if max_iters is not None and i == max_iters:
                break

    return acc / total