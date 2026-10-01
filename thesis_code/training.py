import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from tqdm.auto import tqdm
import psutil, os

from .srnn import SRNN
from .recording import SpikeRecorder, PerformanceRecorder, GradientRecorder

def measure_accuracy(mem_outs: torch.Tensor, targets: torch.Tensor) -> float:
    probs = F.softmax(mem_outs, dim=-1)
    idx = probs.mean(dim=0).argmax(-1)
    return (idx == targets).float().mean().item()

def compute_loss(mem_outs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(mem_outs, dim=-1)
    log_probs = torch.log(probs.mean(dim=0) + 1e-8) # to avoid log(0)
    return F.nll_loss(log_probs, targets)

class Regularizer(nn.Module):
    def __init__(self, lam_lower: float, v_lower: float, lam_uppers: list[float], v_uppers: list[float], L: int = 2) -> None:
        """
        Based on:
        Friedemann Zenke, Tim P. Vogels; The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks. Neural Comput 2021; 33 (4): 899-925. doi: https://doi.org/10.1162/neco_a_01367
        """
        super().__init__()
        assert len(lam_uppers) == len(v_uppers), \
            f"Mismatching parameter counts: lam_uppers={len(lam_uppers)}, v_uppers={len(v_uppers)}"

        self.lam_lower = lam_lower
        self.v_lower = v_lower
        self.lam_uppers = lam_uppers
        self.v_uppers = v_uppers
        self.L = L

    def forward(self, hidden_spks: list[torch.Tensor]) -> torch.Tensor:
        loss_regs = []
        for spks_h, lam_upper, v_upper in zip(hidden_spks, self.lam_uppers, self.v_uppers):
            # spks_h has shape (n_steps, batch_size, n_neurons), for example: (116, 128, 256)
            spk_count = spks_h.sum(dim=0)  # has shape (batch_size, n_neurons)
            g_lower = self.lam_lower * torch.relu(self.v_lower - spk_count).square().mean(dim=-1)
            g_upper = lam_upper * torch.relu(spk_count.mean(dim=-1) - v_upper).pow(self.L)
            loss_regs.append((g_lower + g_upper).mean(dim=0))      # output shape should be scalar ()
        return torch.stack(loss_regs).sum()

def test_net(net: SRNN, testloader: DataLoader, max_iters: int = None) -> float:
    with torch.no_grad():
        acc = 0
        net.eval()

        for i, (data, targets) in enumerate(testloader):           
            _, mem_rec, _ = net(data)
            acc += measure_accuracy(mem_rec, targets)
            if max_iters is not None and i == max_iters:
                break

    return acc / len(testloader)

def train_net(net: SRNN, trainloader: DataLoader, lr: float = 1e-3, n_epochs: int = 1, max_iters: int = None,
        regularizer: Regularizer = None, grad_rec: GradientRecorder = None, spk_rec: SpikeRecorder = None,
    ) -> tuple[nn.Module, PerformanceRecorder, GradientRecorder | None, SpikeRecorder | None]:

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    process = psutil.Process(os.getpid())
    perf_rec = PerformanceRecorder()

    net.train()

    for epoch in range(n_epochs):
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}")
        losses, accs = [], []
        for i, (data, targets) in progress_bar:

            spk_outs, mem_outs, hidden_spks = net(data)
            loss_val = compute_loss(mem_outs, targets)
            
            if regularizer is not None:
                loss_val = loss_val +  regularizer(hidden_spks)

            if spk_rec is not None:
                spk_rec.record(data, hidden_spks, spk_outs, targets)

            optimizer.zero_grad()
            loss_val.backward()

            if grad_rec is not None:
                grad_rec.record(net)

            optimizer.step()

            losses.append(loss_val.item())
            acc = measure_accuracy(mem_outs.detach(), targets)
            accs.append(acc)

            progress_bar.set_postfix(
                loss=f"{loss_val.item():.2f}",
                acc=f"{acc * 100:.2f}%",
                time_elapsed=f"{progress_bar.format_dict['elapsed']:.1f}s",
                used_memory=f"{process.memory_info().rss / 1e9:.2f} GB",
            )

            if max_iters is not None and i == max_iters:
                break

        perf_rec.record(float(np.mean(losses)), float(np.mean(accs)))
    return net, perf_rec, grad_rec, spk_rec