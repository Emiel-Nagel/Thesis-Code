import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import mlflow
import numpy as np
from tqdm.std import EMA
from datetime import datetime
from zoneinfo import ZoneInfo
import time
from dataclasses import dataclass, field

from .srnn import SRNN
from .recording import Recorder
from .datasets import DatasetConfig, PreparedLoader

def get_timestamp() -> int:
    return int(datetime.now(tz=ZoneInfo("Europe/Amsterdam")).timestamp() * 1000)

class RateTracker:
    def __init__(self, smoothing=0.3):
        self.ema_dn = EMA(smoothing)
        self.ema_dt = EMA(smoothing)
        self.last_n = 0
        self.last_t = time.perf_counter()

    def update(self, n):
        now = time.perf_counter()
        self.ema_dn(n - self.last_n)
        self.ema_dt(now - self.last_t)
        self.last_n, self.last_t = n, now
        return self.ema_dn() / self.ema_dt()

def measure_accuracy(mem_outs: torch.Tensor, targets: torch.Tensor) -> float:
    probs = F.softmax(mem_outs, dim=-1)
    idx = probs.mean(dim=0).argmax(-1)
    return (idx == targets).float().mean().item()

def compute_loss(mem_outs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Should converge to cross-entropy loss
    It does, cross-entropy loss is basically nll losss if you first apply softmax and log around the outputs.
    See:
    https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html
    and
    https://docs.pytorch.org/docs/2.14/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
    """
    probs = F.softmax(mem_outs, dim=-1)
    log_probs = torch.log(probs.mean(dim=0) + 1e-8) # to avoid log(0)
    return F.nll_loss(log_probs, targets)

def get_gradients(net: SRNN) -> tuple[float, float]:
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    if grads:
        n = sum(g.numel() for g in grads)
        max_grad = torch.stack(torch._foreach_norm(grads, float("inf"))).max()
        avg_grad = torch.stack(torch._foreach_norm(grads, 1)).sum() / n
    else:
        max_grad = avg_grad = torch.zeros(())
    return max_grad.item(), avg_grad.item()

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

def test_net(net: SRNN, testloader: DataLoader, max_iters: int = None) -> tuple[float, float]:
    with torch.no_grad():
        acc = 0
        loss = 0
        net.eval()

        for i, (data, targets) in enumerate(testloader):
            net.reset()
            _, mem_rec, _, _ = net(data)
            acc += measure_accuracy(mem_rec, targets)
            loss += compute_loss(mem_rec, targets)
            if max_iters is not None and i == max_iters:
                break

    return acc / len(testloader), loss / len(testloader)

def train_net(net: SRNN, seed_id: str, device: torch.device, trainloader: PreparedLoader, testloader: PreparedLoader, lr: float = 1e-3, n_epochs: int = 1, max_iters: int = None, 
        separate_timesteps: bool = False, regularizer: Regularizer = None, recorder: Recorder = None,
    ) -> tuple[SRNN, torch.optim.Optimizer, Recorder]:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(n_epochs):
        mlflow.set_tag("progress", f"Epoch {epoch + 1}/{n_epochs}")

        net.train()

        losses, accs = [], []
        tracker = RateTracker()
        for iteration, (data, targets) in enumerate(trainloader):
            mlflow.log_metric("it/s", tracker.update(iteration), step=iteration, run_id=seed_id)

            net.reset()
            forward_fn = net.forward_sequence if not separate_timesteps else net
            spk_outs, mem_outs, hidden_spks, hidden_mems = forward_fn(data)

            loss_val = compute_loss(mem_outs, targets)
            
            if regularizer is not None:
                loss_val = loss_val +  regularizer(hidden_spks)

            optimizer.zero_grad()
            loss_val.backward()

            if recorder is not None:
                recorder.record_iteration(net=net)

            optimizer.step()
            net.apply_daleslaw()
            # net.constrain_betas()

            losses.append(loss_val.item())
            acc = measure_accuracy(mem_outs.detach(), targets)
            accs.append(acc)

        acc_test, loss_test = test_net(net, testloader, max_iters)
        max_grad, avg_grad = get_gradients(net)
        mlflow.log_metrics(
            {"train_loss": loss_val.item(), "train_accuracy": acc, "test_loss": loss_test, "test_accuracy": acc_test, 
            "cuda_memory_peak GiB": (torch.cuda.max_memory_allocated(device) / 2**30),
            "max_grad": max_grad, "avg_grad": avg_grad},
            step=epoch, synchronous=False, timestamp=get_timestamp(), run_id=seed_id
        )

        if recorder is not None:
            recorder.record_epoch(
                loss_train=float(np.mean(losses)),
                acc_train=float(np.mean(accs)),
                loss_test=loss_test,
                acc_test=acc_test,
                weights=net.get_weights(),
                spk_ins=data,
                hidden_spks=hidden_spks,
                mem_outs=mem_outs,
                targets=targets,
            )

    return net, optimizer, recorder

@dataclass
class RegularizationConfig:
    apply: bool
    lam_lower: float
    v_lower: float
    lam_uppers: list[float]
    v_uppers: list[float]
    L: int

@dataclass
class TrainingConfig:
    dt: float
    batch_size: int
    max_iters: int
    n_epochs: int
    lr: float
    separate_timesteps: bool
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)