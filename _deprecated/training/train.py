import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Callable

from .functions import measure_accuracy

# TODO: finish the gradient recording bit

def record_grads(net: nn.Module) -> tuple[list, list]:
    grads = {name: p.grad.detach().clone()
        for name, p in net.named_parameters() if p.grad is not None}
    if len(grads) > 0:
        flat_grads = torch.cat([t.flatten() for t in grads.values()])
        return flat_grads.max(dim=0).values.cpu().numpy(), flat_grads.mean(dim=0).cpu().numpy()
    return 0, 0

def train_net(net: nn.Module, trainloader: DataLoader, loss_fn: Callable, lr: float = 1e-3, n_epochs: int = 1, max_iters: int = None) -> tuple[nn.Module, list, list]:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-6, betas=(0.9, 0.99))

    loss_hist, acc_hist, spike_hist = [], [], []
    net.train()
    max_grad_rec = []
    avg_grad_rec = []

    for epoch in range(n_epochs):
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}")
        for i, (data, targets) in progress_bar:
            _, mem_outs, hidden_spks = net(data)
            loss_val = loss_fn(mem_outs, targets, hidden_spks)

            optimizer.zero_grad()
            loss_val.backward()

            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1e6)

            max_grad, avg_grad = record_grads(net)
            max_grad_rec.append(max_grad)
            avg_grad_rec.append(avg_grad)

            optimizer.step()

            loss_hist.append(loss_val.item())

            acc = measure_accuracy(mem_outs, targets)
            acc_hist.append(acc)

            progress_bar.set_postfix(loss=f"{loss_val.item():.2f}", acc=f"{acc * 100:.2f}%")

            if i % 25 == 0:
                elapsed = progress_bar.format_dict["elapsed"]
                tqdm.write(
                    f"Epoch {epoch}, Iteration {i} — Loss: {loss_val.item():.2f}, "
                    f"Acc: {acc*100:.2f}%, Time elapsed: {elapsed:.1f}s"
                )

            if max_iters is not None and i == max_iters:
                break

    return net, loss_hist, acc_hist, spike_hist, max_grad_rec, avg_grad_rec