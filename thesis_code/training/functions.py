import torch
import torch.nn.functional as F
import numpy as np

def measure_accuracy(mem_outs: torch.Tensor, targets: torch.Tensor):
    _, idx = mem_outs.sum(dim=0).max(1)
    return np.mean((targets == idx).detach().cpu().numpy())

def compute_loss(mem_outs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(mem_outs, dim=-1)
    log_probs = torch.log(probs.mean(dim=0) + 1e-8) # to avoid log(0)
    return F.nll_loss(log_probs, targets)

def get_compute_loss_reg_fn(lam_lower: float, v_lower: float, lam_uppers: list[float], v_uppers: list[float], L: int = 2) -> callable:
    """
    Based on:
    Friedemann Zenke, Tim P. Vogels; The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks. Neural Comput 2021; 33 (4): 899-925. doi: https://doi.org/10.1162/neco_a_01367
    """
    assert len(lam_uppers) == len(v_uppers), \
        f"Mismatching parameter counts: lam_uppers={len(lam_uppers)}, v_uppers={len(v_uppers)}"
    
    def compute_loss_reg(hidden_spks: list[torch.Tensor]) -> torch.Tensor:
        loss_regs = []
        for spks_h, lam_upper, v_upper in zip(hidden_spks, lam_uppers, v_uppers):
            # spks_h has shape (n_steps, batch_size, n_neurons), for example: (116, 128, 256)
            spk_count = spks_h.sum(dim=0)  # has shape (batch_size, n_neurons)
            g_lower = lam_lower * torch.relu(v_lower - spk_count).square().mean(dim=-1)
            g_upper = lam_upper * torch.relu(spk_count.mean(dim=-1) - v_upper).pow(L)
            loss_regs.append((g_lower + g_upper).mean(dim=0))      # output shape should be scalar ()
        return torch.stack(loss_regs).sum()
    return compute_loss_reg