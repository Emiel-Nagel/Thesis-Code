import torch
import torch.nn.functional as F

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