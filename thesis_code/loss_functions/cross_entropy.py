import torch
import torch.nn.functional as F

def measure_accuracy(mem_outs: torch.Tensor, targets: torch.Tensor) -> float:
    probs = F.softmax(mem_outs, dim=-1)
    idx = probs.mean(dim=0).argmax(-1)
    return (idx == targets).float().mean().item()   # counts the number of output indices that match with the targets

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

def measure_accuracy_multi_target(mem_outs: torch.Tensor, multi_targets: torch.Tensor) -> float:
    # multi_targets is of shape (N, C) where N is the batch dimension and C the number of classes
    # mem_outs has shape (T, N, C) where T is the time dimension
    # It has zeros everywhere, except ones for the correct labels
    n_classes = multi_targets.count_nonzero(dim=-1)         # shape = (N)
    mean_probs = F.softmax(mem_outs, dim=-1).mean(dim=0)    # shape = (N, C)

    mem_outs_ranked = mean_probs.argsort(dim=-1, descending=True).argsort(dim=-1)
    outs_predicted = mem_outs_ranked < n_classes[:, None]       # select n highest predictions

    hits = (outs_predicted & multi_targets.bool()).sum(dim=-1)  # equivalent of (idx == targets)
    return (hits / n_classes).mean().item()

def compute_loss_multi_target(mem_outs: torch.Tensor, multi_targets: torch.Tensor) -> torch.Tensor:
    # multi_targets is of shape (N, C) where N is the batch dimension and C the number of classes
    # It has zeros everywhere, except ones for the correct labels
    targets_dist = multi_targets / multi_targets.sum(-1, keepdim=True)  # normalize over the number of correct classes to match the magnitude with def compute_loss
    probs = F.softmax(mem_outs, dim=-1)
    log_probs = torch.log(probs.mean(dim=0) + 1e-8) # to avoid log(0)
    return -(log_probs * targets_dist).sum(-1).mean()

if __name__ == "__main__":
    T, N, C = 5, 8, 10
    mem_outs = torch.randn(T, N, C)
    targets = torch.randint(0, C, (N,))
    multi = F.one_hot(targets, C)

    assert torch.isclose(torch.tensor(measure_accuracy(mem_outs, targets)),
                        torch.tensor(measure_accuracy_multi_target(mem_outs, multi)))
    assert torch.isclose(compute_loss(mem_outs, targets),
                        compute_loss_multi_target(mem_outs, multi))
    print("Loss functions work")