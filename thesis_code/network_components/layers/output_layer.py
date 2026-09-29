import torch
import torch.nn as nn
import snntorch as snn

class OutputLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, beta: float) -> None:
        super().__init__()
        self.weights = nn.Linear(n_in, n_out, bias=False)
        self.neurons = snn.Leaky(beta=beta, init_hidden=False, reset_mechanism='none')
        self.mem = self.neurons.init_leaky()

    def reset(self) -> None:
        self.mem = self.neurons.init_leaky()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.weights(x)
        spk, self.mem = self.neurons(x, self.mem)
        return spk, self.mem