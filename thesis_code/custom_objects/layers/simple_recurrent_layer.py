import torch
import torch.nn as nn
import snntorch as snn

class SimpleRecurrentLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, beta: float, fully_learnable: bool = True) -> None:
        super().__init__()
        self.weights = nn.Linear(n_in, n_out, bias=False)
        self.neurons = snn.RLeaky(beta=beta, linear_features=n_out, init_hidden=False, learn_beta=fully_learnable)
        self.spk, self.mem = self.neurons.init_rleaky()

    def reset(self) -> None:
        self.spk, self.mem = self.neurons.init_rleaky()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weights(x)
        self.spk, self.mem = self.neurons(x, self.spk, self.mem)
        return self.spk