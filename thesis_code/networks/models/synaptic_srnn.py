import torch
from torch import Tensor
import torch.nn as nn
import snntorch as snn

class RecurrentLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, alpha: float, beta: float, fully_learnable: bool = True) -> None:
        super().__init__()
        self.weights = nn.Linear(n_in, n_out, bias=False)
        self.neurons = snn.RSynaptic(alpha=alpha, beta=beta, linear_features=n_out, init_hidden=False, learn_alpha=fully_learnable, learn_beta=fully_learnable)
        self.spk, self.syn, self.mem = self.neurons.init_rsynaptic()

    def reset(self) -> None:
        self.spk, self.syn, self.mem = self.neurons.init_rsynaptic()

    def forward(self, x: Tensor) -> Tensor:
        x = self.weights(x)
        self.spk, self.syn, self.mem = self.neurons(x, self.spk, self.syn, self.mem)
        return self.spk

class OutputLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.weights = nn.Linear(n_in, n_out, bias=False)
        self.neurons = snn.Leaky(beta=0.9, init_hidden=False, reset_mechanism='none')
        self.mem = self.neurons.init_leaky()

    def reset(self) -> None:
        self.mem = self.neurons.init_leaky()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.weights(x)
        spk, self.mem = self.neurons(x, self.mem)
        return spk, self.mem

class SynapticSRNN(nn.Module):
    def __init__(self, n_in: int, ns_hidden: list[int], n_out: int, alpha: float, beta: float, fully_learnable: bool = True) -> None:
        super().__init__()
        layers = [n_in] + ns_hidden
        modules = [nn.Flatten()]

        for _n_in, _n_out in zip(layers[:-1], layers[1:]):
            modules.append(RecurrentLayer(_n_in, _n_out, alpha, beta, fully_learnable))

        modules.append(OutputLayer(ns_hidden[-1], n_out))
        self.net = nn.Sequential(*modules)

    def reset(self) -> None:
        for module in self.net:
            if isinstance(module, (RecurrentLayer, OutputLayer)):
                module.reset()

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        spk_rec, mem_rec = [], []
        self.reset()

        for step in range(data.size(0)):
            spk_out, mem_out = self.net(data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)