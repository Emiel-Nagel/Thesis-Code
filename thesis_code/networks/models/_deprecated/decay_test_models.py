import torch
from torch import Tensor
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

def sample_heterogeneous_decays(n_neurons: int, device) -> Tensor:
    upper_bound = 0.96
    lower_bound = 0.69
    decay_range = upper_bound - lower_bound
    return torch.rand(n_neurons, dtype=torch.float, device=device) * decay_range + lower_bound

class SimpleSRNNHeterogeneousDecay(nn.Module):
    def __init__(self, n_in: int, ns_hidden: list[int], n_out: int, beta: float) -> None:
        super().__init__()
        layers = [n_in] + ns_hidden
        modules = [nn.Flatten()]

        for _n_in, _n_out in zip(layers[:-1], layers[1:]):
            modules.append(nn.Linear(_n_in, _n_out, bias=False))
            beta_layer = sample_heterogeneous_decays(_n_out)
            modules.append(snn.RLeaky(beta=beta_layer, linear_features=_n_out, init_hidden=True, learn_beta=False))

        modules.append(nn.Linear(ns_hidden[-1], n_out, bias=False))
        modules.append(snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism='none'))
        self.net = nn.Sequential(*modules)

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        spk_rec, mem_rec = [], []
        utils.reset(self.net)

        for step in range(data.size(0)):
            spk_out, mem_out = self.net(data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

class SimpleSRNNAdaptiveDecay(nn.Module):
    def __init__(self, n_in: int, ns_hidden: list[int], n_out: int, beta: float) -> None:
        super().__init__()
        layers = [n_in] + ns_hidden
        modules = [nn.Flatten()]

        for _n_in, _n_out in zip(layers[:-1], layers[1:]):
            modules.append(nn.Linear(_n_in, _n_out, bias=False))
            modules.append(snn.RLeaky(beta=beta, linear_features=_n_out, init_hidden=True, learn_beta=True))

        modules.append(nn.Linear(ns_hidden[-1], n_out, bias=False))
        modules.append(snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism='none'))
        self.net = nn.Sequential(*modules)

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        spk_rec, mem_rec = [], []
        utils.reset(self.net)

        for step in range(data.size(0)):
            spk_out, mem_out = self.net(data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

class SimpleSRNNHeterogeneousAdaptiveDecay(nn.Module):
    def __init__(self, n_in: int, ns_hidden: list[int], n_out: int, beta: float) -> None:
        super().__init__()
        layers = [n_in] + ns_hidden
        modules = [nn.Flatten()]

        for _n_in, _n_out in zip(layers[:-1], layers[1:]):
            modules.append(nn.Linear(_n_in, _n_out, bias=False))
            beta_layer = sample_heterogeneous_decays(_n_out)
            modules.append(snn.RLeaky(beta=beta_layer, linear_features=_n_out, init_hidden=True, learn_beta=True))

        modules.append(nn.Linear(ns_hidden[-1], n_out, bias=False))
        modules.append(snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism='none'))
        self.net = nn.Sequential(*modules)

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        spk_rec, mem_rec = [], []
        utils.reset(self.net)

        for step in range(data.size(0)):
            spk_out, mem_out = self.net(data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)