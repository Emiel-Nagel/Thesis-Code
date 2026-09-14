import torch
from torch import Tensor
import torch.nn as nn
import snntorch as snn

class RecurrentLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, beta: float) -> None:
        super().__init__()

        self.fc = nn.Linear(n_in, n_out)
        self.lif = snn.Leaky(beta=beta)
        self.rc = nn.Linear(n_out, n_out)

        self.lif_mem = self.lif.init_leaky()
        self.rc_mem: Tensor = None

    def reset_mem(self) -> None:
        self.lif_mem = self.lif.init_leaky()
        self.rc_mem: Tensor = None

    def forward(self, x: Tensor) -> Tensor:
        if self.rc_mem is None:
            self.rc_mem = torch.zeros(x.shape[0], self.rc.out_features, device=x.device)     # init with random recurrent values on first forward pass

        currents = self.fc(x) + self.rc_mem
        spikes, self.lif_mem = self.lif(currents, self.lif_mem)
        self.rc_mem = self.rc(spikes)
        return spikes

class OutputLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, beta: float) -> None:
        super().__init__()

        self.fc = nn.Linear(n_in, n_out)
        self.lif = snn.Leaky(beta=beta)
        self.lif_mem = self.lif.init_leaky()

    def reset_mem(self) -> None:
        self.lif_mem = self.lif.init_leaky()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        currents = self.fc(x)
        spikes, self.lif_mem = self.lif(currents, self.lif_mem)
        return spikes, self.lif_mem

class SRNN(nn.Module):
    def __init__(self, n_in: int, ns_hidden: list[int], n_out: int, beta: float) -> None:
        super().__init__()

        layers = [n_in] + ns_hidden
        self.rec_layers = nn.ModuleList(
            [RecurrentLayer(a, b, beta) for a, b in zip(layers[:-1], layers[1:])]
        )
        self.out_layer = OutputLayer(ns_hidden[-1], n_out, beta)

    def reset_mems(self) -> None:
        for layer in self.rec_layers:
            layer.reset_mem()
        self.out_layer.reset_mem()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.rec_layers:
            x = layer(x)
        return self.out_layer(x)

srnn = SRNN(
    n_in=10,
    ns_hidden=[100, 200],
    n_out=5,
    beta=0.9,
)