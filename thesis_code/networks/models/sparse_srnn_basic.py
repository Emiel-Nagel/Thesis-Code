import torch
from torch import Tensor
import torch.nn as nn
import snntorch as snn
from snntorch import utils

from ...custom_objects.neurons import SparseRLeaky
from ...connectivity import sample_ee_pv_som_neurons, build_rec_matrices

class BasicSparseSRNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, rec_layers: list[SparseRLeaky], out_layer: snn.Leaky) -> None:
        super().__init__()
        ns_hidden = [l.n_neurons for l in rec_layers]              # assuming all rec_matrices are square
        layers = [n_in] + ns_hidden
        modules = [nn.Flatten()]

        for i, (_n_in, _n_out) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Linear(_n_in, _n_out))
            modules.append(rec_layers[i])

        modules.append(nn.Linear(ns_hidden[-1], n_out))
        modules.append(out_layer)
        self.net = nn.Sequential(*modules)

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        spk_rec, mem_rec = [], []
        utils.reset(self.net)

        for step in range(data.size(0)):
            spk_out, mem_out = self.net(data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)


def build_basic_pv_som_srnn(n_in: int, ns_hidden: list[int], n_out: int, beta: float, percent_pv: float, percent_som: float) -> BasicSparseSRNN:
    neurons = sample_ee_pv_som_neurons(ns_hidden, percent_pv, percent_som)
    # prob_neuron = lambda neurons: neurons.
    rec_matrices = build_rec_matrices(neurons)
    return BasicSparseSRNN(
        n_in=n_in,
        n_out=n_out,
        rec_layers=[SparseRLeaky(beta=beta, init_hidden=True, rec_matrix=rc) for rc in rec_matrices],
        out_layer=snn.Leaky(beta=beta, init_hidden=True, output=True),
    )