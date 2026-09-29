"""
Based on:
B. Cramer, Y. Stradmann, J. Schemmel and F. Zenke, "The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 7, pp. 2744-2757, July 2022, doi: 10.1109/TNNLS.2020.3044364.
keywords: {Benchmark testing;Task analysis;Biological neural networks;Training;Licenses;Encoding;Voltage control;Audio;benchmark;classification;data set;neuromorphic computing;spiking neural networks;spoken digits;surrogate gradients},
"""

import torch
from torch import Tensor
import torch.nn as nn
import snntorch as snn
from snntorch import utils

class KaimingLinear(nn.Linear):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__(n_in, n_out, bias=False)
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

class KaimingSRNN(nn.Module):
    def __init__(self, n_in: int, ns_hidden: list[int], n_out: int, beta: float) -> None:
        super().__init__()
        layers = [n_in] + ns_hidden
        modules = [nn.Flatten()]

        for _n_in, _n_out in zip(layers[:-1], layers[1:]):
            modules.append(KaimingLinear(_n_in, _n_out))
            modules.append(snn.RLeaky(beta=beta, linear_features=_n_out, init_hidden=True))

        modules.append(KaimingLinear(ns_hidden[-1], n_out))
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