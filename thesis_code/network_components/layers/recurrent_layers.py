import torch
import torch.nn as nn
from typing import Any

from ..neurons import MaskedRLeaky, MaskedRSynaptic
from ..linears import DaleslawLinear

class RecurrentLayer(nn.Module):
    def __init__(self,
        forward_matrix: torch.Tensor,
        recurrent_matrix: torch.Tensor,
        refractory_period: int = 0,
    ) -> None:
        super().__init__()
        self.n_neurons = recurrent_matrix.shape[0]

        assert forward_matrix.shape[1] == self.n_neurons, \
            f"forward_matrix outputs {forward_matrix.shape[1]} neurons, recurrent_matrix has {self.n_neurons}"

        self.weights = DaleslawLinear(forward_matrix)

        assert isinstance(refractory_period, int) and refractory_period >= 0, \
            "refractory_period is set incorrectly"

        self.refractory_period = refractory_period
        self.refractory_buffer: torch.Tensor | None = None

    def reset(self) -> None:
        self._reset_neuron_state()
        self.weights.reset()
        self.refractory_buffer = None

    def _reset_neuron_state(self) -> None:
        raise NotImplementedError

    def get_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weights.get_weights(), self.neurons.get_recurrent_weights()

    def _compute_refractory_period(self, spk: torch.Tensor, mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sets a mask to mute all spiking activity for a neuron activity for n timesteps after this neuron has spiked has occurred.
        Also keeps membrane potential at 0 (in snntorch language. This corresponds with -70mV in real-life).
        It leaves the synapses of SynapticRecurrentLayer alone.

        Inspired by:
        Li, Yang, Xinyi Zeng, Zhe Xue, Pinxian Zeng, Zikai Zhang, and Yan Wang. 2025. 
        “Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics.” 
        In arXiv [Cs.CV]. September 22. arXiv. https://doi.org/10.48550/arXiv.2509.17769.
        """
        if self.refractory_period <= 0:
            return spk, mem

        if self.refractory_buffer is None or self.refractory_buffer.shape != spk.shape:
            self.refractory_buffer = torch.zeros_like(spk, dtype=torch.int16)

        refractory = self.refractory_buffer > 0
        spk = spk.masked_fill(refractory, 0.0)                      # disable spiking for neurons in refractory periods
        mem = torch.where(refractory, torch.zeros_like(mem), mem)   # clamp mems to 0 for neurons in refractory periods

        self.refractory_buffer = torch.where(
            spk > 0,
            torch.full_like(self.refractory_buffer, self.refractory_period),    # set the buffer to the integer number of timesteps the buffer should apply if spiking
            (self.refractory_buffer - 1).clamp_min(0),                          # decrease the buffer by 1 for each timestep and clamp to 0
        )
        return spk, mem

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._step(self.weights(x))

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_seq = self.weights(x_seq)
        return torch.stack([self._step(x) for x in x_seq.unbind(0)], dim=0)

class StandardRecurrentLayer(RecurrentLayer):
    def __init__(self,
        forward_matrix: torch.Tensor,
        recurrent_matrix: torch.Tensor,
        beta: torch.Tensor | float,
        spike_grad: Any,
        fully_learnable: bool = True,
        refractory_period: int = 0,
    ) -> None:
        super().__init__(forward_matrix, recurrent_matrix, refractory_period)
        self.neurons = MaskedRLeaky(
            beta=beta,
            linear_features=self.n_neurons,
            init_hidden=False,
            spike_grad=spike_grad,
            learn_beta=fully_learnable,
            rec_matrix=recurrent_matrix,
            threshold=torch.ones(self.n_neurons),
        )
        self.spk, self.mem = self.neurons.init_rleaky()

    def _reset_neuron_state(self) -> None:
        self.spk, self.mem = self.neurons.init_rleaky()
        self.neurons.reset_recurrent_weights()

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        spk, mem = self.neurons(x, self.spk, self.mem)
        self.spk, self.mem = self._compute_refractory_period(spk, mem)
        return self.spk

class SynapticRecurrentLayer(RecurrentLayer):
    def __init__(self,
        forward_matrix: torch.Tensor,
        recurrent_matrix: torch.Tensor,
        alpha: torch.Tensor | float,
        beta: torch.Tensor | float,
        spike_grad: Any,
        fully_learnable: bool = True,
        refractory_period: int = 0, 
    ) -> None:
        super().__init__(forward_matrix, recurrent_matrix, refractory_period)
        self.neurons = MaskedRSynaptic(
            alpha=alpha,
            beta=beta,
            linear_features=self.n_neurons,
            init_hidden=False,
            spike_grad=spike_grad,
            learn_alpha=fully_learnable,
            learn_beta=fully_learnable,
            rec_matrix=recurrent_matrix,
            threshold=torch.ones(self.n_neurons),
        )
        self.spk, self.syn, self.mem = self.neurons.init_rsynaptic()

    def _reset_neuron_state(self) -> None:
        self.spk, self.syn, self.mem = self.neurons.init_rsynaptic()
        self.neurons.reset_recurrent_weights()

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        spk, self.syn, mem = self.neurons(x, self.spk, self.syn, self.mem)
        self.spk, self.mem = self._compute_refractory_period(spk, mem)
        return self.spk