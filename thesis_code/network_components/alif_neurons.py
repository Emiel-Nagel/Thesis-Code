"""
Implementation based on:
https://arxiv.org/html/2408.07517v3
https://synapticflow.readthedocs.io/en/latest/ALIF.html



LIF neuron equation:
τ_u * du/dt = -u(t) + I(t)

ALIF neuron equation:
τ_u * du/dt = -u(t) + I(t) - w(t)
τ_w * dw/dt = -w(t) + a*u(t) + b*z(t)

snntorch.RLeaky implementation:
U[t+1] = βU[t] + I_{\\rm in}[t+1] + V(S_{\\rm out}[t]) - RU_{\\rm thr}

my alif RLeaky implementation:
U[t+1] = βU[t] + I_{\\rm in}[t+1] + V(S_{\\rm out}[t]) - RU_{\\rm thr} - w(t)
"""

import torch
import torch.nn as nn
import snntorch as snn

from .linears import DaleslawLinear, MaskedKaimingLinear

class ALIFRLeaky(snn.RLeaky):
    def __init__(self, *args, rec_matrix: torch.Tensor, use_daleslaw: bool = True, **kwargs) -> None:
        assert rec_matrix.shape[0] == rec_matrix.shape[1], \
            f"expected a square matrix, got shape {tuple(rec_matrix.shape)}"

        rec_matrix = rec_matrix.clone()
        rec_matrix.fill_diagonal_(0.0)
    
        super().__init__(*args, **kwargs)
        self.recurrent = DaleslawLinear(rec_matrix) if use_daleslaw else MaskedKaimingLinear(rec_matrix)
        self.recurrent.zero_diagonal()
        if not self.learn_recurrent:
            for param in self.recurrent.parameters():
                param.requires_grad = False

    def reset_recurrent_weights(self) -> None:
        self.recurrent.reset()

    def get_recurrent_weights(self) -> torch.Tensor:
        return self.recurrent.get_weights()

    def _step_membrane(self, beta_u: torch.Tensor, u: torch.Tensor, I: torch.Tensor, V: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Implements _base_state_function from RLeaky:
        U[t+1] = βU[t] + I_{\\rm in}[t+1] + V(S_{\\rm out}[t]) - RU_{\\rm thr} - w(t)
        """
        return beta_u * u + I + V - w

    def _step_adaptation(self, beta_w: torch.Tensor, w: torch.Tensor, u: torch.Tensor, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Implements the following snntorch adaptation on τ_w * dw/dt = -w(t) + a*u(t) + b*z(t):
        W[t+1] = β_w * W[t] + a*U[t] + b*Z[t]
        """
        return beta_w * w + a*u - b*z

    def forward(self, input_, spk=None, mem=None, adpt=None) -> ...:
        self.reset = self.mem_reset(self.mem)

        u_hat = self._step_membrane(
            beta_u=self.beta.clamp(0, 1),
            u=mem,
            I=input_,
            V=self.recurrent(self.spk),
            w=adpt
        )
        u = u_hat - self.reset * u_hat    # implements _base_zero from RLeaky
        spk = self.fire(self.mem)

        adpt = self._step_adaptation(
            beta_w=self.beta_w.clamp(0, 1),
            w=adpt,
            u=u,
            a=self.a,
            b=self.b,
            z=spk,
        )

        return spk, u, adpt
        # more code to come...