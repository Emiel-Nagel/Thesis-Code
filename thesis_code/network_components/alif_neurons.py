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

my alif RLeaky implementation (both self-derived from SynapticFlow and taken from Baronig et al.):
U[t+1] = βU[t] + I_{\\rm in}[t+1] + V(S_{\\rm out}[t]) - RU_{\\rm thr} - (1-β)w(t)
W[t+1] = βw*W[t] + (1 - βw)(aU[t] + bS[t])

where βw is the adaptation decay constant, a the voltage coupling, b the spike coupling and S whether there is a spike at this timepoint.
"""

import torch
import torch.nn as nn
import snntorch as snn

from .linears import DaleslawLinear, MaskedKaimingLinear

class ALIFRLeaky(snn.RLeaky):
    def __init__(self, *args, beta_w: float | torch.Tensor, a: float | torch.Tensor, b: float | torch.Tensor, 
            learn_beta_w: bool = True, learn_a: bool = True, learn_b: bool = True, **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self._register_variable("beta_w", beta_w, learn_beta_w)
        self._register_variable("a", a, learn_a)
        self._register_variable("b", b, learn_b)

    def _register_variable(self, name: str, value: float | torch.Tensor, learnable: bool = True) -> None:
        value = torch.as_tensor(value, dtype=torch.float32)
        if learnable:
            self.register_parameter(name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def _init_mem(self):
        spk = torch.zeros(0)
        mem = torch.zeros(0)
        adpt = torch.zeros(0)

        self.register_buffer("spk", spk, False)
        self.register_buffer("mem", mem, False)
        self.register_buffer("adpt", adpt, False)

    def reset_mem(self):
        device = self.spk.device
        self.spk = torch.zeros(0, device=device)
        self.mem = torch.zeros(0, device=device)
        self.adpt = torch.zeros(0, device=device)
        return self.spk, self.mem, self.adpt

    def _step_membrane(self, beta_u: torch.Tensor, U: torch.Tensor, I: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Implements _base_state_function from RLeaky:
        U[t+1] = βU[t] + I_{\\rm in}[t+1] + V(S_{\\rm out}[t]) - RU_{\\rm thr} - W[t]
        """
        return beta_u*U + I + V - (1-beta_u)*W

    def _step_adaptation(self, beta_w: torch.Tensor, W: torch.Tensor, U: torch.Tensor, a: torch.Tensor, b: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Implements the following snntorch adaptation on τ_w * dw/dt = -w(t) + a*U[t] + b*S[t]:
        W[t+1] = βw*W[t] + (1 - βw)(aU[t] + bS[t])
        """
        return beta_w*W + a*U + b*S

    def _base_state_function(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Overwrites _base_state_function from snn.RLeaky with my custom adaptation that includes the adaptation parameter
        """
        base_fn = self._step_membrane(
            beta_u=self.beta.clamp(0, 1),
            U=self.mem,
            I=input_,
            V=self.recurrent(self.spk),
            W=self.adpt,
        )
        return base_fn

    def forward(self, input_: torch.Tensor, spk: torch.Tensor = None, mem: torch.Tensor = None, adpt: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if adpt is not None:
            self.adpt = adpt
        if not self.adpt.shape == input_.shape:
            self.adpt = torch.zeros_like(input_, device=self.adpt.device)

        mem_prev = mem if mem is not None else self.mem
        if mem_prev.shape != input_.shape:          # was: numel() == 0
            mem_prev = torch.zeros_like(input_)

        spk, mem = super().forward(input_, spk, mem)

        self.adpt = self._step_adaptation(
            beta_w=self.beta_w.clamp(0, 1),
            W=self.adpt,
            U=mem_prev,
            a=self.a,
            b=self.b,
            S=spk,
        )

        return spk, mem, self.adpt

class MaskedALIFRLeaky(ALIFRLeaky):
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

    def clamp_beta(self)
        
