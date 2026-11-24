import torch
from snntorch.surrogate import fast_sigmoid

from pathlib import Path
from typing import Union, Literal, Optional
from dataclasses import dataclass, field

from . import SRNN
from .network_components.layers import StandardRecurrentLayer, SynapticRecurrentLayer, ALIFRecurrentLayer, ALIFFloatingBetaRecurrentLayer, OutputLayer, LAYER_CLASSES
from .neuron_emulation.neurons import Neuron, sample_exc_pv_som_neurons, sample_exc_inh_neurons, sample_exc_neurons, save_neurons
from .neuron_emulation.connectivity import build_recurrent_and_pruned_matrices, create_forward_matrix_output, save_matrices
from .neuron_emulation.decays import get_membrane_decays_ee_pv_som, get_floating_beta_ee_pv_som
from .parameter_sampling import tau_to_beta, sample_values, SamplingConfig, sample_heterogeneous_normal

Layer = Union[StandardRecurrentLayer, SynapticRecurrentLayer, ALIFRecurrentLayer]

@dataclass
class RecurrentLayerConfig:
    n_neurons: int
    p_pv: float
    p_som: float
    p_inh: float
    use_daleslaw: bool
    use_connectivity: bool
    refractory_period: int
    rec_layer_type: Literal["standard", "synaptic", "alif"]
    beta: SamplingConfig = field(default_factory=SamplingConfig)
    learn_beta: bool = True
    beta_w: Optional[SamplingConfig] = None
    a: Optional[SamplingConfig] = None
    b: Optional[SamplingConfig] = None

    def __post_init__(self) -> None:
        required = {"beta_w", "a", "b"} if self.rec_layer_type == "alif" else set()
        set_params = {p for p in ("beta_w", "a", "b") if getattr(self, p) is not None}
        if set_params != required:
            raise ValueError(f"{self.rec_layer_type} requires exactly {sorted(required)} "
                             f"to be set, but got {sorted(set_params)}")

@dataclass
class SRNNConfig:
    recurrent_layers: list[RecurrentLayerConfig]
    add_control_net: bool
    tau_out: float  # ms

def build_layer(rl_cfg: RecurrentLayerConfig, neurons_prev: list[Neuron], dt: float, output_dir: Path, device: torch.device) -> tuple[list[Neuron], Layer, torch.Tensor, Layer | None, torch.Tensor | None]:
    n_neurons = rl_cfg.n_neurons
    if rl_cfg.use_daleslaw and rl_cfg.use_connectivity:
        neurons = sample_exc_pv_som_neurons(n_neurons, rl_cfg.p_pv, rl_cfg.p_som)
    else:
        neurons = sample_exc_inh_neurons(n_neurons, rl_cfg.p_inh) if rl_cfg.use_connectivity else sample_exc_neurons(n_neurons)
    
    save_neurons(output_dir, neurons)
    rec_matrix, pruned_matrix = build_recurrent_and_pruned_matrices(neurons)

    def _build_layer(rec_matrix: torch.Tensor) -> Layer:
        return ALIFFloatingBetaRecurrentLayer(
            forward_matrix=torch.ones(len(neurons_prev), n_neurons, device=device),
            recurrent_matrix=rec_matrix.to(device),
            floating_beta=get_floating_beta_ee_pv_som(neurons, dt, device),
            beta_w=sample_heterogeneous_normal(dt, n_neurons, tau_mean=30, tau_std=5, device=device),
            a=sample_heterogeneous_normal(dt, n_neurons, tau_mean=3, tau_std=2, device=device),
            b=sample_heterogeneous_normal(dt, n_neurons, tau_mean=1, tau_std=1, device=device),
            spike_grad=fast_sigmoid(slope=25),
            threshold=torch.ones(n_neurons, device=device),
            refractory_period=0,
            use_daleslaw=rl_cfg.use_daleslaw,
        ).to(device)

    return neurons, _build_layer(rec_matrix), rec_matrix, _build_layer(pruned_matrix), pruned_matrix

def build_srnns(
    dt: float,
    input_dim: int,
    output_dim: int,
    recurrent_layer_cfgs: list[RecurrentLayerConfig],
    tau_out: float,
    device: torch.device,
    output_dir: Path,
    add_control_net: bool,
) -> tuple[SRNN, SRNN | None]:
    layer_outs = []
    neurons_prev = [Neuron.EXCITATORY] * input_dim
    for rl_cfg in recurrent_layer_cfgs:
        neurons_prev, *other = build_layer(rl_cfg, neurons_prev, dt, output_dir, device)
        layer_outs.append(other)
    rec_layers, rec_matrices, rec_layers_control, pruned_matrices = map(list, zip(*layer_outs))
    forward_matrix_output = create_forward_matrix_output(neurons_prev, output_dim, device=device)

    names=[f"recurrent_mask_{i}" for i, _ in enumerate(rec_matrices)] + ["forward_mask_1"]
    save_matrices(output_dir, rec_matrices + [forward_matrix_output], names, is_control=False)
    if add_control_net:
        save_matrices(output_dir, pruned_matrices + [forward_matrix_output], names, is_control=True)

    beta_out = tau_to_beta(dt, tau_out).to(device)
    net = SRNN(rec_layers, out_layer=OutputLayer(forward_matrix_output.clone(), beta_out).to(device))
    net_control = SRNN(rec_layers_control, out_layer=OutputLayer(forward_matrix_output.clone(), beta_out).to(device)) if add_control_net else None
    return net, net_control