import torch.nn as nn
import torch

class MaskedLinear(nn.Module):
    def __init__(self, con_matrix: torch.Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(*con_matrix.shape, bias=False)
        self.register_buffer('con_matrix_t', con_matrix.T.contiguous())
        self._masked_weight = None

    def reset(self) -> None:
        self._masked_weight = None

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.clone().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._masked_weight is None:
            self._masked_weight = self.linear.weight * self.con_matrix_t    # we only compute this on the first forward, so it's reused until reset() is called after a learning step
        return nn.functional.linear(x, self._masked_weight)

class MaskedKaimingLinear(MaskedLinear):
    def __init__(self, con_matrix: torch.Tensor) -> None:
        super().__init__(con_matrix)
        nn.init.kaiming_uniform_(self.linear.weight, mode="fan_in", nonlinearity="relu")    # changed this in DaleslawLinear to compute my own fan_in, fan_in is lower due to my custom mask
        # nn.init.xavier_normal_(w)

        # explanation for fan_in versus fan_out (see https://docs.pytorch.org/docs/2.14/nn.init.html): 
        # fan_in is the input dim of the weights matrix, fan_out is the output dim

        # we want to keep the signal scale stable through the forward passes
        # say we have Var(y) = fan_in * Var(w) * Var(x), we want to keep the scales of Var(y) and Var(x) the same
        # so Var(w) must be 1/fan_in, so we get Var(y) = 1 * Var(x), which preserves the scale

        # but (except for recurrent layers) fan_in and fan_out are often not equal
        # if fan_in is set correctly for the forward pass, when we do the backwards pass, we get an undesired result, cuz we want Var(w) = 1 / fan_out for the backwards pass
        # so we make a tradeoff

        # xavier init is similar to kaiming init, except that xavier takes an average between fan_in and fan_out, so it gives half of both worlds
        # I choose to fully preserve the scale of the forward pass

        # might also use for recurrent layers: nn.init.orthogonal()

class DaleslawLinear(MaskedLinear):
    """
    For explanation, see:
    Barranca, V. J., Bhuiyan, A., Sundgren, M., & Xing, F. (2022). Functional implications of dale's law in balanced neuronal network dynamics and decision making. Frontiers in Neuroscience, 16. https://doi.org/10.3389/fnins.2022.801847
    """
    def __init__(self, con_matrix: torch.Tensor, w_min: float = 0.01, gain: float = 1.0) -> None:
        super().__init__(con_matrix)
        with torch.no_grad():                                           # applies my own custom kaiming init to compensate for fan_in loss from the mask
            mask = self.con_matrix_t != 0
            fan_in = mask.sum(dim=1, keepdim=True).clamp(min=1)
            std = gain / torch.sqrt(fan_in)
            std = std.to(self.linear.weight.device)
            self.linear.weight.copy_(torch.randn_like(self.linear.weight) * std)

        self.w_min = w_min
        self.apply_daleslaw()

    @torch.no_grad()
    def apply_daleslaw(self) -> None:
        """Call after every optimizer.step()."""
        self.linear.weight.abs_().clamp_(min=self.w_min)                # make all weights positive and nonzero (inhibition comes from the mask)
        self.reset()