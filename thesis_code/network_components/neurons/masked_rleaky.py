import torch
import snntorch as snn

from ..linears import MaskedKaimingLinear

class MaskedRLeaky(snn.RLeaky):
    def __init__(self, *args, rec_matrix: torch.Tensor, **kwargs) -> None:
        assert rec_matrix.shape[0] == rec_matrix.shape[1], \
            f"expected a square matrix, got shape {tuple(rec_matrix.shape)}"

        super().__init__(*args, **kwargs)
        self.recurrent = MaskedKaimingLinear(rec_matrix)
        if not self.learn_recurrent:
            for param in self.recurrent.parameters():
                param.requires_grad = False

    def reset_recurrent_weights(self) -> None:
        self.recurrent.reset()