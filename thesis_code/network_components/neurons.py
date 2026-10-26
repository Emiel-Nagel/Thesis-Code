import torch
import snntorch as snn

from .linears import DaleslawLinear, MaskedKaimingLinear

class MaskedRLeaky(snn.RLeaky):
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

class MaskedRSynaptic(snn.RSynaptic):
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