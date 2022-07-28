import torch
from nn_modules._base import IntertowerPoolerBase


class IntertowerConcatPooler(IntertowerPoolerBase):
    """
    Intertower concatenation pooler.
    """

    def __init__(self, hidden_size: int, dropout_p: float):
        super().__init__(
            hidden_size=hidden_size,
            dropout_p=dropout_p,
            mult_param=2,  # hardcode
        )

    @staticmethod
    def _concat(
        x1_emb: torch.Tensor,
        x2_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenation.
        """

        return torch.cat([x1_emb, x2_emb], dim=-1)


class IntertowerConcatPoolerWithAbsDiff(IntertowerPoolerBase):
    """
    Intertower concatenation pooler with absolute difference of embeddings.
    """

    def __init__(self, hidden_size: int, dropout_p: float):
        super().__init__(
            hidden_size=hidden_size,
            dropout_p=dropout_p,
            mult_param=3,  # hardcode
        )

    @staticmethod
    def _concat(
        x1_emb: torch.Tensor,
        x2_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenation with absolute difference.
        """

        abs_diff = torch.abs(x1_emb - x2_emb)
        return torch.cat([x1_emb, x2_emb, abs_diff], dim=-1)


class IntertowerConcatPoolerWithAbsDiffAndProduct(IntertowerPoolerBase):
    """
    Intertower concatenation pooler with absolute difference of embeddings and element-wise product.
    """

    def __init__(self, hidden_size: int, dropout_p: float):
        super().__init__(
            hidden_size=hidden_size,
            dropout_p=dropout_p,
            mult_param=4,  # hardcode
        )

    @staticmethod
    def _concat(
        x1_emb: torch.Tensor,
        x2_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenation with absolute difference and element-wise product.
        """

        abs_diff = torch.abs(x1_emb - x2_emb)
        prod = x1_emb * x2_emb
        return torch.cat([x1_emb, x2_emb, abs_diff, prod], dim=-1)
