import torch


class IntertowerConcatPooler(torch.nn.Module):
    """
    Intertower concatenation pooler.
    """

    def __init__(self, hidden_size: int, dropout_p: float):
        super().__init__()
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size * 2

        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.pooler = torch.nn.Linear(self.hidden_size, 1)

    def _concat(self, x1_emb: torch.Tensor, x2_emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([x1_emb, x2_emb], dim=-1)

    def forward(self, x1_emb: torch.Tensor, x2_emb: torch.Tensor) -> torch.Tensor:
        concat = self._concat(x1_emb, x2_emb)
        dropout_concat = self.dropout(concat)
        return self.pooler(dropout_concat).squeeze(-1)  # logit


class IntertowerConcatPoolerWithAbsDiff(torch.nn.Module):
    """
    Intertower concatenation pooler with absolute difference of embeddings.
    """

    def __init__(self, hidden_size: int, dropout_p: float):
        super().__init__()
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size * 3

        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.pooler = torch.nn.Linear(self.hidden_size, 1)

    def _concat(self, x1_emb: torch.Tensor, x2_emb: torch.Tensor) -> torch.Tensor:
        x3_abs_diff = torch.abs(x1_emb - x2_emb)
        return torch.cat([x1_emb, x2_emb, x3_abs_diff], dim=-1)

    def forward(self, x1_emb: torch.Tensor, x2_emb: torch.Tensor) -> torch.Tensor:
        concat = self._concat(x1_emb, x2_emb)
        dropout_concat = self.dropout(concat)
        return self.pooler(dropout_concat).squeeze(-1)  # logit


class IntertowerConcatPoolerWithAbsDiffAndProduct(torch.nn.Module):
    """
    Intertower concatenation pooler with absolute difference of embeddings and element-wise product.
    """

    def __init__(self, hidden_size: int, dropout_p: float):
        super().__init__()
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size * 4

        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.pooler = torch.nn.Linear(self.hidden_size, 1)

    def _concat(self, x1_emb: torch.Tensor, x2_emb: torch.Tensor) -> torch.Tensor:
        x3_abs_diff = torch.abs(x1_emb - x2_emb)
        x4_prod = x1_emb * x2_emb
        return torch.cat([x1_emb, x2_emb, x3_abs_diff, x4_prod], dim=-1)

    def forward(self, x1_emb: torch.Tensor, x2_emb: torch.Tensor) -> torch.Tensor:
        concat = self._concat(x1_emb, x2_emb)
        dropout_concat = self.dropout(concat)
        return self.pooler(dropout_concat).squeeze(-1)  # logit
