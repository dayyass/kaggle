from typing import List

import torch
import transformers


class ClsPooler(torch.nn.Module):
    """
    CLS pooler.
    """

    def forward(
        self,
        x: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
    ) -> torch.Tensor:
        return x["pooler_output"]


class MeanPooler(torch.nn.Module):
    """
    last_hidden_state average pooler.
    """

    def forward(
        self,
        x: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
    ) -> torch.Tensor:
        return x["last_hidden_state"].mean(dim=1)


class MaxPooler(torch.nn.Module):
    """
    last_hidden_state max-over-time pooler.
    """

    def forward(
        self,
        x: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
    ) -> torch.Tensor:
        return x["last_hidden_state"].max(dim=1)[0]


class ConcatPooler(torch.nn.Module):
    """
    Concatenation pooler for other poolers.
    """

    def __init__(self, poolers: List[torch.nn.Module]):
        """
        ConcatPooler initialization.

        Args:
            poolers (List[torch.nn.Module]): list of poolers to concatenate.
        """

        super().__init__()

        self.poolers = poolers

    def forward(
        self,
        x: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
    ) -> torch.Tensor:
        """
        Concatenate poolers results.

        Args:
            x (transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions): bert_model output.

        Returns:
            torch.Tensor: embedding.
        """

        pool_list = []
        for pooler in self.poolers:
            pool_list.append(pooler(x))
        return torch.cat(pool_list, dim=-1)
