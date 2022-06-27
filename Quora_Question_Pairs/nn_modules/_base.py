from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import transformers


class SiameseBase(torch.nn.Module, ABC):
    """
    Siamese Abstract Base Class.
    """

    @abstractmethod
    def _vectorize(
        self,
        x: transformers.BatchEncoding,
    ) -> torch.Tensor:
        """
        Get embedding from tokenized sentences x.

        Args:
            x (transformers.BatchEncoding): tokenized sentences.

        Returns:
            torch.Tensor: embedding.
        """

    @abstractmethod
    def vectorize(
        self,
        texts: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        tokenizer_kwargs: Dict[str, int],
    ) -> torch.Tensor:
        """
        Inference-time method to get text embedding.

        Args:
            texts (List[str]): list of sentences.
            tokenizer (transformers.PreTrainedTokenizer): transformers tokenizer.
            tokenizer_kwargs (Dict[str, int]): transformers parameters.

        Returns:
            torch.Tensor: text embedding.
        """

    @abstractmethod
    def similarity(
        self,
        x1_emb: torch.Tensor,
        x2_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate semantic similarity scores given embedding of the sentences batches.

        Args:
            x1_emb (torch.Tensor): embeddings of sentences batch.
            x2_emb (torch.Tensor): embeddings of sentences batch.

        Returns:
            torch.Tensor: semantic similarity scores.
        """
