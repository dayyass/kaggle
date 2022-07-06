from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import transformers


class SiameseBase(torch.nn.Module, ABC):
    """
    Siamese Abstract Base Class.
    """

    def __init__(
        self,
        bert_model: transformers.PreTrainedModel,
        pooler: torch.nn.Module,
    ):
        """
        Model initialization with BERT model and pooler.

        Args:
            bert_model (transformers.PreTrainedModel): BERT model.
            pooler (torch.nn.Module): pooler to get embeddings from bert_model output.
        """

        super().__init__()

        self.bert_model = bert_model
        self.pooler = pooler

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

        return self.pooler(self.bert_model(**x))

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

        device = self.bert_model.device

        tokens = tokenizer(texts, **tokenizer_kwargs).to(device)
        with torch.no_grad():
            embedding = self._vectorize(tokens)

        return embedding

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


class IntertowerPoolerBase(torch.nn.Module, ABC):
    """
    Intertower Pooler Abstract Base Class.
    """

    def __init__(self, hidden_size: int, dropout_p: float, mult_param: int):
        super().__init__()
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size * mult_param

        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.pooler = torch.nn.Linear(self.hidden_size, 1)

    @staticmethod
    @abstractmethod
    def _concat(x1_emb: torch.Tensor, x2_emb: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, x1_emb: torch.Tensor, x2_emb: torch.Tensor) -> torch.Tensor:
        concat = self._concat(x1_emb, x2_emb)
        dropout_concat = self.dropout(concat)
        return self.pooler(dropout_concat).squeeze(-1)  # logit
