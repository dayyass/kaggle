from typing import Dict, List, Tuple

import torch
import transformers
from _base import SiameseBase


class SiameseManhattanBERT(SiameseBase):
    """
    Siamese Manhattan LSTM but BERT is used instead of LSTM.
    https://blog.mlreview.com/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07
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

    def forward(
        self,
        x1: transformers.BatchEncoding,
        x2: transformers.BatchEncoding,
    ) -> torch.Tensor:
        """
        Forward pass with two tokenized sentences x1 and x2.

        Args:
            x1 (transformers.BatchEncoding): tokenized sentences.
            x2 (transformers.BatchEncoding): tokenized sentences.

        Returns:
            torch.Tensor: semantic similarity scores.
        """

        x1 = self._vectorize(x1)
        x2 = self._vectorize(x2)
        return self.similarity(x1, x2)

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

        return embedding.cpu()

    @staticmethod
    def similarity(
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

        manhattan_distance = torch.nn.functional.pairwise_distance(x1_emb, x2_emb, p=1)
        return torch.exp(-manhattan_distance)


class SiameseSigmoidBERT(SiameseBase):
    """
    Siamese Sigmoid BERT.
    """

    def __init__(
        self,
        bert_model: transformers.PreTrainedModel,
        pooler: torch.nn.Module,
        intertower_pooler: torch.nn.Module,
    ):
        """
        Model initialization with BERT model and pooler.

        Args:
            bert_model (transformers.PreTrainedModel): BERT model.
            pooler (torch.nn.Module): pooler to get embeddings from bert_model output.
            intertower_pooler (torch.nn.Module): intertower pooler to process two towers embedding.
        """

        super().__init__()

        self.bert_model = bert_model
        self.pooler = pooler
        self.intertower_pooler = intertower_pooler

    def forward(
        self,
        x1: transformers.BatchEncoding,
        x2: transformers.BatchEncoding,
    ) -> torch.Tensor:
        """
        Forward pass with two tokenized sentences x1 and x2.

        Args:
            x1 (transformers.BatchEncoding): tokenized sentences.
            x2 (transformers.BatchEncoding): tokenized sentences.

        Returns:
            torch.Tensor: semantic similarity logit.
        """

        x1 = self._vectorize(x1)
        x2 = self._vectorize(x2)
        return self.intertower_pooler(x1, x2)

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

    def similarity(
        self,
        x1_emb: torch.Tensor,
        x2_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate semantic similarity sigmoid scores given embedding of the sentences batches.

        Args:
            x1_emb (torch.Tensor): embeddings of sentences batch.
            x2_emb (torch.Tensor): embeddings of sentences batch.

        Returns:
            torch.Tensor: semantic similarity sigmoid scores.
        """

        with torch.no_grad():
            scores = self.intertower_pooler(x1_emb, x2_emb).sigmoid()

        return scores


class SiameseContrastiveBERT(SiameseBase):
    """
    Siamese BERT with Contrastive Loss.
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

    def forward(
        self,
        x1: transformers.BatchEncoding,
        x2: transformers.BatchEncoding,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with two tokenized sentences x1 and x2.

        Args:
            x1 (transformers.BatchEncoding): tokenized sentences.
            x2 (transformers.BatchEncoding): tokenized sentences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: x1 and x2 embeddings.
        """

        x1 = self._vectorize(x1)
        x2 = self._vectorize(x2)
        return x1, x2

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

    # TODO: validate
    @staticmethod
    def similarity(
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

        euclidean_distance = torch.nn.functional.pairwise_distance(x1_emb, x2_emb, p=2)
        return torch.exp(-euclidean_distance)
