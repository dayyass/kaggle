from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import transformers


class SiameseManhattanBERT(torch.nn.Module):
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
        return self.exponent_neg_manhattan_distance(x1, x2)

    def vectorize(
        self,
        texts: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        tokenizer_kwargs: Dict[str, int],
    ) -> np.ndarray:
        """
        Inference-time method to get text embedding.

        Args:
            texts (List[str]): list of sentences.
            tokenizer (transformers.PreTrainedTokenizer): transformers tokenizer.
            tokenizer_kwargs (Dict[str, int]): transformers parameters.

        Returns:
            np.ndarray: text embedding.
        """

        device = self.bert_model.device

        tokens = tokenizer(texts, **tokenizer_kwargs).to(device)
        with torch.no_grad():
            embedding = self._vectorize(tokens)

        return embedding.cpu().numpy()

    def _vectorize(self, x: transformers.BatchEncoding) -> torch.Tensor:
        """
        Get embedding from tokenized sentences x.

        Args:
            x (transformers.BatchEncoding): tokenized sentences.

        Returns:
            torch.Tensor: embedding.
        """

        return self.pooler(self.bert_model(**x))

    @staticmethod
    def exponent_neg_manhattan_distance(
        x1_emb: Union[np.ndarray, torch.Tensor],
        x2_emb: Union[np.ndarray, torch.Tensor],
        type: str = "pt",
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Calculate semantic similarity scores given embedding of the sentences batches.

        Args:
            x1_emb (Union[np.ndarray, torch.Tensor]): embeddings of sentences batch.
            x2_emb (Union[np.ndarray, torch.Tensor]): embeddings of sentences batch.
            type (str, optional): torch or numpy calculation. Defaults to 'pt'.

        Raises:
            ValueError: only 'pt' (torch) and 'np' (numpy) values are allowed.

        Returns:
            Union[np.ndarray, torch.Tensor]: semantic similarity scores.
        """

        if type == "pt":
            manhattan_distance = torch.nn.functional.pairwise_distance(
                x1_emb, x2_emb, p=1
            )
            scores = torch.exp(-manhattan_distance)
        elif type == "np":
            manhattan_distance = np.linalg.norm(x1_emb - x2_emb, ord=1, axis=1)
            scores = np.exp(-manhattan_distance)
        else:
            raise ValueError(f"type '{type}' is not known, use 'pt' or 'np'")
        return scores


class SiameseSigmoidBERT(torch.nn.Module):
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

    def _vectorize(self, x: transformers.BatchEncoding) -> torch.Tensor:
        """
        Get embedding from tokenized sentences x.

        Args:
            x (transformers.BatchEncoding): tokenized sentences.

        Returns:
            torch.Tensor: embedding.
        """

        return self.pooler(self.bert_model(**x))

    def similarity_sigmoid_score(
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


class SiameseContrastiveBERT(torch.nn.Module):
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

    def _vectorize(self, x: transformers.BatchEncoding) -> torch.Tensor:
        """
        Get embedding from tokenized sentences x.

        Args:
            x (transformers.BatchEncoding): tokenized sentences.

        Returns:
            torch.Tensor: embedding.
        """

        return self.pooler(self.bert_model(**x))

    @staticmethod
    def exponent_neg_manhattan_distance(
        x1_emb: Union[np.ndarray, torch.Tensor],
        x2_emb: Union[np.ndarray, torch.Tensor],
        type: str = "pt",
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Calculate semantic similarity scores given embedding of the sentences batches.

        Args:
            x1_emb (Union[np.ndarray, torch.Tensor]): embeddings of sentences batch.
            x2_emb (Union[np.ndarray, torch.Tensor]): embeddings of sentences batch.
            type (str, optional): torch or numpy calculation. Defaults to 'pt'.

        Raises:
            ValueError: only 'pt' (torch) and 'np' (numpy) values are allowed.

        Returns:
            Union[np.ndarray, torch.Tensor]: semantic similarity scores.
        """

        if type == "pt":
            manhattan_distance = torch.nn.functional.pairwise_distance(
                x1_emb, x2_emb, p=1
            )
            scores = torch.exp(-manhattan_distance)
        elif type == "np":
            manhattan_distance = np.linalg.norm(x1_emb - x2_emb, ord=1, axis=1)
            scores = np.exp(-manhattan_distance)
        else:
            raise ValueError(f"type '{type}' is not known, use 'pt' or 'np'")
        return scores


# TOWER POOLERS

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


# INTERTOWER POOLERS

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
