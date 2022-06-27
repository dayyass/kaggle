from typing import Dict, List, Tuple

import torch
import transformers


class SiameseTripletBERT(torch.nn.Module):
    """
    Siamese BERT with Triplet Loss.
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
        anchor: transformers.BatchEncoding,
        positive: transformers.BatchEncoding,
        negative: transformers.BatchEncoding,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with anchor, positive and negative tokenized sentences.

        Args:
            anchor (transformers.BatchEncoding): anchor sentences.
            positive (transformers.BatchEncoding): positive sentences.
            negative (transformers.BatchEncoding): negative sentences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: anchor, positive and negative embeddings.
        """

        anchor = self._vectorize(anchor)
        positive = self._vectorize(positive)
        negative = self._vectorize(negative)

        return anchor, positive, negative

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

    @staticmethod
    def similarity(
        x1_emb: torch.Tensor,
        x2_emb: torch.Tensor,
    ) -> torch.Tensor:
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

        euclidean_distance = torch.nn.functional.pairwise_distance(x1_emb, x2_emb, p=2)
        return torch.exp(-euclidean_distance)
