from typing import Tuple

import torch
import transformers
from models import SiameseContrastiveBERT


class SiameseTripletBERT(SiameseContrastiveBERT):
    """
    Siamese BERT with Triplet Loss.
    """

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
