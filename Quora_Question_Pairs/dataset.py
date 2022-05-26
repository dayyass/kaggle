from typing import Dict, Tuple

import pandas as pd
import torch
import transformers


class Dataset(torch.utils.data.Dataset):
    """
    Quora Question Pairs dataset.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Dataset initialization.

        Args:
            df (pd.DataFrame): Quora Question Pairs dataframe.
        """

        self.df = df

    def __len__(self) -> int:
        """
        Dataset length.

        Returns:
            int: dataset length.
        """

        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        """
        Get dataframe row with question1, question2 and target.

        Args:
            idx (int): row index.

        Returns:
            Tuple[str, str, int]: question1, question2 and target.
        """

        row = self.df.iloc[idx]

        q1 = row["question1"]
        q2 = row["question2"]
        tgt = row["is_duplicate"]

        return q1, q2, tgt


class Collator:
    """
    Quora Question Pairs collator that tokenizes sentences.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        tokenizer_kwargs: Dict[str, int],
    ):
        """
        Collator initialization with transformers tokenizer and its parameters.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): transformers tokenizer.
            tokenizer_kwargs (Dict[str, int]): transformers parameters.
        """

        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(
        self, batch: Tuple[str, str, int]
    ) -> Tuple[transformers.BatchEncoding, transformers.BatchEncoding, torch.Tensor]:
        """
        Create tokenized batch from strings.

        Args:
            batch (Tuple[str, str, int]): source strings data.

        Returns:
            Tuple[transformers.BatchEncoding, transformers.BatchEncoding, torch.Tensor]: tokenized batch.
        """

        q1, q2, tgt = zip(*batch)  # type: ignore

        q1 = self.tokenizer(list(q1), **self.tokenizer_kwargs)
        q2 = self.tokenizer(list(q2), **self.tokenizer_kwargs)
        tgt = torch.Tensor(tgt)

        return q1, q2, tgt
