from typing import Dict, Tuple

import pandas as pd
import torch
import transformers


class Dataset(torch.utils.data.Dataset):
    """
    Predict Closed Questions on Stack Overflow dataset.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Dataset initialization.

        Args:
            df (pd.DataFrame): Predict Closed Questions on Stack Overflow dataframe.
        """

        self.df = df

    def __len__(self) -> int:
        """
        Dataset length.

        Returns:
            int: dataset length.
        """

        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """
        Get dataframe row with question and target.

        Args:
            idx (int): row index.

        Returns:
            Tuple[str, int]: question and target.
        """

        row = self.df.iloc[idx]

        title = row["Title"]
        body = row["BodyMarkdown"]
        tgt = row["OpenStatus"]

        return f"{title} {body}", tgt


class Collator:
    """
    Predict Closed Questions on Stack Overflow collator that tokenizes sentences.
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
        self, batch: Tuple[str, int]
    ) -> Tuple[transformers.BatchEncoding, torch.LongTensor]:
        """
        Create tokenized batch from strings.

        Args:
            batch (Tuple[str, int]): source strings data.

        Returns:
            Tuple[transformers.BatchEncoding, torch.LongTensor]: tokenized batch.
        """

        txt, tgt = zip(*batch)  # type: ignore

        txt = self.tokenizer(list(txt), **self.tokenizer_kwargs)
        tgt = torch.LongTensor(tgt)

        return txt, tgt
