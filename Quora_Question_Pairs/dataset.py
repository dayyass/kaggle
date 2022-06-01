from typing import Dict, List, Tuple

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


class TripletDataset(torch.utils.data.Dataset):
    """
    Quora Question Pairs triplet dataset.
    """

    def __init__(self, df: pd.DataFrame, n_negative_samples: int):
        """
        Dataset initialization.

        Args:
            df (pd.DataFrame): Quora Question Pairs dataframe.
        """

        self.df = df
        self.n_negative_samples = n_negative_samples

    def __len__(self) -> int:
        """
        Dataset length.

        Returns:
            int: dataset length.
        """

        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str], List[str]]:
        """
        Get dataframe row with anchor, positive and negative questions.

        Args:
            idx (int): row index.

        Returns:
            Tuple[List[str], List[str], List[str]]: anchor, positive and negative questions.
        """

        row = self.df.iloc[idx]

        anchor = row["question1"]
        positive = row["question2"]
        negative = self.df[
            self.df["question1"] != anchor
        ]["question2"].sample(self.n_negative_samples).to_list()

        anchor = [anchor] * self.n_negative_samples
        positive = [positive] * self.n_negative_samples

        return anchor, positive, negative


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


class TripletCollator:
    """
    Quora Question Pairs collator that tokenizes triplets.
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
        self, batch: Tuple[List[str], List[str], List[str]]
    ) -> Tuple[transformers.BatchEncoding, transformers.BatchEncoding, torch.Tensor]:
        """
        Create tokenized batch from strings.

        Args:
            batch (Tuple[List[str], List[str], List[str]]): source strings data triplets.

        Returns:
            Tuple[transformers.BatchEncoding, transformers.BatchEncoding, torch.Tensor]: tokenized batch.
        """

        anchor, positive, negative = zip(*batch)  # type: ignore

        anchor = self.tokenizer(self.flatten(anchor), **self.tokenizer_kwargs)  # type: ignore
        positive = self.tokenizer(self.flatten(positive), **self.tokenizer_kwargs)  # type: ignore
        negative = self.tokenizer(self.flatten(negative), **self.tokenizer_kwargs)  # type: ignore

        return anchor, positive, negative

    @staticmethod
    def flatten(list_of_lists: List[List[str]]) -> List[str]:
        return [item for lst in list_of_lists for item in lst]
