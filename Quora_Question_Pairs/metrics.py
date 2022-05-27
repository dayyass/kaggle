from typing import Dict

import numpy as np
import pandas as pd
import torch
import transformers
from model import SiameseManhattanBERT
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

tqdm.pandas()


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true (np.ndarray): targets.
        y_score (np.ndarray): prediction scores.

    Returns:
        Dict[str, float]: metrics.
    """

    y_pred = (y_score > 0.5).astype('int')

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_score)
    logloss = log_loss(y_true=y_true, y_pred=y_score)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": logloss,
    }

    return metrics


def compute_metrics_on_df(
    model: SiameseManhattanBERT,
    df: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_kwargs: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute binary classification metrics on dataframe.

    Args:
        model (SiameseManhattanBERT): model.
        df (pd.DataFrame): Quora Question Pairs dataframe.
        tokenizer (transformers.PreTrainedTokenizer): transformers tokenizer.
        tokenizer_kwargs (Dict[str, int]): transformers parameters.

    Returns:
        Dict[str, float]: metrics.
    """

    model.eval()

    with torch.no_grad():
        tqdm.pandas(desc="vectorize question1")
        q1_emb = df["question1"].progress_apply(
            lambda text: model.vectorize(text, tokenizer, tokenizer_kwargs),
        )
        q1_emb = np.array(q1_emb.to_list())

        tqdm.pandas(desc="vectorize question2")
        q2_emb = df["question2"].progress_apply(
            lambda text: model.vectorize(text, tokenizer, tokenizer_kwargs)
        )
        q2_emb = np.array(q2_emb.to_list())

    y_true = df["is_duplicate"].values
    y_score = model._exponent_neg_manhattan_distance(q1_emb, q2_emb, type="np")

    return compute_metrics(y_true=y_true, y_score=y_score)
