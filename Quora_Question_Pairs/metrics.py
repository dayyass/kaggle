import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
import transformers
from nn_modules._base import SiameseBase
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from utils import chunks


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        outputs (torch.Tensor): targets.
        targets (torch.Tensor): model outputs.

    Returns:
        Dict[str, float]: metrics.
    """

    # TODO: maybe mode .long() into dataset
    y_true = targets.long().cpu().numpy()
    y_score = outputs.cpu().numpy()

    metrics = _compute_metrics(
        y_true=y_true,
        y_score=y_score,
    )

    return metrics


def _compute_metrics(
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

    y_pred = (y_score > 0.5).astype("int")

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_score)
    except ValueError:
        roc_auc = 0

    try:
        logloss = log_loss(y_true=y_true, y_pred=y_score)
    except ValueError:
        logloss = 0

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
    model: SiameseBase,
    df: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_kwargs: Dict[str, int],
    batch_size: int,
) -> Dict[str, float]:
    """
    Compute binary classification metrics on dataframe.

    Args:
        model (SiameseBase): model.
        df (pd.DataFrame): Quora Question Pairs dataframe.
        tokenizer (transformers.PreTrainedTokenizer): transformers tokenizer.
        tokenizer_kwargs (Dict[str, int]): transformers parameters.
        batch_size (int): batch size.

    Returns:
        Dict[str, float]: metrics.
    """

    model.eval()

    tqdm_total = math.ceil(len(df) / batch_size)

    q1_emb, q2_emb = [], []

    for texts in tqdm(
        chunks(df["question1"].to_list(), n=batch_size),
        total=tqdm_total,
        desc="vectorize question1",
    ):
        emb = model.vectorize(texts, tokenizer, tokenizer_kwargs)
        q1_emb.append(emb)

    for texts in tqdm(
        chunks(df["question2"].to_list(), n=batch_size),
        total=tqdm_total,
        desc="vectorize question2",
    ):
        emb = model.vectorize(texts, tokenizer, tokenizer_kwargs)
        q2_emb.append(emb)

    # TODO: optimize
    q1_emb = torch.vstack(q1_emb)
    q2_emb = torch.vstack(q2_emb)

    y_true = df["is_duplicate"].values
    y_score = model.similarity(q1_emb, q2_emb).cpu().numpy()

    return _compute_metrics(y_true=y_true, y_score=y_score)
