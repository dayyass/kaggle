from typing import Dict

import numpy as np
import torch
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
    model: SiameseManhattanBERT,
    dataloader: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall and f1 metrics.

    Args:
        model (SiameseManhattanBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.

    Returns:
        Dict[str, float]: metrics.
    """

    model.eval()

    df = dataloader.dataset.df

    tokenizer = dataloader.collate_fn.tokenizer
    tokenizer_kwargs = dataloader.collate_fn.tokenizer_kwargs

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

    scores = model._exponent_neg_manhattan_distance(q1_emb, q2_emb, type="np")
    y_pred = (scores > 0.5).astype("int")

    metric_kwargs = {
        'y_true': y_true,
        'y_pred': y_pred,
    }

    accuracy = accuracy_score(**metric_kwargs)
    precision = precision_score(**metric_kwargs, zero_division=0)
    recall = recall_score(**metric_kwargs, zero_division=0)
    f1 = f1_score(**metric_kwargs, zero_division=0)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    logloss = log_loss(**metric_kwargs)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": logloss,
    }

    return metrics
