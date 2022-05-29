import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from utils import chunks


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute multiclass classification metrics.

    Args:
        y_true (np.ndarray): targets.
        y_pred (np.ndarray): predictions.

    Returns:
        Dict[str, float]: metrics.
    """

    metrics_kwargs = {
        'y_true': y_true,
        'y_pred': y_pred,
    }

    accuracy = accuracy_score(**metrics_kwargs)

    precision_micro = precision_score(
        **metrics_kwargs,
        average='micro',
        zero_division=0,
    )
    precision_macro = precision_score(
        **metrics_kwargs,
        average='macro',
        zero_division=0,
    )
    precision_weighted = precision_score(
        **metrics_kwargs,
        average='weighted',
        zero_division=0,
    )

    recall_micro = recall_score(
        **metrics_kwargs,
        average='micro',
        zero_division=0,
    )
    recall_macro = recall_score(
        **metrics_kwargs,
        average='macro',
        zero_division=0,
    )
    recall_weighted = recall_score(
        **metrics_kwargs,
        average='weighted',
        zero_division=0,
    )

    f1_micro = f1_score(
        **metrics_kwargs,
        average='micro',
        zero_division=0,
    )
    f1_macro = f1_score(
        **metrics_kwargs,
        average='macro',
        zero_division=0,
    )
    f1_weighted = f1_score(
        **metrics_kwargs,
        average='weighted',
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,

        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,

        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,

        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    return metrics


def compute_metrics_on_df(
    model: AutoModelForSequenceClassification,
    df: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_kwargs: Dict[str, int],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute milticlass classification metrics on dataframe.

    Args:
        model (AutoModelForSequenceClassification): model.
        df (pd.DataFrame): Predict Closed Questions on Stack Overflow dataframe.
        tokenizer (transformers.PreTrainedTokenizer): transformers tokenizer.
        tokenizer_kwargs (Dict[str, int]): transformers parameters.
        batch_size (int): batch size.
        device (torch.device): cpu or cuda.

    Returns:
        Dict[str, float]: metrics.
    """

    model.eval()

    tqdm_total = math.ceil(len(df) / batch_size)
    df_question = df['Title'] + ' ' + df['BodyMarkdown']

    y_pred = []
    for texts in tqdm(
        chunks(df_question.to_list(), n=batch_size),
        total=tqdm_total,
        desc='inference',
    ):
        tokens = tokenizer(texts, **tokenizer_kwargs)
        tokens = tokens.to(device)

        with torch.no_grad():
            logits = model(**tokens)['logits']
            y_pred_batch = logits.argmax(dim=-1).cpu().numpy()

        y_pred.append(y_pred_batch)

    y_true = df['OpenStatus'].values
    y_pred = np.concatenate(y_pred)

    return compute_metrics(y_true=y_true, y_pred=y_pred)
