from typing import Dict

import torch
from model import SiameseManhattanBERT
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
        q1 = df["question1"].progress_apply(
            lambda text: model.vectorize(text, tokenizer, tokenizer_kwargs),
        )

        tqdm.pandas(desc="vectorize question2")
        q2 = df["question2"].progress_apply(
            lambda text: model.vectorize(text, tokenizer, tokenizer_kwargs)
        )

    y_true = df["is_duplicate"].values

    scores = model._exponent_neg_manhattan_distance(q1, q2, type="np")
    y_pred = (scores > 0.5).astype("int")

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return metrics
