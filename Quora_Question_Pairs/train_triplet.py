from collections import defaultdict

import numpy as np
import torch
from metrics import _compute_metrics
from nn_modules.triplet_models import SiameseTripletBERT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    n_epochs: int,
    model: SiameseTripletBERT,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
) -> None:
    """
    Training loop.

    Args:
        n_epochs (int): number of epochs to train.
        model (SiameseTripletBERT): model.
        train_dataloader (torch.utils.data.DataLoader): train_dataloader.
        test_dataloader (torch.utils.data.DataLoader): test_dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
    """

    for epoch in range(n_epochs):

        print(f"Epoch [{epoch+1} / {n_epochs}]\n")

        train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
            device=device,
            epoch=epoch,
        )
        evaluate_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            writer=writer,
            device=device,
            epoch=epoch,
        )


def train_epoch(
    model: SiameseTripletBERT,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One training cycle (loop).

    Args:
        model (SiameseTripletBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.train()

    epoch_loss = []
    batch_metrics_list = defaultdict(list)

    for i, (anchor, positive, negative) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):

        anchor, positive, negative = (
            anchor.to(device),
            positive.to(device),
            negative.to(device),
        )

        optimizer.zero_grad()

        anchor_loss, positive_loss, negative_loss = model(anchor, positive, negative)
        loss = criterion(anchor_loss, positive_loss, negative_loss)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar(
            "batch loss / train", loss.item(), epoch * len(dataloader) + i
        )

        with torch.no_grad():
            model.eval()

            anchor, positive, negative = model(anchor, positive, negative)

            y_score_positive_batch = model.similarity(anchor, positive).cpu().numpy()
            y_score_negative_batch = model.similarity(anchor, negative).cpu().numpy()

            model.train()

        y_true_positive_batch = np.ones_like(y_score_positive_batch)
        y_true_negative_batch = np.zeros_like(y_score_negative_batch)

        y_score_batch = np.concatenate([y_score_positive_batch, y_score_negative_batch])
        y_true_batch = np.concatenate([y_true_positive_batch, y_true_negative_batch])

        batch_metrics = _compute_metrics(y_true=y_true_batch, y_score=y_score_batch)

        for metric_name, metric_value in batch_metrics.items():
            batch_metrics_list[metric_name].append(metric_value)
            writer.add_scalar(
                f"batch {metric_name} / train",
                metric_value,
                epoch * len(dataloader) + i,
            )

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    writer.add_scalar("loss / train", avg_loss, epoch)

    for metric_name, metric_value_list in batch_metrics_list.items():
        metric_value = np.mean(metric_value_list)
        print(f"Train {metric_name}: {metric_value}\n")
        writer.add_scalar(f"{metric_name} / train", metric_value, epoch)


def evaluate_epoch(
    model: SiameseTripletBERT,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One evaluation cycle (loop).

    Args:
        model (SiameseTripletBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.eval()

    epoch_loss = []
    batch_metrics_list = defaultdict(list)

    with torch.no_grad():

        for i, (anchor, positive, negative) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):

            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            anchor, positive, negative = model(anchor, positive, negative)
            loss = criterion(anchor, positive, negative)

            epoch_loss.append(loss.item())
            writer.add_scalar(
                "batch loss / test", loss.item(), epoch * len(dataloader) + i
            )

            y_score_positive_batch = model.similarity(anchor, positive).cpu().numpy()
            y_true_positive_batch = np.ones_like(y_score_positive_batch)

            y_score_negative_batch = model.similarity(anchor, negative).cpu().numpy()
            y_true_negative_batch = np.zeros_like(y_score_negative_batch)

            y_score_batch = np.concatenate(
                [y_score_positive_batch, y_score_negative_batch]
            )
            y_true_batch = np.concatenate(
                [y_true_positive_batch, y_true_negative_batch]
            )

            batch_metrics = _compute_metrics(y_true=y_true_batch, y_score=y_score_batch)

            for metric_name, metric_value in batch_metrics.items():
                batch_metrics_list[metric_name].append(metric_value)
                writer.add_scalar(
                    f"batch {metric_name} / test",
                    metric_value,
                    epoch * len(dataloader) + i,
                )

        avg_loss = np.mean(epoch_loss)
        print(f"Test loss:  {avg_loss}\n")
        writer.add_scalar("loss / test", avg_loss, epoch)

        for metric_name, metric_value_list in batch_metrics_list.items():
            metric_value = np.mean(metric_value_list)
            print(f"Test {metric_name}: {metric_value}\n")
            writer.add_scalar(f"{metric_name} / test", metric_value, epoch)
