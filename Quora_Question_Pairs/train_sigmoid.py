from collections import defaultdict

import numpy as np
import torch
from metrics import compute_metrics
from nn_modules.models import SiameseSigmoidBERT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    n_epochs: int,
    model: SiameseSigmoidBERT,
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
        model (SiameseSigmoidBERT): model.
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
    model: SiameseSigmoidBERT,
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
        model (SiameseSigmoidBERT): model.
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

    for i, (q1, q2, tgt) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):

        q1, q2, tgt = q1.to(device), q2.to(device), tgt.to(device)

        optimizer.zero_grad()

        logits = model(q1, q2)
        loss = criterion(logits, tgt)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar(
            "batch loss / train", loss.item(), epoch * len(dataloader) + i
        )

        with torch.no_grad():
            model.eval()
            scores_inference = model(q1, q2).sigmoid()
            model.train()

        batch_metrics = compute_metrics(
            outputs=scores_inference,
            targets=tgt,
        )

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
    model: SiameseSigmoidBERT,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One evaluation cycle (loop).

    Args:
        model (SiameseSigmoidBERT): model.
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

        for i, (q1, q2, tgt) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):

            q1, q2, tgt = q1.to(device), q2.to(device), tgt.to(device)

            logits = model(q1, q2)
            loss = criterion(logits, tgt)

            epoch_loss.append(loss.item())
            writer.add_scalar(
                "batch loss / test", loss.item(), epoch * len(dataloader) + i
            )

            batch_metrics = compute_metrics(
                outputs=logits.sigmoid(),
                targets=tgt,
            )

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
