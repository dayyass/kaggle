from pprint import pprint

import numpy as np
import torch
from metrics import compute_metrics
from model import SiameseManhattanBERT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


def train(
    model: SiameseManhattanBERT,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    n_epochs: int,
) -> None:
    """Training loop.

    Args:
        model (SiameseManhattanBERT): model.
        train_dataloader (torch.utils.data.DataLoader): train_dataloader.
        test_dataloader (torch.utils.data.DataLoader): test_dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        n_epochs (int): number of epochs to train.
    """

    for epoch in trange(n_epochs, desc="loop over epochs"):

        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
        )
        test_loss = evaluate_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            writer=writer,
        )

        print(f"Train loss: {train_loss}")
        print(f"Test loss:  {test_loss}")

        writer.add_scalar("epoch loss / train", train_loss, epoch)
        writer.add_scalar("epoch loss / test", test_loss, epoch)

        train_metrics = compute_metrics(model=model, dataloader=train_dataloader)
        test_metrics = compute_metrics(model=model, dataloader=test_dataloader)

        print("Train metrics:")
        pprint(train_metrics)

        print("Test metrics:")
        pprint(test_metrics)

        writer.add_scalar("epoch accuracy / train", train_metrics["accuracy"], epoch)
        writer.add_scalar("epoch accuracy / test", test_metrics["accuracy"], epoch)

        writer.add_scalar("epoch precision / train", train_metrics["precision"], epoch)
        writer.add_scalar("epoch precision / test", test_metrics["precision"], epoch)

        writer.add_scalar("epoch recall / train", train_metrics["recall"], epoch)
        writer.add_scalar("epoch recall / test", test_metrics["recall"], epoch)

        writer.add_scalar("epoch f1 / train", train_metrics["f1"], epoch)
        writer.add_scalar("epoch f1 / test", test_metrics["f1"], epoch)


def train_epoch(
    model: SiameseManhattanBERT,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
) -> float:
    """
    One training cycle (loop).

    Args:
        model (SiameseManhattanBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.

    Returns:
        float: average loss.
    """

    model.train()

    epoch_loss = []

    for i, (q1, q2, tgt) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):

        optimizer.zero_grad()

        pred = model(q1, q2)
        loss = criterion(pred, tgt)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar("batch loss / train", loss.item(), i)

    return np.mean(epoch_loss)


def evaluate_epoch(
    model: SiameseManhattanBERT,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
) -> float:
    """
    One evaluation cycle (loop).

    Args:
        model (SiameseManhattanBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.

    Returns:
        float: average loss.
    """

    model.eval()

    epoch_loss = []

    with torch.no_grad():

        for i, (q1, q2, tgt) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over train batches",
        ):

            pred = model(q1, q2)
            loss = criterion(pred, tgt)

            epoch_loss.append(loss.item())
            writer.add_scalar("batch loss / test", loss.item(), i)

    return np.mean(epoch_loss)
