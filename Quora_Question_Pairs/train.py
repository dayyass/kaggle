import numpy as np
import torch
from metrics import compute_metrics
from model import SiameseManhattanBERT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    n_epochs: int,
    model: SiameseManhattanBERT,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
) -> None:
    """Training loop.

    Args:
        n_epochs (int): number of epochs to train.
        model (SiameseManhattanBERT): model.
        train_dataloader (torch.utils.data.DataLoader): train_dataloader.
        test_dataloader (torch.utils.data.DataLoader): test_dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
    """

    for epoch in range(n_epochs):

        print(f"Epoch [{epoch+1} / {n_epochs}]\n")

        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
            device=device,
            epoch=epoch,
        )
        test_loss = evaluate_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            writer=writer,
            device=device,
            epoch=epoch,
        )

        print(f"Train loss: {train_loss}")
        print(f"Test loss:  {test_loss}\n")

        writer.add_scalar("Loss / train", train_loss, epoch)
        writer.add_scalar("Loss / test", test_loss, epoch)

        train_metrics = compute_metrics(model=model, dataloader=train_dataloader)

        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(f"{metric_name} / train", metric_value, epoch)

        print(f"Train metrics:\n{train_metrics}\n")

        test_metrics = compute_metrics(model=model, dataloader=test_dataloader)

        for metric_name, metric_value in test_metrics.items():
            writer.add_scalar(f"{metric_name} / test", metric_value, epoch)

        print(f"Test metrics:\n{test_metrics}\n\n")


def train_epoch(
    model: SiameseManhattanBERT,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> float:
    """
    One training cycle (loop).

    Args:
        model (SiameseManhattanBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.

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

        q1, q2, tgt = q1.to(device), q2.to(device), tgt.to(device)

        optimizer.zero_grad()

        pred = model(q1, q2)
        loss = criterion(pred, tgt)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar("Batch loss / train", loss.item(), epoch * len(dataloader) + i)

    return np.mean(epoch_loss)


def evaluate_epoch(
    model: SiameseManhattanBERT,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> float:
    """
    One evaluation cycle (loop).

    Args:
        model (SiameseManhattanBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.

    Returns:
        float: average loss.
    """

    model.eval()

    epoch_loss = []

    with torch.no_grad():

        for i, (q1, q2, tgt) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):

            q1, q2, tgt = q1.to(device), q2.to(device), tgt.to(device)

            pred = model(q1, q2)
            loss = criterion(pred, tgt)

            epoch_loss.append(loss.item())
            writer.add_scalar("Batch loss / test", loss.item(), epoch * len(dataloader) + i)

    return np.mean(epoch_loss)
