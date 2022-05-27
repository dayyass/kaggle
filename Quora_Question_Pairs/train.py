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
    """
    Training loop.

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
    model: SiameseManhattanBERT,
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
        model (SiameseManhattanBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.train()

    epoch_loss = []
    y_true_list = []
    y_pred_list = []

    for i, (q1, q2, tgt) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):

        q1, q2, tgt = q1.to(device), q2.to(device), tgt.to(device)

        optimizer.zero_grad()

        scores = model(q1, q2)
        loss = criterion(scores, tgt)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar("batch loss / train", loss.item(), epoch * len(dataloader) + i)

        with torch.no_grad():
            y_true_batch = tgt.long().cpu().numpy()
            y_pred_batch = (scores > 0.5).long().cpu().numpy()

        y_true_list.append(y_true_batch)
        y_pred_list.append(y_pred_batch)

        batch_metrics = compute_metrics(y_true=y_true_batch, y_pred=y_pred_batch)

        for metric_name, metric_value in batch_metrics.items():
            writer.add_scalar(f"batch {metric_name} / train", metric_value, epoch * len(dataloader) + i)

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    writer.add_scalar("loss / train", avg_loss, epoch)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    print(f"Train metrics:\n{metrics}\n")

    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f"{metric_name} / train", metric_value, epoch)


def evaluate_epoch(
    model: SiameseManhattanBERT,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One evaluation cycle (loop).

    Args:
        model (SiameseManhattanBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.eval()

    epoch_loss = []
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():

        for i, (q1, q2, tgt) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):

            q1, q2, tgt = q1.to(device), q2.to(device), tgt.to(device)

            scores = model(q1, q2)
            loss = criterion(scores, tgt)

            epoch_loss.append(loss.item())
            writer.add_scalar("batch loss / test", loss.item(), epoch * len(dataloader) + i)

            y_true_batch = tgt.long().cpu().numpy()
            y_pred_batch = (scores > 0.5).long().cpu().numpy()

            y_true_list.append(y_true_batch)
            y_pred_list.append(y_pred_batch)

            batch_metrics = compute_metrics(y_true=y_true_batch, y_pred=y_pred_batch)

            for metric_name, metric_value in batch_metrics.items():
                writer.add_scalar(f"batch {metric_name} / test", metric_value, epoch * len(dataloader) + i)

        avg_loss = np.mean(epoch_loss)
        print(f"Test loss:  {avg_loss}\n")
        writer.add_scalar("loss / test", avg_loss, epoch)

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)

        metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
        print(f"Test metrics:\n{metrics}\n")

        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f"{metric_name} / test", metric_value, epoch)
