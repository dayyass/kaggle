import numpy as np
import torch
from metrics import compute_metrics
from model import SiameseContrastiveBERT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    n_epochs: int,
    model: SiameseContrastiveBERT,
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
        model (SiameseContrastiveBERT): model.
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
    model: SiameseContrastiveBERT,
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
        model (SiameseContrastiveBERT): model.
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
    y_score_list = []

    for i, (q1, q2, tgt) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):

        q1, q2, tgt = q1.to(device), q2.to(device), tgt.to(device)

        optimizer.zero_grad()

        q1, q2 = model(q1, q2)
        loss = criterion(q1, q2, tgt)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar("batch loss / train", loss.item(), epoch * len(dataloader) + i)

        with torch.no_grad():
            model.eval()

            q1, q2 = model(q1, q2)
            y_true_batch = tgt.long().cpu().numpy()
            y_score_batch = model.similarity_euclidean_score(q1, q2).cpu().numpy()

            model.train()

        y_true_list.append(y_true_batch)
        y_score_list.append(y_score_batch)

        batch_metrics = compute_metrics(y_true=y_true_batch, y_score=y_score_batch)

        for metric_name, metric_value in batch_metrics.items():
            writer.add_scalar(f"batch {metric_name} / train", metric_value, epoch * len(dataloader) + i)

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    writer.add_scalar("loss / train", avg_loss, epoch)

    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)

    metrics = compute_metrics(y_true=y_true, y_score=y_score)
    print(f"Train metrics:\n{metrics}\n")

    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f"{metric_name} / train", metric_value, epoch)

    writer.add_pr_curve('pr_curve / train', y_true, y_score, epoch)


def evaluate_epoch(
    model: SiameseContrastiveBERT,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One evaluation cycle (loop).

    Args:
        model (SiameseContrastiveBERT): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        criterion (torch.nn.Module): criterion.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.eval()

    epoch_loss = []
    y_true_list = []
    y_score_list = []

    with torch.no_grad():

        for i, (q1, q2, tgt) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):

            q1, q2, tgt = q1.to(device), q2.to(device), tgt.to(device)

            q1, q2 = model(q1, q2)
            loss = criterion(q1, q2, tgt)

            epoch_loss.append(loss.item())
            writer.add_scalar("batch loss / test", loss.item(), epoch * len(dataloader) + i)

            y_true_batch = tgt.long().cpu().numpy()
            y_score_batch = model.similarity_euclidean_score(q1, q2).cpu().numpy()

            y_true_list.append(y_true_batch)
            y_score_list.append(y_score_batch)

            batch_metrics = compute_metrics(y_true=y_true_batch, y_score=y_score_batch)

            for metric_name, metric_value in batch_metrics.items():
                writer.add_scalar(f"batch {metric_name} / test", metric_value, epoch * len(dataloader) + i)

        avg_loss = np.mean(epoch_loss)
        print(f"Test loss:  {avg_loss}\n")
        writer.add_scalar("loss / test", avg_loss, epoch)

        y_true = np.concatenate(y_true_list)
        y_score = np.concatenate(y_score_list)

        metrics = compute_metrics(y_true=y_true, y_score=y_score)
        print(f"Test metrics:\n{metrics}\n")

        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f"{metric_name} / test", metric_value, epoch)

        writer.add_pr_curve('pr_curve / test', y_true, y_score, epoch)
