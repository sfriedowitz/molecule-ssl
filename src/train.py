from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.vae import MolecularVAE
from src.tracker import MetricTracker


def reconstruction_count(x: torch.Tensor, x_recon: torch.Tensor):
    actual_labels = x.argmax(dim=-1)
    recon_labels = x_recon.argmax(dim=-1)
    return (recon_labels == actual_labels).all(dim=-1).sum()


def elbo_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    z_mean: torch.Tensor,
    z_logvar: torch.Tensor,
):
    # Cross entropy should be computed across one-hot labels,
    # so transpose tensors so labels in dim=1
    cross_entropy = F.cross_entropy(x_recon.transpose(2, 1), x.transpose(2, 1), reduction="sum")
    kld = -0.5 * torch.sum(1.0 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return cross_entropy + kld


def mse_loss(y: torch.Tensor, y_pred: torch.Tensor):
    return F.mse_loss(y_pred, y, reduction="sum")


def train_one_epoch(
    model: MolecularVAE,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    data_loader: DataLoader,
    include_mse: bool,
):
    model.train()

    metrics = {"train_elbo": 0.0, "train_mse": 0.0, "train_accuracy": 0.0}
    for x_batch, y_batch in data_loader:
        x_recon, y_pred, z_mean, z_logvar = model(x_batch)

        elbo = elbo_loss(x_batch, x_recon, z_mean, z_logvar)
        mse = mse_loss(y_batch, y_pred)
        recon = reconstruction_count(x_batch, x_recon)

        if include_mse:
            loss = elbo + mse
        else:
            loss = elbo

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics["train_elbo"] += elbo.item()
        metrics["train_mse"] += mse.item()
        metrics["train_accuracy"] += recon.item()

    if scheduler is not None:
        scheduler.step()

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def test_one_epoch(model: MolecularVAE, data_loader: DataLoader):
    model.eval()

    metrics = {"test_elbo": 0.0, "test_mse": 0.0, "test_accuracy": 0.0}
    for x_batch, y_batch in data_loader:
        x_recon, y_pred, z_mean, z_logvar = model(x_batch)
        elbo = elbo_loss(x_batch, x_recon, z_mean, z_logvar)
        mse = mse_loss(y_batch, y_pred)
        recon = reconstruction_count(x_batch, x_recon)

        metrics["test_elbo"] += elbo.item()
        metrics["test_mse"] += mse.item()
        metrics["test_accuracy"] += recon.item()

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


def train_vae(
    model: MolecularVAE,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    n_epochs: int,
    include_mse: bool,
    print_every: int = 10,
) -> MetricTracker:
    tracker = MetricTracker()
    for epoch in range(1, n_epochs + 1):
        train_metrics = train_one_epoch(model, optimizer, scheduler, train_loader, include_mse)
        test_metrics = test_one_epoch(model, test_loader)

        metrics = {**train_metrics, **test_metrics}
        tracker.record(epoch, metrics)

        if epoch == 1 or epoch % print_every == 0:
            tracker.log(
                [
                    "train_elbo",
                    "test_elbo",
                    "train_mse",
                    "test_mse",
                    "train_accuracy",
                    "test_accuracy",
                ]
            )

    return tracker
