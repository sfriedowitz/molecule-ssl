from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.loss import VAELoss
from src.tracker import MetricTracker
from src.vae import MolecularVAE


def train_one_epoch(
    model: MolecularVAE,
    criterion: VAELoss,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    data_loader: DataLoader,
):
    model.train()

    metrics = {"train_elbo": 0.0, "train_mse": 0.0, "train_accuracy": 0.0}
    for x, y in data_loader:
        x_recon, y_hat, z_mean, z_logvar = model(x)

        loss = criterion(x, x_recon, y, y_hat, z_mean, z_logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics["train_elbo"] += criterion.current_ce
        metrics["train_mse"] += criterion.current_mse
        metrics["train_accuracy"] += criterion.current_recon

    if scheduler is not None:
        scheduler.step()

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def test_one_epoch(model: MolecularVAE, criterion: VAELoss, data_loader: DataLoader):
    model.eval()

    metrics = {"test_elbo": 0.0, "test_mse": 0.0, "test_accuracy": 0.0}
    for x, y in data_loader:
        x_recon, y_hat, z_mean, z_logvar = model(x)
        _ = criterion(x, x_recon, y, y_hat, z_mean, z_logvar)

        metrics["test_elbo"] += criterion.current_ce
        metrics["test_mse"] += criterion.current_mse
        metrics["test_accuracy"] += criterion.current_recon

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


def train_vae(
    model: MolecularVAE,
    criterion: VAELoss,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    n_epochs: int,
    print_every: int = 10,
) -> MetricTracker:
    tracker = MetricTracker()
    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(model, criterion, optimizer, scheduler, train_loader)
        test_metrics = test_one_epoch(model, criterion, test_loader)

        epoch_metrics = {**train_metrics, **test_metrics}
        tracker.record(epoch, epoch_metrics)

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
