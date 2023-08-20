from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import mlflow

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

    metrics = {"ce": 0.0, "kld": 0.0, "mse": 0.0, "accuracy": 0.0}
    for x, y in data_loader:
        x_recon, y_hat, z_mean, z_logvar = model(x)

        loss = criterion(x, x_recon, y, y_hat, z_mean, z_logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics["ce"] += criterion.current_ce
        metrics["kld"] += criterion.current_kld
        metrics["mse"] += criterion.current_mse
        metrics["accuracy"] += criterion.current_recon

    if scheduler is not None:
        scheduler.step()

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def test_one_epoch(model: MolecularVAE, criterion: VAELoss, data_loader: DataLoader):
    model.eval()

    metrics = {"ce": 0.0, "kld": 0.0, "mse": 0.0, "accuracy": 0.0}
    for x, y in data_loader:
        x_recon, y_hat, z_mean, z_logvar = model(x)
        _ = criterion(x, x_recon, y, y_hat, z_mean, z_logvar)

        metrics["ce"] += criterion.current_ce
        metrics["kld"] += criterion.current_kld
        metrics["mse"] += criterion.current_mse
        metrics["accuracy"] += criterion.current_recon

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
) -> MetricTracker:
    with mlflow.start_run():
        # Parameter logging

        for epoch in tqdm(range(n_epochs)):
            train_metrics = train_one_epoch(model, criterion, optimizer, scheduler, train_loader)
            for metric, value in train_metrics.items():
                key = f"train_{metric}"
                mlflow.log_metric(key, value, step=epoch)

            test_metrics = test_one_epoch(model, criterion, test_loader)
            for metric, value in test_metrics.items():
                key = f"test_{metric}"
                mlflow.log_metric(key, value, step=epoch)
