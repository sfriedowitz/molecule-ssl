from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.vae import MolecularVAE
from src.tracker import MetricTracker


class VAELoss(nn.Module):
    def __init__(self, *, include_mse: bool, kld_scale: float = 1e-5):
        super().__init__()
        self.include_mse = include_mse
        self.kld_scale = kld_scale

        # Cache values from forward pass for tracking
        self.elbo: Optional[float] = None
        self.mse: Optional[float] = None
        self.recon: Optional[int] = None

    def forward(
        self,
        x: torch.Tensor,
        xr: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> torch.Tensor:
        # Cross entropy should be computed across one-hot labels,
        # so transpose tensors so labels in dim=1
        cross_entropy = F.cross_entropy(xr.transpose(2, 1), x.transpose(2, 1), reduction="sum")
        kld = -0.5 * torch.sum(1.0 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        elbo = cross_entropy + self.kld_scale * kld
        mse = F.mse_loss(y_hat, y, reduction="sum")

        loss = elbo
        if self.include_mse:
            loss += mse

        recon = (x.argmax(dim=-1) == xr.argmax(dim=-1)).all(dim=-1).sum()

        self.elbo = elbo.item()
        self.mse = mse.item()
        self.recon = recon.item()

        return loss


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

        metrics["train_elbo"] += criterion.elbo
        metrics["train_mse"] += criterion.mse
        metrics["train_accuracy"] += criterion.recon

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

        metrics["test_elbo"] += criterion.elbo
        metrics["test_mse"] += criterion.mse
        metrics["test_accuracy"] += criterion.recon

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
    for epoch in range(1, n_epochs + 1):
        train_metrics = train_one_epoch(model, criterion, optimizer, scheduler, train_loader)
        test_metrics = test_one_epoch(model, criterion, test_loader)

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
