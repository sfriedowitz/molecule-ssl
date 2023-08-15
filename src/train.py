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
    return (recon_labels == actual_labels).all(dim=-1).sum().item()


def vae_loss(
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


def train_one_epoch(
    model: MolecularVAE,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    data_loader: DataLoader,
):
    model.train()

    total_loss = 0.0
    total_recon = 0
    for batch in data_loader:
        x = batch[0]
        optimizer.zero_grad()
        x_recon, z_mean, z_logvar = model(x)
        loss = vae_loss(x, x_recon, z_mean, z_logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += reconstruction_count(x, x_recon)

    if scheduler is not None:
        scheduler.step()

    n = len(data_loader.dataset)
    return total_loss / n, total_recon / n


@torch.no_grad()
def test_one_epoch(model: MolecularVAE, data_loader: DataLoader):
    model.eval()

    total_loss = 0.0
    total_recon = 0
    for batch in data_loader:
        x = batch[0]
        x_recon, z_mean, z_logvar = model(x)
        loss = vae_loss(x, x_recon, z_mean, z_logvar)
        total_loss += loss.item()
        total_recon += reconstruction_count(x, x_recon)

    n = len(data_loader.dataset)
    return total_loss / n, total_recon / n


def train_vae(
    model: MolecularVAE,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
    print_every: int = 10,
) -> MetricTracker:
    tracker = MetricTracker()
    for epoch in range(1, n_epochs + 1):
        train_loss_epoch, train_accuracy_epoch = train_one_epoch(
            model, optimizer, scheduler, train_loader
        )
        test_loss_epoch, test_accuracy_epoch = test_one_epoch(model, test_loader)

        tracker.record_metric("train_loss", epoch, train_loss_epoch)
        tracker.record_metric("train_accuracy", epoch, train_accuracy_epoch)
        tracker.record_metric("test_loss", epoch, test_loss_epoch)
        tracker.record_metric("test_accuracy", epoch, test_accuracy_epoch)

        if epoch == 1 or epoch % print_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch} | "
                f"Train Loss = {train_loss_epoch:.4f} | "
                f"Test loss = {test_loss_epoch:.4f} | "
                f"Train Accuracy = {train_accuracy_epoch:.4f} | "
                f"Test Accuracy = {test_accuracy_epoch:.4f} | "
                f"LR = {current_lr:.3e}"
            )
    return tracker
