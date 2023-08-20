import dataclasses
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler, ExponentialLR
import mlflow

from src.loss import VAELoss
from src.vae import MolecularVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    latent_size: int
    target_size: int
    encoder_hidden_size: int = 400
    gru_hidden_size: int = 500
    mlp_hidden_size: int = 300
    gru_layers: int = 3
    gru_dropout: float = 0.2
    include_mse: bool = 1.0
    beta_max: float = 1.0
    beta_start: int = 0
    beta_end: int = 0
    lr_init: float = 1e-3
    lr_decay: float = 1.0
    weight_decay: float = 1e-5
    batch_size: int = 250


def train_one_epoch(
    model: MolecularVAE,
    criterion: VAELoss,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    data_loader: DataLoader,
    epoch: int,
):
    model.train()
    metrics = defaultdict(lambda: 0.0)
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        x_recon, y_hat, z_mean, z_logvar = model(x)

        loss = criterion(x, x_recon, y, y_hat, z_mean, z_logvar, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item()
        metrics["ce"] += criterion.current_ce
        metrics["kld"] += criterion.current_kld
        metrics["accuracy"] += criterion.current_accuracy

        if criterion.current_mse is not None:
            metrics["mse"] += criterion.current_mse

    if scheduler is not None:
        scheduler.step()

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def test_one_epoch(model: MolecularVAE, criterion: VAELoss, data_loader: DataLoader, epoch: int):
    model.eval()
    metrics = defaultdict(lambda: 0.0)
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        x_recon, y_hat, z_mean, z_logvar = model(x)
        loss = criterion(x, x_recon, y, y_hat, z_mean, z_logvar, epoch)

        metrics["loss"] += loss.item()
        metrics["ce"] += criterion.current_ce
        metrics["kld"] += criterion.current_kld
        metrics["accuracy"] += criterion.current_accuracy

        if criterion.current_mse is not None:
            metrics["mse"] += criterion.current_mse

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


def train_model(
    config: TrainingConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    n_epochs: int,
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = "MolecularVAE",
) -> MolecularVAE:
    # Specify tracking server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=run_name):
        for field in dataclasses.fields(config):
            mlflow.log_param(field.name, getattr(config, field.name))

        logger.info(f"Using device {device}")

        # Configure model and parameters
        model = MolecularVAE(
            latent_size=config.latent_size,
            target_size=config.target_size,
            encoder_hidden_size=config.encoder_hidden_size,
            gru_hidden_size=config.gru_hidden_size,
            mlp_hidden_size=config.mlp_hidden_size,
            gru_layers=config.gru_layers,
            gru_dropout=config.gru_dropout,
        )
        model = model.to(device)
        logger.info(f"Initializing model with {model.n_parameters()} parameters")

        optimizer = Adam(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=config.lr_decay)
        criterion = VAELoss(
            include_mse=config.include_mse,
            beta_max=config.beta_max,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        best_test_loss = float("inf")
        for epoch in range(n_epochs):
            train_metrics = train_one_epoch(
                model, criterion, optimizer, scheduler, train_loader, epoch
            )
            for metric, value in train_metrics.items():
                key = f"train_{metric}"
                mlflow.log_metric(key, value, step=epoch)

            test_metrics = test_one_epoch(model, criterion, test_loader, epoch)
            for metric, value in test_metrics.items():
                key = f"test_{metric}"
                mlflow.log_metric(key, value, step=epoch)

            print(epoch, " ", criterion.get_beta(epoch))
            mlflow.log_metric("kld_beta", criterion.get_beta(epoch), step=epoch)

            if test_metrics["loss"] < best_test_loss:
                best_test_loss = test_metrics["loss"]
                logger.info(f"Achieved best test loss of {best_test_loss:.3f} -- logging model")
                mlflow.pytorch.log_model(model, "model")

            logger.info(
                f"Epoch {epoch} | "
                f"Train Accuracy {train_metrics['accuracy']:.3f} | "
                f"Test Accuracy {test_metrics['accuracy']:.3f} | "
                f"LR {scheduler.get_last_lr()[0]:.3e}"
            )

        # Log final model
        mlflow.pytorch.log_model(model, "model")

    return model.eval()
