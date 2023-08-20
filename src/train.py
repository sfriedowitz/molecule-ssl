import dataclasses
from dataclasses import dataclass
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
    gru_dropout: float = 0.1
    mse_scale: float = 1.0
    kld_scale: float = 1e-3
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
):
    model.train()

    metrics = {"ce": 0.0, "kld": 0.0, "mse": 0.0, "accuracy": 0.0}
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
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
        x, y = x.to(device), y.to(device)
        x_recon, y_hat, z_mean, z_logvar = model(x)
        _ = criterion(x, x_recon, y, y_hat, z_mean, z_logvar)

        metrics["ce"] += criterion.current_ce
        metrics["kld"] += criterion.current_kld
        metrics["mse"] += criterion.current_mse
        metrics["accuracy"] += criterion.current_recon

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


def run_training(
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

        criterion = VAELoss(kld_scale=config.kld_scale, mse_scale=config.mse_scale)
        optimizer = Adam(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=config.lr_decay)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        for epoch in range(n_epochs):
            train_metrics = train_one_epoch(model, criterion, optimizer, scheduler, train_loader)
            for metric, value in train_metrics.items():
                key = f"train_{metric}"
                mlflow.log_metric(key, value, step=epoch)

            test_metrics = test_one_epoch(model, criterion, test_loader)
            for metric, value in test_metrics.items():
                key = f"test_{metric}"
                mlflow.log_metric(key, value, step=epoch)

            logger.info(
                f"Epoch {epoch} | "
                f"Train Accuracy {train_metrics['accuracy']:.4f} | "
                f"Test Accuracy {test_metrics['accuracy']:.4f} | "
            )

        # Log final model
        mlflow.pytorch.log_model(model, "model")

    return model.eval()
