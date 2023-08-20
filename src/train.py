import os
import dataclasses
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR
import mlflow

from src.loss import VAELoss
from src.vae import MolecularVAE

EXPERIMENT_NAME = "MolecularVAE"


@dataclass
class TrainingConfig:
    n_epochs: int
    latent_size: int
    target_size: int
    encoder_hidden_size: int = 400
    gru_hidden_size: int = 500
    mlp_hidden_size: int = 300
    gru_layers: int = 3
    gru_dropout: float = 0.1
    mse_scale: float = 1.0
    beta_max: float = 1.0
    beta_epoch_start: int = 0
    beta_epoch_end: int = 0
    lr_init: float = 2e-3
    lr_gamma: float = 0.5
    lr_milestones: list[int] = field(default_factory=lambda: [])
    weight_decay: float = 1e-5
    batch_size: int = 250


def load_training_data() -> (Dataset, Dataset):
    x_train = torch.load(os.path.join("data", "qm9_inputs_train.pt")).float()
    x_test = torch.load(os.path.join("data", "qm9_inputs_test.pt")).float()

    y_train = torch.load(os.path.join("data", "qm9_descriptors_train.pt")).float()
    y_test = torch.load(os.path.join("data", "qm9_descriptors_test.pt")).float()

    return x_train, x_test, y_train, y_test


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
        x_recon, y_hat, z_mean, z_logvar = model(x)

        loss = criterion(x, x_recon, y, y_hat, z_mean, z_logvar, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item()
        metrics["ce"] += criterion.current_ce
        metrics["kld"] += criterion.current_kld
        metrics["mse"] += criterion.current_mse
        metrics["accuracy"] += criterion.current_accuracy

    if scheduler is not None:
        scheduler.step()

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def test_one_epoch(model: MolecularVAE, criterion: VAELoss, data_loader: DataLoader, epoch: int):
    model.eval()
    metrics = defaultdict(lambda: 0.0)
    for x, y in data_loader:
        x_recon, y_hat, z_mean, z_logvar = model(x)
        loss = criterion(x, x_recon, y, y_hat, z_mean, z_logvar, epoch)

        metrics["loss"] += loss.item()
        metrics["ce"] += criterion.current_ce
        metrics["kld"] += criterion.current_kld
        metrics["mse"] += criterion.current_mse
        metrics["accuracy"] += criterion.current_accuracy

    n = len(data_loader.dataset)
    return {k: v / n for k, v in metrics.items()}


def train_model(config: TrainingConfig, *, run_name: Optional[str] = None):
    # Specify tracking server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        for field in dataclasses.fields(config):
            mlflow.log_param(field.name, getattr(config, field.name))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device {device}")

        x_train, x_test, y_train, y_test = (x.to(device) for x in load_training_data())
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

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
        logger.info(f"Initialized model with {model.n_parameters()} parameters")

        optimizer = Adam(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
        scheduler = MultiStepLR(optimizer, gamma=config.lr_gamma, milestones=config.lr_milestones)
        criterion = VAELoss(
            mse_scale=config.mse_scale,
            beta_max=config.beta_max,
            beta_epoch_start=config.beta_epoch_start,
            beta_epoch_end=config.beta_epoch_end,
        )

        for epoch in range(1, config.n_epochs + 1):
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

            mlflow.log_metric("kld_beta", criterion.get_beta(epoch), step=epoch)

            logger.info(
                f"Epoch {epoch} | "
                f"Train Loss {train_metrics['loss']:.3f} | "
                f"Test Loss {test_metrics['loss']:.3f} | "
                f"LR {scheduler.get_last_lr()[0]:.3e}"
            )

        # Log final model
        model_info = mlflow.pytorch.log_model(model, "model")
        logger.info(f"Logged trained model at {model_info.model_uri}")
