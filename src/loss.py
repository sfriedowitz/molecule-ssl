from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BetaAnnealing(ABC):
    @abstractmethod
    def get_beta(self, epoch: int) -> float:
        pass


class ConstantAnnealing(BetaAnnealing):
    def __init__(self, beta: float):
        self.beta = beta

    def get_beta(self, _: int) -> float:
        return self.beta


class SigmoidAnnealing(BetaAnnealing):
    def __init__(self, *, start_epoch: int, rate: float = 1.0, scale: float = 1.0):
        self.start_epoch = start_epoch
        self.rate = rate
        self.scale = scale

    def get_beta(self, epoch: int) -> float:
        delta = epoch - self.start_epoch
        sigmoid = 1.0 / (1.0 + np.exp(-self.rate * delta))
        sigmoid = np.maximum(0.0, 2.0 * sigmoid - 1.0)
        return self.scale * sigmoid


class CyclicAnnealing(BetaAnnealing):
    def __init__(
        self,
        *,
        period: int,
        scale: float = 1.0,
        ratio: float = 0.75,
    ):
        self.period = period
        self.scale = scale
        self.ratio = ratio

    def get_beta(self, epoch: int) -> float:
        step = 1.0 / (self.period * self.ratio)
        idx = epoch % self.period
        loc = idx * step
        return self.scale / (1.0 + np.exp(-(loc * 10.0 - 5.0)))


class VAELoss(nn.Module):
    def __init__(self, *, include_mse: bool, beta_annealing: BetaAnnealing):
        super().__init__()
        self.include_mse = include_mse
        self.beta_annealing = beta_annealing

        # Cache values from forward pass for tracking
        self.current_elbo: Optional[float] = None
        self.current_mse: Optional[float] = None
        self.current_recon: Optional[int] = None

    def forward(
        self,
        x: torch.Tensor,
        xr: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        # Cross entropy should be computed across one-hot labels,
        # so transpose tensors so labels in dim=1
        ce = F.cross_entropy(xr.transpose(2, 1), x.transpose(2, 1), reduction="sum")
        kld = -0.5 * torch.sum(1.0 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        beta = self.beta_annealing.get_beta(epoch)
        elbo = ce + beta * kld

        loss = elbo
        if self.include_mse:
            mse = F.mse_loss(y_hat, y, reduction="sum")
            loss += mse
        else:
            mse = torch.tensor(0.0)

        recon = (x.argmax(dim=-1) == xr.argmax(dim=-1)).all(dim=-1).sum()

        self.current_elbo = elbo.item()
        self.current_mse = mse.item()
        self.current_recon = recon.item()

        return loss
