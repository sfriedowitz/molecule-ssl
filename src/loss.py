from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(
        self,
        *,
        include_mse: bool = True,
        beta_max: float = 1.0,
        beta_start: int = 0,
        beta_end: int = 0,
    ):
        super().__init__()
        self.include_mse = include_mse

        # Tracking KL loss contribution
        self.beta_max = beta_max
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Cache values from forward pass for tracking
        self.current_ce: Optional[float] = None
        self.current_kld: Optional[float] = None
        self.current_mse: Optional[float] = None
        self.current_accuracy: Optional[int] = None

    def get_beta(self, epoch):
        if epoch < self.beta_start:
            return 0.0
        elif epoch >= self.beta_end:
            return self.beta_max
        else:
            step_size = self.beta_max / max(self.beta_end - self.beta_start, 1)
            return step_size * (epoch - self.beta_start)

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
        accuracy = (x.argmax(dim=-1) == xr.argmax(dim=-1)).all(dim=-1).sum()

        loss = ce + self.get_beta(epoch) * kld
        if self.include_mse:
            mse = F.mse_loss(y_hat, y, reduction="sum")
            loss += mse
            self.current_mse = mse.item()

        self.current_ce = ce.item()
        self.current_kld = kld.item()
        self.current_accuracy = accuracy.item()

        return loss
