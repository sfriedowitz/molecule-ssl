from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self, *, kld_scale: float = 1.0, mse_scale: float = 1.0):
        super().__init__()
        self.kld_scale = kld_scale
        self.mse_scale = mse_scale

        # Cache values from forward pass for tracking
        self.current_ce: Optional[float] = None
        self.current_kld: Optional[float] = None
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
    ) -> torch.Tensor:
        # Cross entropy should be computed across one-hot labels,
        # so transpose tensors so labels in dim=1
        ce = F.cross_entropy(xr.transpose(2, 1), x.transpose(2, 1), reduction="sum")
        kld = -0.5 * torch.sum(1.0 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        mse = F.mse_loss(y_hat, y, reduction="sum")
        recon = (x.argmax(dim=-1) == xr.argmax(dim=-1)).all(dim=-1).sum()

        self.current_ce = ce.item()
        self.current_kld = kld.item()
        self.current_mse = mse.item()
        self.current_recon = recon.item()

        return ce + self.kld_scale * kld + self.mse_scale * mse
