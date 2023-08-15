from typing import Optional
import torch
from torch import nn

from src.utils import slerp


# Determined from selfies encoding of QM9 dataset
# INPUT_SIZE is the longest length of selfies characters
# INPUT_CHANNELS is the alphabet size of selfies characters
INPUT_SIZE = 21
INPUT_CHANNELS = 29

# After some trial and error...
ENCODER_HIDDEN_SIZE = 400
GRU_HIDDEN_SIZE = 500


class MolecularVAE(nn.Module):
    def __init__(self, latent_size: int, *, sample_eps: float = 1e-2):
        super().__init__()

        self.latent_size = latent_size
        self.sample_eps = sample_eps

        # Encoder layers
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 9, kernel_size=9, padding=2),
            nn.ReLU(),
            nn.Conv1d(9, 9, kernel_size=9, padding=2),
            nn.ReLU(),
            nn.Conv1d(9, 10, kernel_size=10, padding=2),
            nn.ReLU(),
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(80, ENCODER_HIDDEN_SIZE),
            nn.SELU(),
        )
        self.encoder_mean = nn.Linear(ENCODER_HIDDEN_SIZE, latent_size)
        self.encoder_logvar = nn.Linear(ENCODER_HIDDEN_SIZE, latent_size)

        # Decoder layers
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.SELU(),
        )
        self.decoder_gru = nn.GRU(latent_size, GRU_HIDDEN_SIZE, 3, batch_first=True)
        self.decoder_output = nn.Linear(GRU_HIDDEN_SIZE, INPUT_CHANNELS)

    def n_parameters(self) -> int:
        return sum(torch.numel(x) for x in self.parameters())

    def encode(self, x):
        # Input shape is (batch_size x input_len x input_channels)
        # Transpose to put input_channels at dimension 1
        x = x.transpose(2, 1)
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_linear(x)
        return self.encoder_mean(x), self.encoder_logvar(x)

    def decode(self, z):
        z = self.decoder_linear(z)
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, INPUT_SIZE, 1)

        gru, _ = self.decoder_gru(z)
        gru_flat = gru.contiguous().view(-1, gru.size(-1))

        xr = self.decoder_output(gru_flat)
        xr = xr.contiguous().view(gru.size(0), -1, xr.size(-1))

        return xr

    def reparameterize(self, z_mean, z_logvar):
        if self.training:
            epsilon = self.sample_eps * torch.randn_like(z_logvar)
            return z_mean + torch.exp(0.5 * z_logvar) * epsilon
        else:
            return z_mean

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar

    def sample(self, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        z = self.sample_eps * torch.randn((1, self.latent_size), generator=generator)
        return self.decode(z)[0]

    def interpolate(self, x_start: torch.Tensor, x_end: torch.Tensor, n: int) -> torch.Tensor:
        z_start = self.encode(x_start.unsqueeze(0))[0]
        z_end = self.encode(x_end.unsqueeze(0))[0]
        z_interp = [
            slerp(z_start.squeeze(), z_end.squeeze(), t) for t in torch.linspace(0.0, 1.0, n)
        ]
        z_interp = torch.stack(z_interp)
        return self.decode(z_interp)
