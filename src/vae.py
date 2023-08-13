import torch
from torch import nn


# Determined from selfies encoding of QM9 dataset
# INPUT_SIZE is the longest length of selfies characters
# INPUT_CHANNELS is the alphabet size of selfies characters
INPUT_SIZE = 21
INPUT_CHANNELS = 29

# After some trial and error...
LATENT_SIZE = 50
ENCODER_HIDDEN_SIZE = 400
GRU_HIDDEN_SIZE = 500


class MolecularVAE(nn.Module):
    def __init__(self):
        super().__init__()

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
        self.encoder_mean = nn.Linear(ENCODER_HIDDEN_SIZE, LATENT_SIZE)
        self.encoder_logvar = nn.Linear(ENCODER_HIDDEN_SIZE, LATENT_SIZE)

        # Decoder layers
        self.decoder_linear = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            nn.SELU(),
        )
        self.decoder_gru = nn.GRU(LATENT_SIZE, GRU_HIDDEN_SIZE, 3, batch_first=True)
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
            epsilon = 1e-2 * torch.randn_like(z_logvar)
            return z_mean + torch.exp(0.5 * z_logvar) * epsilon
        else:
            return z_mean

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar
