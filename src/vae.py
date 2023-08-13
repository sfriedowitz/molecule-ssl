import torch
from torch import nn


# Determined from selfies encoding of QM9 dataset
# INPUT_SIZE is the longest length of selfies characters
# INPUT_CHANNELS is the alphabet size of selfies characters
INPUT_SIZE = 21
INPUT_CHANNELS = 29


class MolecularVAE(nn.Module):
    def __init__(
        self,
        *,
        latent_size: int = 50,
        encoder_hidden_size: int = 150,
        gru_hidden_size: int = 200,
    ):
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
            nn.Linear(80, encoder_hidden_size),
            nn.SELU(),
        )
        self.encoder_mean = nn.Linear(encoder_hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(encoder_hidden_size, latent_size)

        # Decoder layers
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.SELU(),
        )
        self.decoder_gru = nn.GRU(latent_size, gru_hidden_size, 3, batch_first=True)
        self.decoder_output = nn.Linear(gru_hidden_size, INPUT_CHANNELS)

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

        y = self.decoder_output(gru_flat)
        y = y.contiguous().view(gru.size(0), -1, y.size(-1))

        return y

    def sample(self, z_mean, z_logvar):
        epsilon = torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sample(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar
