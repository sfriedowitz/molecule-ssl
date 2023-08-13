import torch
from torch import nn
from torch.nn import functional as F


class MolecularVAE(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        conv_kernels: list[int] = [9, 9, 5],
        latent_size: int = 292,
        encoder_hidden_size: int = 435,
        gru_layers: int = 3,
        gru_hidden_size: int = 501,
    ):
        super().__init__()

        self.input_size = input_size

        # Per conv layer: output_size = (input_size - kernel_size) / stride + 1
        conv_output_len = input_size - sum(conv_kernels) + 3
        if conv_output_len <= 0:
            raise ValueError("Convolution kernel sizes are greater than input sequence length.")
        conv_output_size = 9 * conv_output_len

        # Encoder layers
        conv_layers = []
        conv_layers.append(nn.Conv1d(input_channels, 9, kernel_size=conv_kernels[0]))
        conv_layers.append(nn.ReLU())
        for kernel in conv_kernels[1:]:
            conv_layers.append(nn.Conv1d(9, 9, kernel_size=kernel))
            conv_layers.append(nn.ReLU())

        self.encoder_conv = nn.Sequential(*conv_layers)
        self.encoder_linear = nn.Sequential(
            nn.Linear(conv_output_size, encoder_hidden_size),
            nn.SELU(),
        )
        self.encoder_mean = nn.Linear(encoder_hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(encoder_hidden_size, latent_size)

        # Decoder layers
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.SELU(),
        )
        self.decoder_gru = nn.GRU(latent_size, gru_hidden_size, gru_layers, batch_first=True)
        self.decoder_output = nn.Sequential(
            nn.Linear(gru_hidden_size, input_channels),
            nn.Softmax(dim=1),
        )

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
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.input_size, 1)

        gru, _ = self.decoder_gru(z)
        gru_flat = gru.contiguous().view(-1, gru.size(-1))

        y = self.decoder_output(gru_flat)
        y = y.contiguous().view(gru.size(0), -1, y.size(-1))

        return y

    def sample(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sample(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar
