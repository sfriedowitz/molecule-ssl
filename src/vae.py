import torch
from torch import nn
from torch.nn import functional as F


class MolecularVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 292):
        super().__init__()

        # The input filter dim should correspond to the size of SMILES alphabet
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_dim, 9, kernel_size=9),
            nn.ReLU(),
            nn.Conv1d(9, 9, kernel_size=9),
            nn.ReLU(),
            nn.Conv1d(9, 10, kernel_size=11),
            nn.ReLU(),
        )
        self.encoder_embedding = nn.Sequential(
            nn.Linear(940, 435),
            nn.SELU(),
        )
        self.encoder_mean = nn.Linear(435, latent_dim)
        self.encoder_logvar = nn.Linear(435, latent_dim)

        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.gru = nn.GRU(latent_dim, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, 35)

    def encode(self, x):
        x = self.encoding_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_embedding(x)
        return self.encoder_mean(x), self.encoder_conv(x)

    def decode(self, z):
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        out, _ = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc3(out_reshape), dim=1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = 1e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)
            return w
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
