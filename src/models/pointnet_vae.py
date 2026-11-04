"""VAE de nuages de points PointNet (étape A4). n_classes > 0 : version conditionnelle (CVAE)."""
import torch
import torch.nn as nn


class PointNetVAE(nn.Module):
    def __init__(self, n_points=2048, latent=128, n_classes=0):
        super().__init__()
        ch, layers = [3, 64, 128, 256, 512], []
        for i in range(4):
            layers += [nn.Conv1d(ch[i], ch[i + 1], 1), nn.BatchNorm1d(ch[i + 1]), nn.ReLU()]
        self.enc = nn.Sequential(*layers)
        self.fc_mu, self.fc_logvar = nn.Linear(512, latent), nn.Linear(512, latent)
        self.dec = nn.Sequential(nn.Linear(latent + n_classes, 512), nn.ReLU(),
                                 nn.Linear(512, 1024), nn.ReLU(),
                                 nn.Linear(1024, n_points * 3))
        self.n, self.n_classes, self.latent = n_points, n_classes, latent

    def encode(self, x):
        h = self.enc(x.transpose(1, 2)).max(2)[0]
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, y=None):
        if self.n_classes:
            z = torch.cat([z, nn.functional.one_hot(y, self.n_classes).float()], 1)
        return self.dec(z).view(-1, self.n, 3)

    def forward(self, x, y=None):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, y), mu, logvar
