"""Auto-encodeur PointNet (baseline, étape A3)."""
import torch.nn as nn


class PointNetAE(nn.Module):
    def __init__(self, n_points=2048, latent=128):
        super().__init__()
        ch, layers = [3, 64, 128, 256, 512], []
        for i in range(4):
            layers += [nn.Conv1d(ch[i], ch[i + 1], 1), nn.BatchNorm1d(ch[i + 1]), nn.ReLU()]
        self.enc = nn.Sequential(*layers)
        self.fc = nn.Linear(512, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 512), nn.ReLU(),
                                 nn.Linear(512, 1024), nn.ReLU(),
                                 nn.Linear(1024, n_points * 3))
        self.n = n_points

    def forward(self, x):                                   # x : (B, N, 3)
        z = self.fc(self.enc(x.transpose(1, 2)).max(2)[0])  # max-pooling sur les points
        return self.dec(z).view(-1, self.n, 3), z
