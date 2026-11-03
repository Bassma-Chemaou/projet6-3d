"""Pertes : distance de Chamfer et divergence KL."""
import torch


def chamfer_batch(a, b):
    """a, b : (B, N, 3) -> distance de Chamfer au carré, une valeur par paire (B,)."""
    d = torch.cdist(a, b)
    return d.min(2)[0].pow(2).mean(1) + d.min(1)[0].pow(2).mean(1)


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
