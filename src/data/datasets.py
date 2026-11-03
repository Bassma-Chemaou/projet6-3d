"""Chargement des données prétraitées et augmentation."""
from pathlib import Path

import numpy as np
import torch


def load_processed(root, splits=("train", "val", "test")):
    """Renvoie (X, Y, categories) : X[split] tenseur (n, points, 3), Y[split] labels numpy."""
    root = Path(root)
    cats = (root / "categories.txt").read_text().split("\n")
    X = {s: torch.from_numpy(np.load(root / f"points_{s}.npy")) for s in splits}
    Y = {s: np.load(root / f"labels_{s}.npy") for s in splits}
    return X, Y, cats


def augment(x, rot_deg=15, noise=0.01):
    """Petite rotation autour de l'axe vertical Z + bruit gaussien. x : (B, N, 3)."""
    B = x.shape[0]
    th = (torch.rand(B, device=x.device) * 2 - 1) * rot_deg * np.pi / 180
    c, s = torch.cos(th), torch.sin(th)
    Rz = torch.zeros(B, 3, 3, device=x.device)
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1
    return x @ Rz.transpose(1, 2) + noise * torch.randn_like(x)
