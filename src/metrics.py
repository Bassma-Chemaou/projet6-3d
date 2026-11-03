"""Métriques de génération 3D : MMD-CD, COV, 1-NNA (étape A6)."""
import torch

from src.losses import chamfer_batch


@torch.no_grad()
def pairwise_cd(A, B, device="cuda", chunk=50):
    """Matrice (len(A), len(B)) des distances de Chamfer."""
    A, B = A.to(device), B.to(device)
    out = torch.zeros(len(A), len(B))
    for i in range(len(A)):
        a = A[i:i + 1]
        for j in range(0, len(B), chunk):
            b = B[j:j + chunk]
            out[i, j:j + chunk] = chamfer_batch(a.expand(len(b), -1, -1).contiguous(), b).cpu()
    return out


def gen_metrics(G, R, device="cuda"):
    """G : formes générées, R : formes réelles de référence (mêmes effectifs)."""
    M_rg = pairwise_cd(R, G, device)
    M_rr = pairwise_cd(R, R, device)
    M_gg = pairwise_cd(G, G, device)
    mmd = M_rg.min(1)[0].mean().item()                        # qualité (plus bas = mieux)
    cov = len(torch.unique(M_rg.argmin(0))) / len(R)          # diversité (plus haut = mieux)
    big = torch.cat([torch.cat([M_rr, M_rg], 1), torch.cat([M_rg.t(), M_gg], 1)], 0)
    big.fill_diagonal_(float("inf"))
    lab = torch.cat([torch.ones(len(R)), torch.zeros(len(G))])
    nna = (lab[big.argmin(1)] == lab).float().mean().item()   # idéal : proche de 0.5
    return {"MMD-CD": mmd, "COV": cov, "1-NNA": nna}
