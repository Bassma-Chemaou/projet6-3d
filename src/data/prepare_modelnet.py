"""Prétraitement ModelNet10 : maillage .off -> nuage de points + voxels (étape A2).

Convention d'axes : ModelNet a l'axe Z vertical. Toute rotation d'augmentation
« autour de la verticale » doit donc se faire autour de Z.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_fill_holes


def read_off(path) -> trimesh.Trimesh:
    """Lecteur .off tolérant (certains fichiers collent 'OFF' aux compteurs)."""
    tokens = open(path).read().split()
    if tokens[0] == "OFF":
        n_v, n_f, pos = int(tokens[1]), int(tokens[2]), 4
    else:
        n_v, n_f, pos = int(tokens[0][3:]), int(tokens[1]), 3
    verts = np.array(tokens[pos:pos + 3 * n_v], dtype=np.float64).reshape(n_v, 3)
    pos += 3 * n_v
    faces = []
    for _ in range(n_f):
        k = int(tokens[pos])
        idx = [int(t) for t in tokens[pos + 1:pos + 1 + k]]
        pos += 1 + k
        for i in range(1, k - 1):
            faces.append([idx[0], idx[i], idx[i + 1]])
    return trimesh.Trimesh(verts, np.array(faces), process=False)


def normalize(mesh):
    """Centre à l'origine et met à l'échelle pour tenir dans [-1, 1]."""
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1.0 / np.abs(mesh.vertices).max())
    return mesh


def to_points(mesh, n_points, seed):
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    return pts.astype(np.float32)


def to_voxels(mesh, res, seed):
    """Grille d'occupation 0/1 pleine, robuste aux maillages non étanches."""
    dense, _ = trimesh.sample.sample_surface(mesh, 100_000, seed=seed)
    idx = np.clip(((dense + 1) / 2 * res).astype(int), 0, res - 1)
    g = np.zeros((res, res, res), dtype=bool)
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    g = binary_fill_holes(binary_closing(g, iterations=1))
    return g.astype(np.uint8)


def list_files(root, categories, split):
    return [(f, ci) for ci, c in enumerate(categories)
            for f in sorted((root / c / split).glob("*.off"))]


def process(files, n_points, res, seed, tag):
    P, V, L, N, skipped = [], [], [], [], 0
    for k, (f, lab) in enumerate(files):
        try:
            m = normalize(read_off(f))
            P.append(to_points(m, n_points, seed + k))
            V.append(to_voxels(m, res, seed + k))
            L.append(lab)
            N.append(f.stem)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            print("  [ignoré]", f.name, ":", e)
        if (k + 1) % 250 == 0:
            print(f"  {tag} : {k + 1}/{len(files)}")
    print(f"{tag} : {len(P)} formes ({skipped} ignorées)")
    return np.stack(P), np.stack(V), np.array(L), np.array(N)


def main(raw, out, categories, n_points=2048, res=32, val_fraction=0.15,
         seed=42, max_train_per_cat=400):
    root, out = Path(raw) / "ModelNet10", Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train_all = list_files(root, categories, "train")
    tr_files, val_files = [], []
    for ci in range(len(categories)):
        cat = [t for t in train_all if t[1] == ci]
        if max_train_per_cat:
            cat = [cat[i] for i in rng.permutation(len(cat))[:max_train_per_cat]]
        perm = rng.permutation(len(cat))
        n_val = int(round(val_fraction * len(cat)))
        val_files += [cat[i] for i in perm[:n_val]]
        tr_files += [cat[i] for i in perm[n_val:]]

    for tag, files in (("train", tr_files), ("val", val_files),
                       ("test", list_files(root, categories, "test"))):
        P, V, L, N = process(files, n_points, res, seed, tag)
        np.save(out / f"points_{tag}.npy", P)
        np.savez_compressed(out / f"voxels_{tag}.npz", voxels=V)
        np.save(out / f"labels_{tag}.npy", L)
        np.save(out / f"names_{tag}.npy", N)
        print(f"  -> points {P.shape}, voxels {V.shape}")
    (out / "categories.txt").write_text("\n".join(categories))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--categories", nargs="+", default=["chair", "table", "sofa", "bed"])
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_per_cat", type=int, default=400)
    a = ap.parse_args()
    main(a.raw, a.out, a.categories, a.n_points, a.res, seed=a.seed,
         max_train_per_cat=a.max_train_per_cat)
