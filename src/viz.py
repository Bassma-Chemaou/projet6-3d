"""Affichage de nuages de points et de voxels (axe Z vertical, comme ModelNet)."""
import matplotlib.pyplot as plt
import numpy as np


def plot_points(ax, pts, size=1):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def plot_voxels(ax, vox):
    ax.voxels(vox.astype(bool))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def control_figure(points, voxels, labels, categories, per_cat=3, path=None):
    """per_cat formes par catégorie : points (ligne du haut) et voxels (ligne du bas)."""
    nc = len(categories)
    fig = plt.figure(figsize=(2.4 * per_cat, 2.4 * 2 * nc))
    for ci, cat in enumerate(categories):
        idx = np.where(labels == ci)[0][:per_cat]
        for j, i in enumerate(idx):
            ax = fig.add_subplot(2 * nc, per_cat, 2 * ci * per_cat + j + 1, projection="3d")
            plot_points(ax, points[i])
            if j == 0:
                ax.set_title(f"{cat} - points", fontsize=8, loc="left")
            ax = fig.add_subplot(2 * nc, per_cat, (2 * ci + 1) * per_cat + j + 1, projection="3d")
            plot_voxels(ax, voxels[i])
            if j == 0:
                ax.set_title(f"{cat} - voxels", fontsize=8, loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
