"""
scflame.viz
===========
t-SNE plotting helpers: a simple true-vs-predicted panel (used after a
single scFLAME fit), and a multi-panel "merge path" plot showing how
clusters change across the greedy merge steps in scflame.merging.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score


def plot_tsne(latents, true_labels, pred_labels, K_true, save_path, seed=42):
    """Side-by-side t-SNE plot of the latent space, coloured by true vs predicted cluster."""
    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, max(5, len(latents) // 4)))
    latents_2d = tsne.fit_transform(latents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = ax1.scatter(latents_2d[:, 0], latents_2d[:, 1], c=true_labels, cmap="tab10",
                            alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
    ax1.set_title("True Cluster Labels", fontsize=12, fontweight="bold")
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=11)
    plt.colorbar(scatter1, ax=ax1, label="True Cluster")

    scatter2 = ax2.scatter(latents_2d[:, 0], latents_2d[:, 1], c=pred_labels, cmap="tab10",
                            alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
    ax2.set_title("scFLAME Predicted Labels", fontsize=12, fontweight="bold")
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=11)
    plt.colorbar(scatter2, ax=ax2, label="Predicted Cluster")

    ari = adjusted_rand_score(true_labels, pred_labels)
    fig.suptitle(f"Dataset (K={K_true}) | ARI = {ari:.3f}", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _clean_tsne_ax(ax):
    ax.set_xlabel("tSNE 1", fontsize=7)
    ax.set_ylabel("tSNE 2", fontsize=7)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def plot_tsne_merge_path(m_np, allocations, K_true, K_init,
                          c_true_np: Optional[np.ndarray] = None,
                          out_path="tsne_merge_path.png", tsne_kwargs=None):
    """One t-SNE panel per merge step (K descending), plus a true-labels panel if available."""
    _tsne_kwargs = dict(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    if tsne_kwargs is not None:
        _tsne_kwargs.update(tsne_kwargs)

    print("  Fitting t-SNE on latent means...")
    embedding = TSNE(**_tsne_kwargs).fit_transform(m_np)
    print("  t-SNE done.")

    # Sort K values descending so panels read left-to-right = merge order
    K_steps = sorted(allocations.keys(), reverse=True)
    n_panels = len(K_steps) + (1 if c_true_np is not None else 0)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = plt.figure(figsize=(5.5 * n_cols, 4.5 * n_rows))
    fig.suptitle(f"t-SNE of latent means -- merge path  |  K_init={K_init}  K_true={K_true}",
                 fontsize=12, y=1.01)

    # Combine two colormaps for enough distinct colours across K_init clusters
    cmap20 = plt.cm.get_cmap("tab20", 20)
    cmap20b = plt.cm.get_cmap("tab20b", 20)
    palette = [cmap20(i) for i in range(20)] + [cmap20b(i) for i in range(20)]

    panel_idx = 1

    if c_true_np is not None:
        ax = fig.add_subplot(n_rows, n_cols, panel_idx)
        for ci, label in enumerate(np.unique(c_true_np)):
            mask = c_true_np == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], color=palette[ci % len(palette)],
                       s=4, alpha=0.7, linewidths=0, rasterized=True)
            cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
            ax.text(cx, cy, str(label), fontsize=7, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))
        ax.set_title("True labels", fontsize=9, fontweight="bold")
        _clean_tsne_ax(ax)
        panel_idx += 1

    for K_val in K_steps:
        labels = allocations[K_val]
        ax = fig.add_subplot(n_rows, n_cols, panel_idx)

        for ci, label in enumerate(np.unique(labels)):
            mask = labels == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], color=palette[ci % len(palette)],
                       s=4, alpha=0.7, linewidths=0, rasterized=True)
            cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
            ax.text(cx, cy, str(label), fontsize=7, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))

        if c_true_np is not None:
            ari = adjusted_rand_score(c_true_np, labels)
            title = f"K_init={K_init} -> K={K_val}  |  ARI={ari:.3f}"
        else:
            title = f"K_init={K_init} -> K={K_val}"

        if K_val == K_true:
            title += "  *"
            ax.set_facecolor("#fffbe6")  # subtle gold tint for the true-K panel

        ax.set_title(title, fontsize=8.5)
        _clean_tsne_ax(ax)
        panel_idx += 1

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
