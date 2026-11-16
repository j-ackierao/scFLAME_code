"""
downstream_scanpy.py
====================
Python/Scanpy equivalent of downstream_seurat.R: bring scFLAME results back into AnnData.

    1. pick the best repeat (highest ELBO: model fit, needs no ground truth)
    2. add scFLAME clusters to adata.obs
    3. add the latent factors to adata.obsm["X_scflame"] (use like X_pca) and run UMAP on them
    4. compare scFLAME clusters with an existing annotation (if there is one)
    5. dot plot of scFLAME's top marker genes per cluster

Run after:  preprocess_scanpy.py (or preprocess.R)  ->  run_realdata.py --save-top-genes

Usage:
    python scripts/downstream_scanpy.py --adata data/my_data/processed.h5ad \\
        --results-dir results/my_data --label-key cell_type
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def align(df: pd.DataFrame, adata: sc.AnnData, what: str) -> pd.DataFrame:
    """Reorder a results table (with a cell_id column) to match adata.obs_names."""
    df = df.copy()
    df["cell_id"] = df["cell_id"].astype(str)
    missing = ~adata.obs_names.isin(df["cell_id"])
    if not missing.any():
        return df.set_index("cell_id").loc[adata.obs_names]
    if len(df) == adata.n_obs:
        print(f"WARNING: {what}: cell IDs don't match the AnnData, assuming the same cell order. "
              "Check this is the AnnData counts.csv was written from.")
        return df.set_index(adata.obs_names)
    raise SystemExit(f"{what}: {missing.sum()} cells in the AnnData are missing from the scFLAME output "
                     f"({len(df)} vs {adata.n_obs} cells). Use the AnnData counts.csv was written from.")


def add_embedding(adata: sc.AnnData, latents: pd.DataFrame, name: str) -> None:
    """Store latents in obsm['X_<name>'] and a UMAP of them in obsm['X_umap_<name>']."""
    adata.obsm[f"X_{name}"] = latents.drop(columns="cell_id", errors="ignore").to_numpy(dtype=np.float32)
    old_umap = adata.obsm.get("X_umap")
    sc.pp.neighbors(adata, use_rep=f"X_{name}", key_added=name)
    sc.tl.umap(adata, neighbors_key=name)
    adata.obsm[f"X_umap_{name}"] = adata.obsm["X_umap"]
    if old_umap is not None:  # don't overwrite an existing UMAP
        adata.obsm["X_umap"] = old_umap
    else:
        del adata.obsm["X_umap"]


def save_embedding(adata, basis, colors, path, titles=None):
    sc.pl.embedding(adata, basis=basis, color=colors, title=titles, legend_loc="on data",
                    legend_fontsize=7, frameon=False, show=False, ncols=len(colors), wspace=0.3)
    plt.gcf().savefig(path, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: {path}")


def run(args: argparse.Namespace) -> None:
    plot_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    adata = sc.read_h5ad(args.adata)
    label_key = args.label_key
    if label_key is not None and label_key not in adata.obs:
        print(f"WARNING: --label-key '{label_key}' not in adata.obs; skipping comparisons.")
        label_key = None

    # ---- 1. Pick a repeat ----
    summary = pd.read_csv(os.path.join(args.results_dir, "scflame_summary.csv"))
    summary["rep"] = summary["repeat"] + 1  # summary is 0-indexed, files are rep1, rep2, ...
    rep = args.repeat if args.repeat is not None else int(summary.loc[summary["elbo_final"].idxmax(), "rep"])
    print(summary.to_string(index=False))
    print(f"Using repeat {rep} (ELBO = {summary.loc[summary['rep'] == rep, 'elbo_final'].iloc[0]:.6g})")

    # ---- 2. Clusters ----
    alloc = align(pd.read_csv(os.path.join(args.results_dir, "scflame_allocations.csv")),
                  adata, "scflame_allocations.csv")
    clusters = alloc[f"rep{rep}_pred"].astype(int)
    adata.obs["scflame_cluster"] = pd.Categorical(clusters.astype(str).to_numpy(),
                                                  categories=[str(k) for k in sorted(clusters.unique())])

    # ---- 3. Latent factors -> UMAP ----
    latent_file = os.path.join(args.results_dir, f"scflame_latents_rep{rep}.csv")
    add_embedding(adata, align(pd.read_csv(latent_file), adata, os.path.basename(latent_file)), "scflame")

    colors = ["scflame_cluster"] + ([label_key] if label_key else [])
    save_embedding(adata, "umap_scflame", colors, os.path.join(plot_dir, "umap_clusters_vs_labels.png"))

    # ---- 4. Compare with an existing annotation ----
    # Each column (scFLAME cluster) shows how its cells split across the existing labels.
    if label_key:
        counts = pd.crosstab(adata.obs[label_key], adata.obs["scflame_cluster"])
        frac = counts / counts.sum(axis=0)
        fig, ax = plt.subplots(figsize=(0.45 * counts.shape[1] + 3, 0.35 * counts.shape[0] + 1.5))
        im = ax.imshow(frac.to_numpy(), cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(counts.shape[1]), counts.columns)
        ax.set_yticks(range(counts.shape[0]), counts.index)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                if counts.iat[i, j] > 0:
                    ax.text(j, i, counts.iat[i, j], ha="center", va="center", fontsize=6)
        ax.set_xlabel("scFLAME cluster")
        ax.set_ylabel(label_key)
        fig.colorbar(im, ax=ax, label="Fraction of cluster")
        path = os.path.join(plot_dir, "cluster_vs_label_heatmap.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ---- 5. Top marker genes per cluster ----
    # scFLAME scores each gene by how much a cluster's latent centre shifts that gene's expression
    # away from the average cell; the dot plot shows the observed expression of those genes.
    genes_file = os.path.join(args.results_dir, "scflame_top_genes.csv")
    if os.path.exists(genes_file):
        top = pd.read_csv(genes_file)
        if not {"cluster", "gene"} <= set(top.columns):
            raise SystemExit(f"Expected 'cluster' and 'gene' columns in scflame_top_genes.csv, "
                             f"found {list(top.columns)}. Edit the column names here.")
        if "repeat" in top:
            top = top[top["repeat"] == rep]
        top = top.groupby("cluster", sort=True).head(args.n_top_genes)
        missing = sorted(set(top["gene"]) - set(adata.var_names))
        if missing:
            print(f"{len(missing)} marker genes aren't in the AnnData and are dropped: {missing[:10]}")
        markers = {str(k): [g for g in dict.fromkeys(grp["gene"]) if g in adata.var_names]
                   for k, grp in top.groupby("cluster")}
        markers = {k: v for k, v in markers.items() if v}

        counts_layer = "counts" if "counts" in adata.layers else None
        lognorm = sc.pp.normalize_total(adata, layer=counts_layer, inplace=False)["X"]
        adata.layers["scflame_lognorm"] = np.log1p(lognorm)  # for plotting only
        sc.pl.dotplot(adata, markers, groupby="scflame_cluster", layer="scflame_lognorm",
                      standard_scale="var", show=False)
        path = os.path.join(plot_dir, "top_genes_dotplot.png")
        plt.gcf().savefig(path, bbox_inches="tight")
        plt.close("all")
        print(f"Saved: {path}")
    else:
        print("No scflame_top_genes.csv; rerun run_realdata.py with --save-top-genes for the dot plot.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load scFLAME results into an AnnData object and plot them.")
    p.add_argument("--adata", required=True, help="processed.h5ad from preprocess_scanpy.py, or your own")
    p.add_argument("--results-dir", required=True, help="--out-dir used for run_realdata.py")
    p.add_argument("--repeat", type=int, default=None, help="Repeat to use (default: highest ELBO)")
    p.add_argument("--label-key", default=None, help="adata.obs column with an existing annotation")
    p.add_argument("--n-top-genes", type=int, default=5, help="Marker genes per cluster in the dot plot")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())