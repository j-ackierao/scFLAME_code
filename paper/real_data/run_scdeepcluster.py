"""
run_scdeepcluster.py
Benchmarks scDeepCluster on one real dataset.

Dataset-specific settings:
  - segerstolpe: gradient clipping (max_norm=5.0) is used during pretraining.
  This was due to differing sequencing protocol; SmartSeq2 is much less sparse.
  We ran into numerical errors without this; other UMI-based datasets did not have this error.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comparison_models.scdeepcluster import scDeepCluster

from load_data import (
    load_uterus,
    load_zeisel,
    load_tm_spleen,
    load_segerstolpe,
    load_baron,
    load_pbmc4k,
)

LOADERS = {
    "uterus": load_uterus,
    "zeisel": load_zeisel,
    "tm_spleen": load_tm_spleen,
    "segerstolpe": load_segerstolpe,
    "baron": load_baron,
    "pbmc4k": load_pbmc4k,
}

# Gradient-norm clipping during pretraining (None = no clipping, original behaviour)
PRETRAIN_CLIP_GRAD = {
    "segerstolpe": 5.0,
}


def clustering_accuracy(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize
    return cm[row_ind, col_ind].sum() / cm.sum()


def parse_args():
    p = argparse.ArgumentParser(description="scDeepCluster benchmark on a real dataset")
    p.add_argument("--dataset", required=True,
        choices=[
            "pbmc4k",
            "uterus",
            "zeisel",
            "tm_spleen",
            "segerstolpe",
            "baron",
        ],
        help="Dataset to load",
    )
    p.add_argument("--counts_path", required=True)
    p.add_argument("--clusters_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run on: original code used cuda if available.")
    return p.parse_args()


# ============================================================
# Preprocessing
# ============================================================

def prepare_data(dataset, counts_path, clusters_path, device):
    loader = LOADERS[dataset]
    dat = loader(counts_path=counts_path, clusters_path=clusters_path)
    X, c_true = dat["X"], dat["c_true"]

    X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    
    adata = ad.AnnData(X.astype(np.float32))

    # Store cell types in obs so Scanpy keeps them aligned during filtering
    adata.obs["celltype"] = pd.Categorical(c_true.astype(str))

    # Basic filtering
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)

    # Extract final true labels and K_true after potential filtering
    K_true = len(adata.obs["celltype"].cat.categories)
    true = adata.obs["celltype"].cat.codes.values

    counts = adata.X.copy()
    adata.obs['size_factors'] = counts.sum(1) / np.median(counts.sum(1))

    adata.raw = adata.copy()

    # Normalize, log transform, and scale
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    X_norm = torch.tensor(adata.X, dtype=torch.float32).to(device)
    X_raw  = torch.tensor(adata.raw.X, dtype=torch.float32).to(device)
    size_factors = torch.tensor(adata.obs['size_factors'].values, dtype=torch.float32).to(device)

    return adata, X_norm, X_raw, size_factors, true, K_true


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = args.device
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    adata, X_norm, X_raw, size_factors, true, K_true = prepare_data(
        args.dataset, args.counts_path, args.clusters_path, device
    )

    clip_grad = PRETRAIN_CLIP_GRAD.get(args.dataset, None)

    input_size = adata.n_vars
    print(f"N={adata.n_obs}, D={input_size}, K_true={K_true}, clip_grad={clip_grad}")

    summary_rows = []

    for repeat in range(5):
        torch.cuda.empty_cache()
        print(f"\n--- Starting Repeat {repeat+1}/5 ---")

        # Static seeds for reproducibility
        torch.manual_seed(25 + repeat)
        np.random.seed(25 + repeat)
        torch.cuda.manual_seed_all(25 + repeat)

        model = scDeepCluster(
            input_dim=input_size,
            z_dim=32,
            encodeLayer=[256, 64],
            decodeLayer=[64, 256],
            device=device
        ).to(device)

        # Step 1: Pretrain
        t0 = time.time()
        model.pretrain_autoencoder(
            X=X_norm, X_raw=X_raw, size_factor=size_factors,
            epochs=600, lr=1e-3, ae_save=False, clip_grad=clip_grad
        )
        pretrain_time = time.time() - t0

        # Step 2: Fit (includes KMeans initialisation of cluster centres)
        t1 = time.time()
        y_pred, _, _, _, _ = model.fit(
            X=X_norm, X_raw=X_raw, size_factor=size_factors,
            n_clusters=K_true
        )
        train_time = time.time() - t1
        total_time = pretrain_time + train_time

        with torch.no_grad():
            z = model.encodeBatch(X_norm)

        # Convert outputs to NumPy
        pred_scdc = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.asarray(y_pred)
        latent_scdc = z.cpu().numpy() if torch.is_tensor(z) else np.asarray(z)

        # Final number of clusters actually produced
        n_clusters_final = len(np.unique(pred_scdc))

        # Calculate Metrics
        ari = adjusted_rand_score(true, pred_scdc)
        nmi = normalized_mutual_info_score(true, pred_scdc)
        acc = clustering_accuracy(true, pred_scdc)

        print(f"--> Repeat {repeat}: n_clusters={n_clusters_final} | "
              f"pretrain={pretrain_time:.1f}s, train={train_time:.1f}s, total={total_time:.1f}s")

        summary_rows.append({
            'model': 'scdeepcluster',
            'repeat': repeat,
            'ari_final': ari,
            'nmi_final': nmi,
            'acc_final': acc,
            'K_final': n_clusters_final,
            'pretrain_time_s': pretrain_time,
            'train_time_s': train_time,
            'total_time_s': total_time
        })

        del model
        torch.cuda.empty_cache()

        # --- Save latents and clusters for repeat 0 ---
        if repeat == 0:
            latent_df = pd.DataFrame(latent_scdc, index=adata.obs_names)
            latent_df.to_csv(f"{out_dir}/scdeepcluster_best_latent.csv")

            cluster_out = pd.DataFrame({
                'cell_id': adata.obs_names,
                'cluster': pred_scdc
            })
            cluster_out.to_csv(f"{out_dir}/scdeepcluster_best_clusters.csv", index=False)
            print(f"--> Saved latent space and clusters from Repeat 0 for visualization.")

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(out_dir, "scdeepcluster_summary.csv")
    summary_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()