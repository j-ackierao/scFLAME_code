"""
run_scdeepcluster.py
Benchmarks scDeepCluster on one simulated dataset.
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


# Model settings for the simulation benchmark
DTYPE = torch.float64
MEAN_MAX = 1e6
 
 
def clustering_accuracy(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize
    return cm[row_ind, col_ind].sum() / cm.sum()
 
 
def load_simulated_dataset(counts_path: str, clusters_path: str) -> dict:
    df = pd.read_csv(counts_path, index_col=0)
 
    log_df = np.log1p(df.values)
    col_var = log_df.var(axis=0)
    top_idx = np.argsort(col_var)[-3000:]
    df = df.iloc[:, top_idx]
 
    X_np = df.values.astype(np.float32)
 
    clusters_df = pd.read_csv(clusters_path)
    c_true = clusters_df["celltype"].astype(int).to_numpy()
    K_true = len(np.unique(c_true))
 
    result = {"X_np": X_np, "c_true": c_true, "K_true": K_true, "genes": list(df.columns)}
 
    return result
 
 
def parse_args():
    p = argparse.ArgumentParser(description="scDeepCluster benchmark on a simulated dataset")
    p.add_argument("--counts_path", required=True)
    p.add_argument("--clusters_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run on: original code used cuda if available.")
    p.add_argument("--replicate", type=int, default=None,
                    help="Defaults to $SLURM_ARRAY_TASK_ID - 1 if unset")
    p.add_argument("--scenario", default=None,
                    help="Optional scenario tag (e.g. Easy/Medium/Hard), added to the "
                         "output filename and summary")
    return p.parse_args()
 
 # Preprocessing

def prepare_data(counts_path, clusters_path, device):
    dat = load_simulated_dataset(counts_path, clusters_path)
    X, c_true, K_true = dat["X_np"], dat["c_true"], dat["K_true"]
 
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
 
    X_norm = torch.tensor(adata.X, dtype=DTYPE).to(device)
    X_raw  = torch.tensor(adata.raw.X, dtype=DTYPE).to(device)
    size_factors = torch.tensor(adata.obs['size_factors'].values, dtype=DTYPE).to(device)
 
    return adata, X_norm, X_raw, size_factors, true, K_true
 
# Main
 
def main():
    args = parse_args()
    device = args.device
    task_id = args.replicate if args.replicate is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) - 1
 
    torch.manual_seed(42 + task_id)
    np.random.seed(42 + task_id)
 
    os.makedirs(args.out_dir, exist_ok=True)
 
    adata, X_norm, X_raw, size_factors, true, K_true = prepare_data(
        args.counts_path, args.clusters_path, device
    )
 
    input_size = adata.n_vars
    print(f"N={adata.n_obs}, D={input_size}, K_true={K_true}")
 
    summary_rows = []
 
    for repeat in range(3):
        torch.cuda.empty_cache()
        print(f"\n--- Starting Repeat {repeat+1}/3 ---")
 
        repeat_seed = (task_id * 100) + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)
 
        model = scDeepCluster(
            input_dim=input_size,
            z_dim=32,
            encodeLayer=[256, 64],
            decodeLayer=[64, 256],
            device=device,
            dtype=DTYPE,
            mean_max=MEAN_MAX
        ).to(device)
 
        # Step 1: Pretrain
        t0 = time.time()
        model.pretrain_autoencoder(
            X=X_norm, X_raw=X_raw, size_factor=size_factors,
            epochs=600, lr=1e-3, ae_save=False
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
 
        # Convert outputs to NumPy
        pred_scdc = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.asarray(y_pred)
        # Final number of clusters actually produced
        n_clusters_final = len(np.unique(pred_scdc))
 
        # Calculate Metrics
        ari = adjusted_rand_score(true, pred_scdc)
        nmi = normalized_mutual_info_score(true, pred_scdc)
        acc = clustering_accuracy(true, pred_scdc)
 
        print(f"--> Repeat {repeat}: n_clusters={n_clusters_final} | ARI={ari:.3f} NMI={nmi:.3f} ACC={acc:.3f} | "
              f"pretrain={pretrain_time:.1f}s, train={train_time:.1f}s, total={total_time:.1f}s")
 
        summary_rows.append({
            'model': 'scdeepcluster',
            'scenario': args.scenario,
            'repeat': repeat,
            'K_true': K_true,
            'K_final': n_clusters_final,
            'ari_final': ari,
            'nmi_final': nmi,
            'acc_final': acc,
            'pretrain_time_s': pretrain_time,
            'train_time_s': train_time,
            'total_time_s': total_time
        })
 
        del model
        torch.cuda.empty_cache()
 
    summary_df = pd.DataFrame(summary_rows)
    if args.scenario is not None:
        out_path = os.path.join(args.out_dir, f"scdeepcluster_{args.scenario}_{task_id}_summary.csv")
    else:
        out_path = os.path.join(args.out_dir, f"scdeepcluster_{task_id}_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
 
 
if __name__ == "__main__":
    main()
 