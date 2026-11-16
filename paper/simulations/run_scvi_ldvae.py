"""
benchmark_scvi.py
Benchmarks scVI and LinearSCVI (k-means on the latent space) against
one simulated dataset's ground-truth labels. 
"""

import argparse
import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

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
    p = argparse.ArgumentParser(description="scVI / LinearSCVI benchmark on a simulated dataset")
    p.add_argument("--counts_path", required=True)
    p.add_argument("--clusters_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--accelerator", default="cpu",
                    help="Passed to scvi model.train(); CPU by default, matching the original script")
    p.add_argument("--replicate", type=int, default=None,
                    help="Defaults to $SLURM_ARRAY_TASK_ID - 1 if unset")
    p.add_argument("--scenario", default=None,
                    help="Optional scenario tag (e.g. Easy/Medium/Hard), added to the output filename and summary")
    return p.parse_args()


def run_one_model(adata, model_cls, model_key, latent_key, args, task_id, repeat, out_dir, K_true, c_true):
    model_cls.setup_anndata(adata)
    model = model_cls(adata, n_latent=10)
    model.train(accelerator=args.accelerator, check_val_every_n_epoch=10)

    adata.obsm[latent_key] = model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata)

    kmeans_key = f"kmeans_{model_key}"
    kmeans = KMeans(n_clusters=K_true, random_state=0)
    adata.obs[kmeans_key] = kmeans.fit_predict(adata.obsm[latent_key]).astype(str)
    pred = adata.obs[kmeans_key].values.astype(int)

    return {
        "model": model_key,
        "scenario": args.scenario,
        "repeat": repeat, "K_true": K_true,
        "K_final": len(np.unique(pred)),
        "ari_final": adjusted_rand_score(c_true, pred),
        "nmi_final": normalized_mutual_info_score(c_true, pred),
        "acc_final": clustering_accuracy(c_true, pred),
    }


def main():
    args = parse_args()
    task_id = args.replicate if args.replicate is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) - 1

    torch.manual_seed(42 + task_id)
    np.random.seed(42 + task_id)

    os.makedirs(args.out_dir, exist_ok=True)

    dat = load_simulated_dataset(args.counts_path, args.clusters_path)
    X_np, c_true, K_true = dat["X_np"], dat["c_true"], dat["K_true"]
    print(f"N={X_np.shape[0]}, D={X_np.shape[1]}, K_true={K_true}")

    adata = ad.AnnData(X_np)
    adata.obs["celltype"] = pd.Categorical(c_true.astype(str))

    model_registry = {"scvi": (scvi.model.SCVI, "X_scVI"), "linear_scvi": (scvi.model.LinearSCVI, "X_LinearSCVI")}

    summary_rows = []
    for repeat in range(3):
        print(f"Starting Repeat {repeat + 1}/3...")
        repeat_seed = (task_id * 100) + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        for model_name in ["scvi", "linear_scvi"]:
            model_cls, latent_key = model_registry[model_name]
            row = run_one_model(adata, model_cls, model_name, latent_key, args, task_id, repeat, args.out_dir, K_true, c_true)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if args.scenario is not None:
        out_path = os.path.join(args.out_dir, f"scvi_{args.scenario}_{task_id}_summary.csv")
    else:
        out_path = os.path.join(args.out_dir, f"scvi_{task_id}_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()