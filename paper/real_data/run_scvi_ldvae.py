"""
run_scvi_ldvae.py
Benchmarks scVI and LDVAE/LinearSCVI (k-means on the latent space) on one real dataset.
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

def clustering_accuracy(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize
    return cm[row_ind, col_ind].sum() / cm.sum()


def parse_args():
    p = argparse.ArgumentParser(description="scVI / LinearSCVI benchmark on a real dataset")
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
    p.add_argument("--accelerator", default="cpu",
                    help="Passed to scvi model.train(); CPU by default, matching the original script")
    return p.parse_args()


def run_one_model(adata, model_cls, model_key, latent_key, args, repeat, K_true, c_true, best_info):
    model_cls.setup_anndata(adata)
    model = model_cls(adata, n_latent=10)
    model.train(
        accelerator=args.accelerator,
        check_val_every_n_epoch=10,
        **(
            {"plan_kwargs": {"lr": 1e-3}}
            if model_key == "scvi"
            else {}
        ),
    )
    current_elbo = -model.history["elbo_validation"].iloc[-1, 0]

    latent = model.get_latent_representation()
    adata.obsm[latent_key] = latent
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata)

    kmeans_key = f"kmeans_{model_key}"
    kmeans = KMeans(n_clusters=K_true, random_state=42 + repeat, n_init=10)
    adata.obs[kmeans_key] = kmeans.fit_predict(adata.obsm[latent_key]).astype(str)
    pred = adata.obs[kmeans_key].values.astype(int)

    row = {
        "model": model_key, "repeat": repeat, "K_true": K_true,
        "K_final": len(np.unique(pred)),
        "elbo": current_elbo,
        "ari_final": adjusted_rand_score(c_true, pred),
        "nmi_final": normalized_mutual_info_score(c_true, pred),
        "acc_final": clustering_accuracy(c_true, pred),
    }

    if current_elbo > best_info[model_key]["elbo"]:
        best_info[model_key] = {
            "elbo": current_elbo,
            "repeat": repeat,
            "latent": latent.copy(),
            "clusters": pred.copy(),
        }

        print(
            f"New best {model_key} model: "
            f"repeat={repeat}, ELBO={current_elbo:.4f}"
        )

    return row, best_info


def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(args.out_dir, exist_ok=True)

    loader = LOADERS[args.dataset]

    dat = loader(counts_path=args.counts_path, clusters_path=args.clusters_path)

    X, c_true, K_true = dat["X"], dat["c_true"], dat["K_true"]
    X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    print(f"N={X.shape[0]}, D={X.shape[1]}, K_true={K_true}")

    adata = ad.AnnData(X.astype(np.float32))
    if "cells" in dat:
        adata.obs_names = [str(c) for c in dat["cells"]]
    adata.obs["celltype"] = pd.Categorical(c_true.astype(str))

    model_registry = {"scvi": (scvi.model.SCVI, "X_scVI"), "linear_scvi": (scvi.model.LinearSCVI, "X_LinearSCVI")}

    summary_rows = []

    best_info = {
        model_name: {
            "elbo": -np.inf,
            "repeat": None,
            "latent": None,
            "clusters": None,
        }
        for model_name in model_registry
    }
    for repeat in range(5):
        print(f"Starting Repeat {repeat + 1}/5...")
        repeat_seed = 42 + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        for model_name in ["scvi", "linear_scvi"]:
            model_cls, latent_key = model_registry[model_name]
            row, best_info = run_one_model(adata, model_cls, model_name, latent_key, args, repeat, K_true, c_true, best_info)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(args.out_dir, f"scvi_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # 2. Save Best Latent Spaces and Clusters
    for m_name in ["scvi", "linear_scvi"]:
        # Save Latent Matrix (N x 10)
        latent_df = pd.DataFrame(best_info[m_name]["latent"], index=adata.obs_names)
        latent_df.to_csv(f"{args.out_dir}/{m_name}_latents.csv")
        
        # Save Cluster Labels
        cluster_df = pd.DataFrame({
            'cell_id': adata.obs_names,
            'cluster': best_info[m_name]["clusters"]
        })
        cluster_df.to_csv(f"{args.out_dir}/{m_name}_clusters.csv", index=False)


if __name__ == "__main__":
    main()