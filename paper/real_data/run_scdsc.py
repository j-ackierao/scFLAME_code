"""
run_scdsc.py
Benchmarks scDSC on one real dataset.
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comparison_models.scdsc import *

def clustering_accuracy(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize
    return cm[row_ind, col_ind].sum() / cm.sum()


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
    p = argparse.ArgumentParser(description="scVI / LinearSCVI benchmark on a simulated dataset")
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
    p.add_argument("--device", default="cpu",
                    help="Device to run on: original code used cuda.")
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device

    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(args.out_dir, exist_ok=True)

    loader = LOADERS[args.dataset]

    dat = loader(counts_path=args.counts_path, clusters_path=args.clusters_path)

    X_np, c_true, K_true = dat["X"], dat["c_true"], dat["K_true"]
    print(f"N={X_np.shape[0]}, D={X_np.shape[1]}, K_true={K_true}")
    X_np = X_np.detach().cpu().numpy() if torch.is_tensor(X_np) else np.asarray(X_np)
    adata = ad.AnnData(X_np)
    
    adata.obs["celltype"] = pd.Categorical(c_true.astype(str))

    summary_rows = []

    best_info = {
        "scDSC": {"loss": -np.inf, "latent": None, "clusters": None}
    }

    import time

    for repeat in range(5):
        print(f"Starting Repeat {repeat + 1}/5...")
        repeat_seed = 42 + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        adata_tmp = sc.AnnData(X_np.copy())
        sc.pp.normalize_total(adata_tmp, target_sum=1e4)
        sc.pp.log1p(adata_tmp)
        X_norm = adata_tmp.X.astype(np.float32)

        # Size factors (library size / median library size)
        lib_sizes = X_np.sum(axis=1)
        sf = (lib_sizes / np.median(lib_sizes)).astype(np.float32)

        X_raw_np = X_np.copy()  # keep raw counts for ZINB

        model = SDCN(
            n_enc_1=500, n_enc_2=500, n_enc_3=2000,
            n_dec_1=2000, n_dec_2=500, n_dec_3=500,
            n_input=X_np.shape[1], n_z=10, n_clusters=K_true
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        X = torch.tensor(X_norm, dtype=torch.float32).to(device)
        X_raw_t = torch.tensor(X_raw_np, dtype=torch.float32).to(device)
        sf_t = torch.tensor(sf, dtype=torch.float32).to(device)
        adj_t = construct_and_load_graph(X.copy(), method="ncos").to(device)

        # KMeans init (counted as part of pretrain phase)
        t0 = time.time()
        pretrain_ae(model.ae, X, epochs=200, batch_size=256, lr=1e-3)
        with torch.no_grad():
            _, _, _, _, z, _ = model.ae(X)

        kmeans = KMeans(n_clusters=K_true, n_init=20, random_state=repeat_seed)
        kmeans.fit_predict(z.cpu().numpy())
        model.cluster_layer.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        )
        pretrain_time = time.time() - t0

        # Balance parameters matching original
        Balance_para = [0.1, 0.01, 1.0, 0.1]

        best_epoch_loss = np.inf
        best_latent = None
        best_clusters = None

        t1 = time.time()
        for epoch in range(200):

            with torch.no_grad():
                _, tmp_q, _, _, _, _, _, _ = model(X, adj_t)
            
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)     

            x_bar, q, pred, z, mean, disp, pi, zinb_loss_fn = model(X, adj_t)

            # Four losses matching original exactly
            bce_loss  = F.binary_cross_entropy(q, p)
            ce_loss = F.kl_div(pred.clamp(min=1e-8).log(), p, reduction='batchmean')
            re_loss   = F.mse_loss(x_bar, X)
            zinb_loss = zinb_loss_fn(X_raw_t, mean, disp, pi, sf_t)
            print(f"bce={bce_loss.item():.4f} ce={ce_loss.item():.4f} re={re_loss.item():.4f} zinb={zinb_loss.item():.4f}")

            loss = (Balance_para[0] * bce_loss +
                    Balance_para[1] * ce_loss  +
                    Balance_para[2] * re_loss  +
                    Balance_para[3] * zinb_loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if epoch % 10 == 0:
                pred_labels = q.argmax(1).cpu().numpy()
                ari = adjusted_rand_score(c_true, pred_labels)
                nmi = normalized_mutual_info_score(c_true, pred_labels)
                print(f"[Epoch {epoch}] Loss={loss.item():.4f} ARI={ari:.4f} NMI={nmi:.4f}")


        train_time = time.time() - t1
        total_time = pretrain_time + train_time

        best_epoch_loss = loss.item()
        best_latent = z.detach().cpu().numpy()
        best_clusters = q.argmax(1).cpu().numpy()

        # Final number of clusters actually produced
        n_clusters_final = len(np.unique(best_clusters))

        print(f"--> Repeat {repeat}: n_clusters={n_clusters_final} | "
            f"pretrain={pretrain_time:.1f}s, train={train_time:.1f}s, total={total_time:.1f}s")
        
        summary_rows.append({
            "model": "scDSC",
            "repeat": repeat,
            "loss": best_epoch_loss,
            "ari": adjusted_rand_score(c_true, best_clusters),
            "nmi": normalized_mutual_info_score(c_true, best_clusters),
            "acc": clustering_accuracy(c_true, best_clusters),
            "K_final": n_clusters_final,
            "pretrain_time_s": pretrain_time,
            "train_time_s": train_time,
            "total_time_s": total_time
        })

        if best_epoch_loss < best_info["scDSC"]["loss"] or best_info["scDSC"]["loss"] == -np.inf:
            best_info["scDSC"]["loss"] = best_epoch_loss
            best_info["scDSC"]["latent"] = best_latent
            best_info["scDSC"]["clusters"] = best_clusters


    out_dir = args.out_dir
    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(args.out_dir, f"scdsc_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    latent_df = pd.DataFrame(
        best_info["scDSC"]["latent"],
        index=adata.obs_names
    )
    latent_df.to_csv(f"{out_dir}/scDSC_latents.csv")

    cluster_df = pd.DataFrame({
        "cell_id": adata.obs_names,
        "cluster": best_info["scDSC"]["clusters"]
    })
    cluster_df.to_csv(f"{out_dir}/scDSC_clusters.csv", index=False)

    print(f"\nFinished! Files saved to: {out_dir}")


if __name__ == "__main__":
    main()