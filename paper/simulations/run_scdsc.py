"""
run_scdsc.py
Benchmarks scDSC on one simulated dataset.
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
    p = argparse.ArgumentParser(description="scDSC benchmark on a simulated dataset")
    p.add_argument("--counts_path", required=True)
    p.add_argument("--clusters_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cpu",
                    help="Device to be run on: original used cuda.")
    p.add_argument("--replicate", type=int, default=None,
                    help="Defaults to $SLURM_ARRAY_TASK_ID - 1 if unset")
    p.add_argument("--scenario", default=None,
                    help="Optional scenario tag (e.g. Easy/Medium/Hard), added to the "
                         "output filename and summary")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    task_id = args.replicate if args.replicate is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) - 1

    torch.manual_seed(42 + task_id)
    np.random.seed(42 + task_id)

    os.makedirs(args.out_dir, exist_ok=True)

    dat = load_simulated_dataset(args.counts_path, args.clusters_path)
    X_np, c_true, K_true = dat["X_np"], dat["c_true"], dat["K_true"]
    print(f"N={X_np.shape[0]}, D={X_np.shape[1]}, K_true={K_true}")

    adata = ad.AnnData(X)
    adata.obs["celltype"] = pd.Categorical(c_true.astype(str))

    summary_rows = []

    for repeat in range(3):
        print(f"Starting Repeat {repeat + 1}/3...")
        repeat_seed = (task_id * 100) + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        adata_tmp = sc.AnnData(X_np.copy())
        sc.pp.normalize_total(adata_tmp, target_sum=1e4)
        sc.pp.log1p(adata_tmp)
        X_norm = adata_tmp.X_np.astype(np.float32)

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
        pretrain_ae(model.ae, X, epochs=200, batch_size=256, lr=1e-3)
        with torch.no_grad():
            _, _, _, _, z, _ = model.ae(X)

        kmeans = KMeans(n_clusters=K_true, n_init=20, random_state=seed)
        kmeans.fit_predict(z.cpu().numpy())
        model.cluster_layer.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        )

        # Balance parameters matching original
        Balance_para = [0.1, 0.01, 1.0, 0.1]

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

        final_loss = loss.item()
        final_latent = z.detach().cpu().numpy()
        final_clusters = q.argmax(1).cpu().numpy()

        ari_final = adjusted_rand_score(c_true, final_clusters)
        nmi_final = normalized_mutual_info_score(c_true, final_clusters)
        acc_final = clustering_accuracy(c_true, final_clusters)

        print(f"Repeat {repeat+1} | ARI: {ari_final:.3f} | NMI: {nmi_final:.3f} | ACC: {acc_final:.3f}")

        summary_rows.append({
            'model':     'scdsc',
            'scenario': args.scenario,
            'repeat':    repeat,
            'K_true':    K_true,
            'K_final':   len(np.unique(final_clusters)),
            'ari_final': ari_final,
            'nmi_final': nmi_final,
            'acc_final': acc_final,
        })

    out_dir = args.out_dir
    summary_df = pd.DataFrame(summary_rows)
    if args.scenario is not None:
        out_path = os.path.join(args.out_dir, f"scdsc_{args.scenario}_{task_id}_summary.csv")
    else:
        out_path = os.path.join(args.out_dir, f"scdsc_{task_id}_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()