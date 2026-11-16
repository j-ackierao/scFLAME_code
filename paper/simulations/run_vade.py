"""
run_vade.py
Benchmarks VaDE on one simulated dataset.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comparison_models.vade import VaDE, Encoder, Decoder


def clustering_accuracy(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize
    return cm[row_ind, col_ind].sum() / cm.sum()


def load_simulated_dataset(counts_path: str, clusters_path: str) -> dict:
    """
    Top-3000 HVGs (by variance of log1p counts). For VaDE, library sizes are
    computed on the FULL gene matrix before HVG subsetting, then counts are
    library-size normalised and log1p transformed.
    """
    df = pd.read_csv(counts_path, index_col=0)

    log_df  = np.log1p(df.values)
    col_var = log_df.var(axis=0)
    top_idx = np.argsort(col_var)[-3000:]

    # Log transformation with library size normalisation (full gene matrix)
    full_X_np = df.values.astype(np.float32)
    libsize = full_X_np.sum(axis=1, keepdims=True)
    s0 = np.median(libsize)

    X_norm = df / libsize * s0
    X_log = np.log1p(X_norm)

    # Subset to top HVGs
    df_hv = X_log.iloc[:, top_idx]
    X_np = df_hv.values.astype(np.float32)

    clusters_df = pd.read_csv(clusters_path)
    c_true = clusters_df["celltype"].astype(int).to_numpy()
    K_true = len(np.unique(c_true))

    return {"X_np": X_np, "c_true": c_true, "K_true": K_true, "genes": list(df_hv.columns)}


def parse_args():
    p = argparse.ArgumentParser(description="VaDE benchmark on a simulated dataset")
    p.add_argument("--counts_path", required=True)
    p.add_argument("--clusters_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run on: original code used cuda if available.")
    p.add_argument("--replicate", type=int, default=None,
                    help="Defaults to $SLURM_ARRAY_TASK_ID - 1 if unset")
    p.add_argument("--scenario", default=None,
                    help="Optional scenario tag (e.g. Easy/Medium/Hard), added to the output filename and summary")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args_cli = parse_args()
    device = args_cli.device
    task_id = args_cli.replicate if args_cli.replicate is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) - 1

    torch.manual_seed(42 + task_id)
    np.random.seed(42 + task_id)

    os.makedirs(args_cli.out_dir, exist_ok=True)

    dat = load_simulated_dataset(args_cli.counts_path, args_cli.clusters_path)
    X_np, true, K_true = dat["X_np"], dat["c_true"], dat["K_true"]

    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    N, D = X.shape
    print(f"N={N}, D={D}, K_true={K_true}")

    class Args:
        pass

    args = Args()
    args.nClusters = K_true
    args.hid_dim   = 10
    args.input_dim = 784 # the layers are built and replaced with D-dimensional ones; this is kept for consistency in rng
    args.cuda      = (device == "cuda")

    dataset = TensorDataset(X, torch.zeros(N, dtype=torch.int64))
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    summary_rows = []

    for repeat in range(3):
        print(f"\nStarting Repeat {repeat+1}/3...")

        repeat_seed = (task_id * 100) + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        pretrain_path = os.path.join(args_cli.out_dir, f"vade_pretrain_{task_id}.pk")
        if os.path.exists(pretrain_path):
            os.remove(pretrain_path)

        t0 = time.time()

        vade = VaDE(args).to(device)

        # Replace encoder/decoder with the correct input dimension
        vade.encoder = Encoder(input_dim=D, inter_dims=[500, 500, 2000], hid_dim=10).to(device)
        vade.decoder = Decoder(input_dim=D, inter_dims=[500, 500, 2000], hid_dim=10).to(device)

        optimizer = torch.optim.Adam(vade.parameters(), lr=2e-4)

        # Pre-train
        vade.pre_train(dataloader, pretrain_path, pre_epoch=30)
        os.remove(pretrain_path)  # temporary file only
        pretrain_time = time.time() - t0

        # Train
        t1 = time.time()
        for epoch in range(200):
            vade.train()
            total_loss = 0
            for x_batch, _ in dataloader:
                optimizer.zero_grad()
                loss = vade.ELBO_Loss(x_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vade.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                temp_pred = vade.predict(X)
                unique, counts = np.unique(temp_pred, return_counts=True)
                print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}, '
                      f'Clusters: {len(unique)}, Sizes: {counts}')

        # Predict
        vade.eval()
        with torch.no_grad():
            pred_vade = vade.predict(X)
        train_time = time.time() - t1
        total_time = pretrain_time + train_time

        # Evaluate
        ari_vade = adjusted_rand_score(true, pred_vade)
        nmi_vade = normalized_mutual_info_score(true, pred_vade)
        acc_vade = clustering_accuracy(true, pred_vade)

        print(f"Repeat {repeat+1} | ARI: {ari_vade:.3f} | NMI: {nmi_vade:.3f} | ACC: {acc_vade:.3f} | "
              f"pretrain={pretrain_time:.1f}s, train={train_time:.1f}s, total={total_time:.1f}s")

        summary_rows.append({
            'model':           'VaDE_PyTorch',\
            'scenario':       args_cli.scenario,
            'repeat':          repeat,
            'K_true':          K_true,
            'K_final':         len(np.unique(pred_vade)),
            'ari_final':       ari_vade,
            'nmi_final':       nmi_vade,
            'acc_final':       acc_vade,
            'pretrain_time_s': pretrain_time,
            'train_time_s':    train_time,
            'total_time_s':    total_time
        })

    summary_df = pd.DataFrame(summary_rows)
    if args_cli.scenario is not None:
        out_path = os.path.join(args_cli.out_dir, f"vade_{args_cli.scenario}_{task_id}_summary.csv")
    else:
        out_path = os.path.join(args_cli.out_dir, f"vade_{task_id}_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()