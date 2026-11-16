"""
run_vade.py
Benchmarks VaDE on one real dataset.
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
from comparison_models.vade import VaDE

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
    p = argparse.ArgumentParser(description="VaDE benchmark on a real dataset")
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

def prepare_data(dataset, counts_path, clusters_path):
    loader = LOADERS[dataset]
    dat = loader(counts_path=counts_path, clusters_path=clusters_path)
    X, c_true = dat["X"], dat["c_true"]

    X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    
    # Library-size normalisation + log1p
    X_np    = np.asarray(X).astype(np.float32)
    libsize = X_np.sum(axis=1, keepdims=True)
    s0      = np.median(libsize)
    X_norm  = X_np / libsize * s0
    X_log   = np.log1p(X_norm).astype(np.float32)

    true_labels = pd.Categorical(c_true).codes.astype(np.int64)
    K_true      = len(np.unique(true_labels))

    return X_log, true_labels, K_true


# ============================================================
# Main
# ============================================================

def main():
    args_cli = parse_args()
    device = args_cli.device
    out_dir = args_cli.out_dir
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(out_dir, exist_ok=True)

    X_log, true_labels, K_true = prepare_data(
        args_cli.dataset, args_cli.counts_path, args_cli.clusters_path
    )

    N, D = X_log.shape
    print(f"N={N}, D={D}, K_true={K_true}")
    cell_ids = [str(i) for i in range(N)]

    X = torch.tensor(X_log, dtype=torch.float32).to(device)

    class Args:
        pass

    args = Args()
    args.nClusters = K_true
    args.hid_dim   = 10
    args.input_dim = D
    args.cuda      = (device == "cuda")

    summary_rows = []
    best_info = {"loss": np.inf, "latent": None, "clusters": None}

    for repeat in range(5):
        print(f"\n{'='*50}")
        print(f"Repeat {repeat + 1}/5")
        print(f"{'='*50}")

        repeat_seed = 42 + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        t_vade_start = time.time()

        dataset    = TensorDataset(X, torch.tensor(true_labels, dtype=torch.int64))
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        pretrain_path = os.path.join(out_dir, f"vade_pretrain_repeat{repeat}.pk")
        if os.path.exists(pretrain_path):
            os.remove(pretrain_path)

        vade = VaDE(args).to(device)
        optimizer = torch.optim.Adam(vade.parameters(), lr=2e-4)

        vade.pre_train(dataloader, pretrain_path, pre_epoch=30)
        os.remove(pretrain_path)  # temporary file only

        t_pretrain_end = time.time()
        pretrain_time = time.time() - t_vade_start
        print(f"Pretraining time: {pretrain_time:.2f} seconds")

        # --- Main training ---
        for epoch in range(200):
            vade.train()
            total_loss = 0.0
            for x_batch, _ in dataloader:
                optimizer.zero_grad()
                loss = vade.ELBO_Loss(x_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vade.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            if epoch % 10 == 0:
                vade.eval()
                with torch.no_grad():
                    temp_pred = vade.predict(X)
                unique, counts = np.unique(temp_pred, return_counts=True)
                print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                      f"Clusters used: {len(unique)} | Sizes: {counts}")
                vade.train()

        # --- Predict & evaluate ---
        vade.eval()
        with torch.no_grad():
            pred    = vade.predict(X)
            latents = vade.get_latent(X)

        train_time = time.time() - t_pretrain_end
        total_time = time.time() - t_vade_start
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Total time (pretrain + train): {total_time:.2f} seconds")

        if avg_loss < best_info["loss"]:
            best_info["loss"]     = avg_loss
            best_info["latent"]   = latents
            best_info["clusters"] = pred
            print(f"  -> New best repeat (loss={avg_loss:.4f})")

        ari = adjusted_rand_score(true_labels, pred)
        nmi = normalized_mutual_info_score(true_labels, pred)
        acc = clustering_accuracy(true_labels, pred)

        print(f"\nRepeat {repeat + 1} Results | ARI: {ari:.4f} | NMI: {nmi:.4f} | ACC: {acc:.4f}")

        summary_rows.append({
            'model':         'VaDE',
            'repeat':        repeat + 1,
            'K_true':        K_true,
            'K_final':       len(np.unique(pred)),
            'ari':           ari,
            'nmi':           nmi,
            'acc':           acc,
            'final_loss':    avg_loss,
            'pretrain_time': pretrain_time,
            'train_time':    train_time,
            'total_time':    total_time
        })

    # ============================================================
    # Save
    # ============================================================
    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(out_dir, "vade_summary.csv")
    summary_df.to_csv(out_path, index=False)

    pd.DataFrame(best_info["latent"], index=cell_ids).to_csv(
        os.path.join(out_dir, "vade_best_latents.csv")
    )
    pd.DataFrame({
        'cell_id': cell_ids,
        'cluster': best_info["clusters"]
    }).to_csv(os.path.join(out_dir, "vade_best_clusters.csv"), index=False)

    print(f"\n{'='*50}")
    print("Summary across repeats:")
    print(summary_df[['repeat', 'ari', 'nmi', 'acc', 'final_loss']].to_string(index=False))
    print(f"\nMean  | ARI: {summary_df.ari.mean():.4f} | NMI: {summary_df.nmi.mean():.4f} | ACC: {summary_df.acc.mean():.4f}")
    print(f"Std   | ARI: {summary_df.ari.std():.4f}  | NMI: {summary_df.nmi.std():.4f}  | ACC: {summary_df.acc.std():.4f}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()