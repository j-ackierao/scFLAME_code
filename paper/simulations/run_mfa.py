"""
run_mfa.py
Fits our implementation of MFA on one simulated dataset's ground-truth labels.
(Data simulated from Splat or ZINBWaVE)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scflame import clustering_accuracy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comparison_models.mfa import *

def load_simulated_dataset(counts_path: str, clusters_path: str,
                            dispersions_path: str = None, libsize_path: str = None) -> dict:
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

    if dispersions_path is not None:
        disp_df = pd.read_csv(dispersions_path)
        result["dispersions"] = disp_df.set_index("gene").loc[df.columns]["dispersion"].to_numpy()
 
    if libsize_path is not None:
        lib_df = pd.read_csv(libsize_path)
        result["C"] = lib_df.set_index("cell_id")["tmm_lib_size"].to_numpy()
 
    return result


def parse_args():
    p = argparse.ArgumentParser(description="MFA benchmark on a simulated dataset")
    p.add_argument("--counts_path", required=True)
    p.add_argument("--clusters_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--replicate", type=int, default=None,
                    help="Defaults to $SLURM_ARRAY_TASK_ID - 1 if unset")
    p.add_argument("--device", default="cuda")
    p.add_argument("--scenario", default=None,
                    help="Optional scenario tag (e.g. Easy/Medium/Hard), added to the output filename and summary")
    return p.parse_args()

def main():
    args = parse_args()
    task_id = args.replicate if args.replicate is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) - 1
    device = args.device

    torch.manual_seed(42 + task_id)
    np.random.seed(42 + task_id)

    os.makedirs(args.out_dir, exist_ok=True)

    dat = load_simulated_dataset(
        args.counts_path, args.clusters_path, n_hvgs=args.n_hvgs,
        dispersions_path=args.dispersions_path, libsize_path=args.libsize_path
    )
    X_np, c_true, K_true = dat["X_np"], dat["c_true"], dat["K_true"]
    print(f"N={X_np.shape[0]}, D={X_np.shape[1]}, K_true={K_true}")

    X = torch.tensor(X_np, dtype=torch.float32, device=device)

    summary_rows = []

    for repeat in range(3):
        print(f"Starting Repeat {repeat + 1}/3...")
        repeat_seed = (task_id * 100) + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        mfa_result = train_mixture_fa_closed_form(
            X, K=K_true, q=10, epochs=100
        )
        r = mfa_result['r']
        final_elbo = mfa_result['trace'][-1]

        c_pred = torch.argmax(r, dim=1).cpu().numpy()

        ari = adjusted_rand_score(c_true, c_pred)
        nmi = normalized_mutual_info_score(c_true, c_pred)
        acc = clustering_accuracy(c_true, c_pred)

        summary_rows.append({
            'repeat':       repeat,
            'method':       'mfa',
            'scenario':     args.scenario,
            'K_true':       K_true,
            'K_final':      len(np.unique(c_pred)),
            'ari':          ari,
            'nmi':          nmi,
            'acc':          acc,
            'final_elbo':   final_elbo,
        })


    summary_df = pd.DataFrame(summary_rows)
    if args.scenario is not None:
        out_path = os.path.join(args.out_dir, f"mfa_{args.scenario}_{task_id}_summary.csv")
    else:
        out_path = os.path.join(args.out_dir, f"mfa_{task_id}_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()