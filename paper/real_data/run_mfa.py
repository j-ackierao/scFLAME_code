"""
run_mfa.py
Fits our implementation of MFA on one real dataset.
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


from load_data import (
    load_uterus,
    load_zeisel,
    load_tm_spleen,
    load_segerstolpe,
    load_baron,
    load_pbmc68k,
)

LOADERS = {
    "uterus": load_uterus,
    "zeisel": load_zeisel,
    "tm_spleen": load_tm_spleen,
    "segerstolpe": load_segerstolpe,
    "baron": load_baron,
    "pbmc68k": load_pbmc68k,
}

def parse_args():
    p = argparse.ArgumentParser(description="MFA benchmark on a real dataset")
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
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    os.makedirs(args.out_dir, exist_ok=True)

    loader = LOADERS[args.dataset]

    dat = loader(counts_path=args.counts_path, clusters_path=args.clusters_path, normalize=True)
    X_np, c_true, K_true = dat["X_np"], dat["c_true"], dat["K_true"]

    print(f"N={X_np.shape[0]}, D={X_np.shape[1]}, K_true={K_true}")

    X = torch.tensor(X_np, dtype=torch.float32, device=device)

    summary_rows = []

    best_elbo    = -np.inf
    best_result  = None
    best_pred    = None

    for repeat in range(5):
        print(f"Starting Repeat {repeat + 1}/5...")
        repeat_seed = 21 + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        mfa_result = train_mixture_fa_closed_form(
            X, K=K_true, q=10, epochs=200
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
            'K_true':       K_true,
            'K_final':      len(np.unique(c_pred)),
            'ari':          ari,
            'nmi':          nmi,
            'acc':          acc,
            'final_elbo':   final_elbo,
        })

        if final_elbo > best_elbo:
            best_elbo   = final_elbo
            best_result = mfa_result
            best_pred   = c_pred


    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(args.out_dir, f"scflame_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Save best-run latents and cluster labels
    r_best = best_result['r']          # N x K  (responsibilities)
    m_best = best_result['m']          # N x K x q  (posterior means)

    # Responsibility-weighted latent: sum_k r_k * m_k  →  N x q
    r_np   = r_best.detach().cpu().numpy()           # N x K
    m_np   = m_best.detach().cpu().numpy()           # N x K x q
    # r_np[..., None] broadcasts to N x K x 1, then sum over K
    latent_2d_np = (r_np[..., None] * m_np).sum(axis=1)   # N x q

    latent_cols = [f"latent_{i}" for i in range(latent_2d_np.shape[1])]
    df_latents = pd.DataFrame(latent_2d_np, columns=latent_cols)
    df_latents.to_csv(args.out_dir, f"mfa_best_latents.csv", index=False)

    df_labels = pd.DataFrame(best_pred, columns=['label'])
    df_labels.to_csv(args.out_dir, f"mfa_best_labels.csv", index=False)


if __name__ == "__main__":
    main()