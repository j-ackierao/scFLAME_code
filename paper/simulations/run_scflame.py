"""
benchmark_scflame.py
Fits scFLAME under three initialization strategies, against one simulated dataset's ground-truth labels.
(Data simulated from Splat or ZINBWaVE)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scflame import (
    DEVICE, train_nb_fa, train_scflame, make_eii_gmm_init, make_random_init,
    make_random_far_init, clustering_accuracy
)

INIT_TYPES = {
    "eii_gmm": make_eii_gmm_init,
    "random": make_random_init,
    "random_far": make_random_far_init,
}
LABELS = {"eii_gmm": "NBFA+EII GMM", "random": "Random", "random_far": "Random far"}

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
    p = argparse.ArgumentParser(description="scFLAME init-strategy benchmark on a simulated dataset")
    p.add_argument("--counts_path", required=True)
    p.add_argument("--clusters_path", required=True)
    p.add_argument("--dispersions_path", required=True)
    p.add_argument("--libsize_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--replicate", type=int, default=None,
                    help="Defaults to $SLURM_ARRAY_TASK_ID - 1 if unset")
    p.add_argument("--device", default="cuda")
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
    dispersions, C = dat["dispersions"], dat["C"]
    print(f"N={X_np.shape[0]}, D={X_np.shape[1]}, K_true={K_true}")

    X = torch.tensor(X_np, dtype=torch.float32, device=device)

    summary_rows = []
    tsne_data = []

    for repeat in range(3):
        print(f"Starting Repeat {repeat + 1}/3...")
        repeat_seed = (task_id * 100) + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        nbfa_result = train_nb_fa(
            X, q=10, C=C, mc_samples=10, epochs=350,
            lr=1e-2, disp=dispersions, verbose=False,
        )

        for init_type, init_fn in INIT_TYPES.items():
            print(f"  [R{repeat + 1}] Init: {LABELS[init_type]}")
            K_init = K_true

            gmm_init = init_fn(nbfa_result, K_init, device, seed=repeat_seed + 500)

            final_result = train_scflame(
                X, nbfa_result, gmm_init, K_init,
                epochs=250, mc_samples=10,
                lr_latent=1e-3, latent_steps=3,
                temp_annealing=True, max_temp=1.5,
                L_steps=3, verbose=True)

            c_pred_final = final_result["r"].argmax(dim=1).cpu().numpy()
            print(np.exp(final_result["log_phi"].cpu().numpy()))

            summary_rows.append({
                "repeat": repeat,
                "K_true": K_true,
                "K_final": len(np.unique(c_pred_final)),
                "init_type": init_type,
                "ari_final": adjusted_rand_score(c_true, c_pred_final),
                "nmi_final": normalized_mutual_info_score(c_true, c_pred_final),
                "acc_final": clustering_accuracy(c_true, c_pred_final),
                "elbo_final": final_result["trace"]["elbo"][-1],
            })

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(args.out_dir, f"scflame_{task_id}_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if tsne_data:
        tsne_path = os.path.join(args.out_dir, f"scflame_{task_id}_tsne_data.npz")
        np.savez(tsne_path, data=np.array(tsne_data, dtype=object), allow_pickle=True)
        print(f"Saved: {tsne_path}")


if __name__ == "__main__":
    main()