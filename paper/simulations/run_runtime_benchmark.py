## ============================================================
## run_runtime_benchmark.py
## Trains scFLAME on one pre-generated Splat dataset
## and records fit time + clustering ARI. Meant to be called once per
## (N, replicate) pair.
##
## The simulated dataset itself (raw counts / library sizes / cluster
## labels, from generate_splat_scaling_data.R) is deleted after being
## loaded unless --keep_sim_data is passed.
## We ran this via SLURM arrays with job numbers 1-20. 
## ============================================================

import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.io
import torch
from sklearn.metrics import adjusted_rand_score

from scflame import EII_GMM, train_nb_fa, train_scflame

def parse_args():
    p = argparse.ArgumentParser(description="scFLAME runtime/ARI scaling benchmark")
    p.add_argument("--n_cells", type=int, required=True,
                    help="Number of cells in the dataset to benchmark (must match a file generate_splat_scaling_data.R already produced)")
    p.add_argument("--sim_dir", type=str, default="sim_data/run_times",
                    help="Where generate_splat_scaling_data.R wrote n{N}_{rep}_*.{mtx,csv} [default: %(default)s]")
    p.add_argument("--results_dir", type=str, default="results/run_times",
                    help="Where per-run timing/ARI summaries are written [default: %(default)s]")
    p.add_argument("--batch_size", type=int, default=8192,
                    help="Fixed across every N in the sweep [default: %(default)s]")
    p.add_argument("--replicate", type=int, default=None,
                    help="Defaults to $SLURM_ARRAY_TASK_ID - 1 if unset")
    p.add_argument("--keep_sim_data", action="store_true",
                    help="Keep the simulated dataset after loading, instead of deleting it")
    return p.parse_args()


def load_sim(prefix: str, delete_after: bool = True):
    mtx_path = f"{prefix}_counts.mtx"
    lib_path = f"{prefix}_libsize.csv"
    clu_path = f"{prefix}_clusters.csv"

    X_sparse = scipy.io.mmread(mtx_path).tocsr().T.tocsr()  # -> cells x genes
    lib_df = pd.read_csv(lib_path)
    lib_size = lib_df["raw_lib_size"].to_numpy()
    clusters_df = pd.read_csv(clu_path)
    labels = clusters_df["celltype"].to_numpy()

    if delete_after:
        for p in (mtx_path, lib_path, clu_path):
            os.remove(p)

    return X_sparse, lib_size, labels


def main():
    args = parse_args()
    rep = args.replicate if args.replicate is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) - 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prefix = f"{args.sim_dir}/n{args.n_cells}_{rep}"
    X_sparse, lib_size, labels = load_sim(prefix, delete_after=not args.keep_sim_data)

    X = torch.tensor(X_sparse.toarray(), dtype=torch.float32, device=device)
    C = torch.tensor(lib_size, dtype=torch.float32, device=device)

    torch.manual_seed(rep)
    np.random.seed(rep)

    t0 = time.perf_counter()
    nbfa_result = train_nb_fa(
        X, C=C, q=15, mc_samples=5, epochs=500,
        lr=1e-2, batch_size=8192, verbose=False,
    )
    nbfa_time = time.perf_counter() - t0

    Z = nbfa_result["m"].detach().cpu().numpy()
    K = len(np.unique(labels))
    gmm = EII_GMM(n_components=K, n_init=10, random_state=rep)
    gmm.fit(Z)

    t1 = time.perf_counter()
    final_result = train_scflame(
        X, nbfa_result, gmm, K=K,
        epochs=300, mc_samples=5,
        batch_size=8192, verbose=False,
    )
    joint_time = time.perf_counter() - t1

    c_pred = final_result["r"].argmax(dim=1).cpu().numpy()
    ari = adjusted_rand_score(labels, c_pred)

    del X, X_sparse
    if device == "cuda":
        torch.cuda.empty_cache()

    row = {
        "N": args.n_cells,
        "rep": rep,
        "batch_size": args.batch_size,
        "nbfa_time": nbfa_time,
        "joint_time": joint_time,
        "total_time": nbfa_time + joint_time,
        "ari": ari,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = f"{args.results_dir}/n{args.n_cells}_rep{rep}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)

    print(f"N={args.n_cells} rep={rep} total_time={row['total_time']:.2f}s ARI={ari:.3f} -> {out_path}")


if __name__ == "__main__":
    main()