"""
Gene scoring for scFLAME.

Computes cluster-specific gene scores s_k = L (nu_k - nu_bar) and saves
ranked gene lists for downstream analysis (e.g. fgsea input).

Shared between the gene-scoring and merging analyses - call with
different --init_checkpoint / --merge_checkpoint / --out_dir for each run.

--init_checkpoint is the original run's checkpoint (K = Kinit) - 
this is where the loading matrix L is read from, since L is shared/fixed 
across merge stages. --merge_checkpoint must be the checkpoint for 
whichever resolution you're currently scoring (e.g. the K=9 merge stage) 
- this is where nu_k/pi_k (the CURRENT, post-merge cluster parameters) 
are read from. These two can be pointed at the same file if no merging
is performed, or you are inspecting scores at K = Kinit.

Outputs
-------
gene_scores_top{N}_{tag}.csv   top-N genes per cluster
gene_ranked_{tag}.csv          full ranked list (fgsea input)
cluster_{a}_vs_{b}_top{N}_c{a}.csv / cluster_{b}_vs_{a}_top{N}_c{b}.csv
    (only written if --pairwise_a/--pairwise_b are given)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch


# ── scoring functions ───────────────────────────────────────────────────────

def gene_scores(L: np.ndarray, nu_k: np.ndarray, pi_k: np.ndarray, cluster_k: int) -> np.ndarray:
    """Calculate gene scores."""
    nu_bar = (pi_k[:, None] * nu_k).sum(0)
    raw_scores = L @ (nu_k[cluster_k] - nu_bar)
    scale_factor = np.abs(L).sum(axis=1) + 1e-5
    return raw_scores / scale_factor


def pairwise_gene_scores(L: np.ndarray, nu_k: np.ndarray, k: int, j: int, scale: bool = True) -> np.ndarray:
    """Pairwise log-fold-change-style score between cluster k and cluster j: s_kj = L(nu_k - nu_j)."""
    raw_scores = L @ (nu_k[k] - nu_k[j])
    if scale:
        scale_factor = np.abs(L).sum(axis=1) + 1e-5
        return raw_scores / scale_factor
    return raw_scores


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NBMCFA gene scoring")
    p.add_argument("--init_checkpoint", required=True,
                    help="Original run's checkpoint (K=Kinit) - source of L")
    p.add_argument("--merge_checkpoint", required=True,
                    help="Checkpoint for the current merge stage/K being scored - source of nu_k, pi_k")
    p.add_argument("--csv_path", required=True, help="Raw/HVG counts CSV whose columns match L's gene order")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", default=None,
                    help="Label embedded in output filenames (e.g. 'Kinit15_K9'); defaults to init/merge K inferred from the checkpoints")
    p.add_argument("--rank_by", choices=["abs", "up"], default="up",
                    help="'abs' ranks by |score| (both up- and down-regulated genes); 'up' ranks upregulated genes only")
    p.add_argument("--top_n", type=int, default=100)
    p.add_argument("--exclude_prefixes", nargs="*", default=["RPL", "MT-", "LINC", "CTD-"],
                    help="Gene-name prefixes (case-insensitive) excluded before scoring, e.g. ribosomal/mitochondrial/non-coding")
    p.add_argument("--pairwise_a", type=int, default=None, help="First cluster index for the optional pairwise comparison")
    p.add_argument("--pairwise_b", type=int, default=None, help="Second cluster index for the optional pairwise comparison")
    p.add_argument("--pairwise_top_n", type=int, default=100)
    p.add_argument("--pairwise_scale", action="store_true", default=True)
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    init_ckpt = torch.load(args.init_checkpoint, map_location="cpu")
    merge_ckpt = torch.load(args.merge_checkpoint, map_location="cpu")

    for key in ("L",):
        if key not in init_ckpt:
            raise KeyError(f"--init_checkpoint ({args.init_checkpoint}) has no '{key}' - is this really the original Kinit checkpoint?")
    for key in ("nu_k", "pi_k"):
        if key not in merge_ckpt:
            raise KeyError(f"--merge_checkpoint ({args.merge_checkpoint}) has no '{key}' - is this really a merge-stage checkpoint?")

    L = init_ckpt["L"].float().numpy()
    nu_k = merge_ckpt["nu_k"].float().numpy()
    pi_k = merge_ckpt["pi_k"].float().numpy()
    K, G = nu_k.shape[0], L.shape[0]

    tag = args.tag or f"Kinit{L.shape[1]}_K{K}"

    df = pd.read_csv(args.csv_path, index_col=0)
    genes_all = list(df.columns)

    if len(genes_all) != G:
        raise ValueError(
            f"Dimension mismatch! L has {G} rows but the CSV has {len(genes_all)} columns. "
            f"Pass the exact counts file that produced this checkpoint's gene ordering."
        )
    print(f"Loaded: G={G}, R={L.shape[1]}, K={K}")

    exclude = tuple(p.upper() for p in args.exclude_prefixes)
    keep_indices = [i for i, gene in enumerate(genes_all) if not gene.upper().startswith(exclude)]
    genes = [genes_all[i] for i in keep_indices]
    L = L[keep_indices, :]
    print(f"Filtered out {G - len(genes)} genes matching {exclude}. New G={L.shape[0]}")

    score_rows, ranked_rows = [], []
    for k in range(K):
        scores = gene_scores(L, nu_k, pi_k, cluster_k=k)
        order = np.argsort(-scores) if args.rank_by == "up" else np.argsort(-np.abs(scores))

        for rank, i in enumerate(order[:args.top_n]):
            score_rows.append(dict(cluster_k=k, gene=genes[i], score=scores[i], rank=rank + 1))
        for rank, i in enumerate(order):
            ranked_rows.append(dict(cluster_k=k, gene=genes[i], score=scores[i], rank=rank + 1))

    df_scores = pd.DataFrame(score_rows)
    df_ranked = pd.DataFrame(ranked_rows)
    df_scores.to_csv(os.path.join(args.out_dir, f"gene_scores_top{args.top_n}_{tag}.csv"), index=False)
    df_ranked.to_csv(os.path.join(args.out_dir, f"gene_ranked_{tag}.csv"), index=False)
    print(f"Saved {len(df_scores):,} rows -> gene_scores_top{args.top_n}_{tag}.csv")
    print(f"Saved {len(df_ranked):,} rows -> gene_ranked_{tag}.csv")

    if args.pairwise_a is not None and args.pairwise_b is not None:
        a, b = args.pairwise_a, args.pairwise_b
        if a >= K or b >= K:
            raise ValueError(f"--pairwise_a/--pairwise_b must be < K={K}")

        print(f"Computing pairwise scores between cluster {a} and cluster {b}...")
        pairwise_scores = pairwise_gene_scores(L, nu_k, k=a, j=b, scale=args.pairwise_scale)

        order_a = np.argsort(-pairwise_scores)
        top_a = [dict(gene=genes[i], score=pairwise_scores[i], rank=r + 1)
                 for r, i in enumerate(order_a[:args.pairwise_top_n])]

        order_b = np.argsort(pairwise_scores)
        top_b = [dict(gene=genes[i], score=-pairwise_scores[i], rank=r + 1)
                 for r, i in enumerate(order_b[:args.pairwise_top_n])]

        pd.DataFrame(top_a).to_csv(os.path.join(args.out_dir, f"cluster_{a}_vs_{b}_top{args.pairwise_top_n}_c{a}.csv"), index=False)
        pd.DataFrame(top_b).to_csv(os.path.join(args.out_dir, f"cluster_{b}_vs_{a}_top{args.pairwise_top_n}_c{b}.csv"), index=False)
        print(f"Saved top {args.pairwise_top_n} for cluster {a} vs {b} and {b} vs {a}")


if __name__ == "__main__":
    run(parse_args())