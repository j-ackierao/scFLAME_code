"""
run_merge.py
============
Fit scFLAME to a real scRNA-seq dataset and perform greedy cluster merging, run down to 
--k-final. Designed for datasets where the "true" number of subtypes is uncertain and 
over-clustering + merging is preferred to fitting K directly, or a hierarchical view of 
the data is desired (see the accompanying preprint).

Expected input files under --data-dir:
    counts.csv            cells x genes raw count matrix, first column = cell ID
    clusters.csv           columns "cell_id", "celltype" (integer-coded ground truth)
    dispersion.csv         columns "gene", "dispersion" (e.g. edgeR tagwise estimates)
    library_sizes.csv      columns "cell_id", "tmm_lib_size" (e.g. TMM size factors)

Usage:
    python scripts/run_merge.py --data-dir data/baron --out-dir results/baron \\
        --k-init 10 15 --k-final 2 --criterion icl
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scflame import DEVICE, train_nb_fa, train_scflame, make_eii_gmm_init
from scflame.utils import load_dataset, top_marker_genes_table, print_top_marker_genes
from scflame.merging import greedy_merge_full, m_step_cluster_params_full
from scflame.viz import plot_tsne_merge_path


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    if args.save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    X, gene_names, c_true_np, dispersions, C = load_dataset(
        args.data_dir, args.n_hvgs, sanitize_gene_names= args.sanitize_gene_names,
    )
    N, D = X.shape
    K_true = len(np.unique(c_true_np))
    print(f"N={N}, D={D}, true clusters={K_true}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Fitting NBFA...")

    nbfa_result = train_nb_fa(
        X, q=args.latent_dim, C=C,
        mc_samples=args.mc_samples, epochs=args.nbfa_epochs,
        lr=1e-2, disp=torch.tensor(dispersions, dtype=torch.float32), verbose=args.verbose,
    )
    if args.save_checkpoints:
        torch.save(nbfa_result, os.path.join(checkpoint_dir, f"nbfa_result_task_{args.seed}.pt"))
    print("NBFA done.")

    history_rows = []
    alloc_cols = {}

    for K_init in args.k_init:
        print(f"\n{'=' * 65}\nK_init={K_init} (K_true={K_true})\n{'=' * 65}")

        gmm_init = make_eii_gmm_init(nbfa_result, K_init, DEVICE, seed=args.seed)

        initial_result = train_scflame(
            X, nbfa_result, gmm_init, K_init,
            epochs=args.scflame_epochs, mc_samples=args.mc_samples,
            lr_latent=1e-3, latent_steps=3,
            temp_annealing=True, max_temp=1.5,
            L_steps=3, L_reg=0.01, verbose=args.verbose,
        )
        if args.save_checkpoints:
            torch.save(initial_result,
                        os.path.join(checkpoint_dir, f"initial_result_task_{args.seed}_Kinit_{K_init}.pt"))

        if args.print_top_genes or args.save_top_genes:
            gene_table = top_marker_genes_table(
                final_result["L"], final_result["nu_k"], final_result["pi_k"],
                gene_names, top_n=args.top_n_genes, upregulated_only=args.upregulated_only,
            )
            if args.print_top_genes:
                print(f"\n--- Top {args.top_n_genes} genes per cluster ---")
                print_top_marker_genes(gene_table)
            if args.save_top_genes:
                gene_table_path = os.path.join(out_dir, f"top_genes_Kinit{K_init}.csv")
                gene_table.to_csv(gene_table_path, index=False)
                print(f"Saved: {gene_table_path}")

        c_pred_init = initial_result["r"].argmax(dim=1).cpu().numpy()
        print(f"  ARI after scFLAME init: {adjusted_rand_score(c_true_np, c_pred_init):.3f}")

        # Switch from the shared-diagonal covariance used during fitting to a full covariance per cluster
        m_init, log_s_init, r_init = initial_result["m"], initial_result["log_s"], initial_result["r"]
        nu_k_init, Sigma_k_init, pi_k_init = m_step_cluster_params_full(
            m_init, torch.exp(log_s_init), r_init,
        )
        merge_input = {
            "r": r_init, "m": m_init, "log_s": log_s_init,
            "nu_k": nu_k_init, "Sigma_k": Sigma_k_init, "pi_k": pi_k_init,
        }

        final_result, K_reached, history, allocations = greedy_merge_full(
            X, merge_input,
            top_m=args.top_m, verbose=True,
            c_true=c_true_np, K_true=K_true, criterion=args.criterion,
            K_final=args.k_final,
            checkpoint_dir=checkpoint_dir,
            checkpoint_tag=f"task_{args.seed}_Kinit_{K_init}",
        )

        plot_tsne_merge_path(
            m_np=m_init.cpu().numpy(), allocations=allocations,
            K_true=K_true, K_init=K_init, c_true_np=c_true_np,
            out_path=os.path.join(out_dir, f"tsne_Kinit{K_init}_{args.seed}.png"),
        )

        m_np = m_init.cpu().numpy()
        for K_step, labels in allocations.items():
            alloc_cols[f"Kinit{K_init}_K{K_step}"] = labels

        ari_final = adjusted_rand_score(c_true_np, final_result["r"].argmax(dim=1).cpu().numpy())
        print(f"  After merge: ARI={ari_final:.3f} | K={K_reached} (true={K_true})")

        for step in history:
            history_rows.append({"K_init": K_init, "ari_init": adjusted_rand_score(c_true_np, c_pred_init), **step})

    history_df = pd.DataFrame(history_rows)

    history_path = os.path.join(out_dir, f"merge_history_{args.seed}.csv")
    history_df.to_csv(history_path, index=False)
    print(f"\nSaved: {history_path}")

    alloc_df = pd.DataFrame(alloc_cols)
    alloc_df.index.name = "cell"
    alloc_path = os.path.join(out_dir, f"cluster_allocations_{args.seed}.csv")
    alloc_df.to_csv(alloc_path)
    print(f"Saved: {alloc_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NBFA -> scFLAME -> greedy cluster merge pipeline.")
    p.add_argument("--data-dir", required=True,
                    help="Directory containing counts.csv, clusters.csv, dispersion.csv, library_sizes.csv")
    p.add_argument("--out-dir", required=True, help="Directory to write results and checkpoints to")
    p.add_argument("--n-hvgs", type=int, default=5000, help="Number of highly variable genes to keep")
    p.add_argument("--latent-dim", type=int, default=10, help="Latent factor dimensionality q")
    p.add_argument("--k-init", type=int, nargs="+", default=[10, 15],
                    help="One or more over-clustering starting points for the merge procedure")
    p.add_argument("--k-final", type=int, default=2, help="Stop merging once K reaches this value")
    p.add_argument("--criterion", default="icl", choices=["bic", "icl", "dbi", "bhattacharyya"],
                    help="Model-selection criterion driving which pair to merge at each step")
    p.add_argument("--top-m", type=int, default=None,
                    help="Pre-screen only the M closest cluster pairs by Mahalanobis distance per "
                         "step (default: evaluate all)")
    p.add_argument("--nbfa-epochs", type=int, default=500, help="Epochs for the NBFA warm-start stage")
    p.add_argument("--scflame-epochs", type=int, default=300, help="Epochs for the joint scFLAME fit")
    p.add_argument("--mc-samples", type=int, default=10, help="Monte Carlo samples per ELBO evaluation")
    p.add_argument("--sanitize-gene-names", action="store_true",
                    help="Perform '-'/'+' -> '.' gene-name normalisation when matching dispersion.csv")
    p.add_argument("--print-top-genes", action="store_true", help="Print top marker genes per cluster for initial K")
    p.add_argument("--save-top-genes", action="store_true", help="Save top marker genes per cluster to CSV for initial K")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    p.add_argument("--save-checkpoints", action="store_true",
                help="Save nbfa_result/scflame_result .pt checkpoints per repeat under out-dir/checkpoints")
    p.add_argument("--verbose", action="store_true", help="Print per-epoch training progress")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
