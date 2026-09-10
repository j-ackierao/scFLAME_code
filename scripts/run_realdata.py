"""
run_realdata.py
================
Fit scFLAME to a real scRNA-seq dataset and report clustering performance
against ground-truth cell-type labels. Saves allocation table, summary metrics, 
and t-SNE plot of latent factors. Optionally saves top marker genes per cluster.

Expected input files under --data-dir:
    counts.csv           cells x genes raw count matrix, first column = cell ID
    clusters.csv         one column "celltype", integer-coded ground-truth labels. 
                        Optional second column "batch_id" for batch correction.
    dispersion.csv       columns "gene", "dispersion" (e.g. edgeR estimates) (optional)
    library_sizes.csv    columns "cell_id", "tmm_lib_size" (e.g. TMM size factors) (optional)

Usage:
    python scripts/run_realdata.py --data-dir data/segerstolpe --out-dir results/segerstolpe --top-n-genes 10 --upregulated-only --print-top-genes
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scflame import (
    DEVICE, train_nb_fa, train_scflame, make_eii_gmm_init,
    clustering_accuracy, get_top_genes_per_cluster, plot_tsne,
)
from scflame.utils import load_dataset, print_top_marker_genes, top_marker_genes_table


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    if args.save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, gene_names, c_true_np, dispersions, C, batch_true = load_dataset(args.data_dir, args.n_hvgs)
    N, D = X.shape
    K_true = len(np.unique(c_true_np))
    print(f"N={N}, D={D}, true clusters={K_true}")

    summary_rows = []
    alloc_cols = {"true_labels": c_true_np}
    gene_table_rows = []
    tsne_data = None

    for repeat in range(args.n_repeats):
        print(f"\nStarting repeat {repeat + 1}/{args.n_repeats}...")
        repeat_seed = args.seed + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        nbfa_result = train_nb_fa(
            X, q=args.latent_dim, C=C, batch=batch_true,
            mc_samples=args.mc_samples, epochs=args.nbfa_epochs,
            lr=1e-2, disp=dispersions, verbose=args.verbose,
        )
        if args.save_checkpoints:
            torch.save(nbfa_result, os.path.join(checkpoint_dir, f"nbfa_result_rep{repeat + 1}.pt"))

        gmm_init = make_eii_gmm_init(nbfa_result, K_true, DEVICE, seed=repeat_seed + 500)

        final_result = train_scflame(
            X, nbfa_result, gmm_init, K_true,
            epochs=args.scflame_epochs, mc_samples=args.mc_samples,
            lr_latent=1e-3, latent_steps=3,
            temp_annealing=True, max_temp=1.5,
            L_steps=3, verbose=args.verbose,
        )
        if args.save_checkpoints:
            torch.save(final_result, os.path.join(checkpoint_dir, f"scflame_result_rep{repeat + 1}.pt"))

        c_pred = final_result["r"].argmax(dim=1).cpu().numpy()
        alloc_cols[f"rep{repeat + 1}_pred"] = c_pred

        summary_rows.append({
            "repeat": repeat,
            "K_true": K_true,
            "K_final": len(np.unique(c_pred)),
            "ari_final": adjusted_rand_score(c_true_np, c_pred),
            "nmi_final": normalized_mutual_info_score(c_true_np, c_pred),
            "acc_final": clustering_accuracy(c_true_np, c_pred),
            "elbo_final": final_result["trace"]["elbo"][-1],
        })

        if args.print_top_genes or args.save_top_genes:
            gene_table = top_marker_genes_table(
                final_result["L"], final_result["nu_k"], final_result["pi_k"],
                gene_names, top_n=args.top_n_genes, upregulated_only=args.upregulated_only,
            )
            if args.print_top_genes:
                print(f"\n--- Top {args.top_n_genes} genes per cluster (repeat {repeat + 1}) ---")
                print_top_marker_genes(gene_table)
            if args.save_top_genes:
                gene_table.insert(0, "repeat", repeat + 1)
                gene_table_rows.append(gene_table)

        if repeat == 0:
            tsne_data = dict(
                latents=final_result["m"].detach().cpu().numpy(),
                true_labels=c_true_np, pred_labels=c_pred, K_true=K_true,
            )

    summary_path = os.path.join(args.out_dir, "scflame_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    alloc_path = os.path.join(args.out_dir, "scflame_allocations.csv")
    pd.DataFrame(alloc_cols).to_csv(alloc_path, index=False)
    print(f"Saved: {alloc_path}")

    if gene_table_rows:
        genes_path = os.path.join(args.out_dir, "scflame_top_genes.csv")
        pd.concat(gene_table_rows, ignore_index=True).to_csv(genes_path, index=False)
        print(f"Saved: {genes_path}")

    if tsne_data is not None:
        print("\nGenerating t-SNE plot...")
        tsne_path = os.path.join(args.out_dir, "tsne.png")
        plot_tsne(save_path=tsne_path, seed=args.seed, **tsne_data)
        print(f"Saved: {tsne_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit scFLAME to a real scRNA-seq dataset.")
    p.add_argument("--data-dir", required=True,
                    help="Directory containing counts.csv, clusters.csv, dispersion.csv, library_sizes.csv")
    p.add_argument("--out-dir", required=True, help="Directory to write results to")
    p.add_argument("--n-hvgs", type=int, default=3000, help="Number of highly variable genes to keep")
    p.add_argument("--latent-dim", type=int, default=10, help="Latent factor dimensionality q")
    p.add_argument("--n-repeats", type=int, default=3, help="Number of independent fits")
    p.add_argument("--nbfa-epochs", type=int, default=500, help="Epochs for the NBFA warm-up stage")
    p.add_argument("--scflame-epochs", type=int, default=300, help="Epochs for the training of scFLAME")
    p.add_argument("--mc-samples", type=int, default=10, help="Monte Carlo samples per ELBO evaluation")
    p.add_argument("--seed", type=int, default=100, help="Base random seed (offset by repeat index)")
    p.add_argument("--sanitize-gene-names", action="store_true",
                    help="Normalise '-'/'+' -> '.' in dispersion.csv's gene column before matching")
    p.add_argument("--print-top-genes", action="store_true", help="Print top marker genes per cluster")
    p.add_argument("--save-top-genes", action="store_true", help="Save top marker genes per cluster to CSV")
    p.add_argument("--top-n-genes", type=int, default=10, help="Number of top genes to display/save per cluster")
    p.add_argument("--upregulated-only", action="store_true", help="Only consider upregulated genes")
    p.add_argument("--save-checkpoints", action="store_true",
                help="Save nbfa_result/scflame_result .pt checkpoints per repeat under out-dir/checkpoints")
    p.add_argument("--verbose", action="store_true", help="Print per-epoch training progress")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
