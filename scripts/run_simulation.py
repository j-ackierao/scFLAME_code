"""
run_simulation.py
==================
Fit scFLAME on simulated data from the scFLAME generative model, comparing cluster-initialisation 
strategies across a set of difficulty scenarios.

Simulation scenarios are defined by the DATASETS constant below -- edit it
directly to add/change scenarios; it isn't exposed as a CLI argument since
each entry is a small research-design choice.

This script generates one dataset per scenario, and runs multiple independent repeats per dataset.
When run under Slurm, the array task ID is used as the random seed offset to generate the dataset.

Usage:
    python scripts/run_simulation.py --out-dir results/simulation
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
    DEVICE, EII_GMM, train_nb_fa, train_scflame, scflame_synthetic_separated,
    make_eii_gmm_init, make_random_init, make_random_far_init,
    clustering_accuracy, plot_tsne,
)

# Edit this list to add/change simulation scenarios.
DATASETS = [
    dict(scenario="Easy", N=800, D=1000, K=4, q=8, r_true=2.5, cluster_separation=2.5, seed=50),
    dict(scenario="Hard", N=1500, D=2000, K=8, q=10, r_true=0.5, cluster_separation=1.5, seed=999),
]

INIT_TYPES = {
    "eii_gmm": make_eii_gmm_init,
    "random": make_random_init,
    "random_far": make_random_far_init,
}
INIT_LABELS = {
    "eii_gmm": "EII GMM",
    "random": "Random",
    "random_far": "Random far",
}


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.save_checkpoints:
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    if args.save_datasets:
        os.makedirs(os.path.join(out_dir, "datasets"), exist_ok=True)

    # Under Slurm, use the array task ID as the seed offset
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.seed))

    summary_rows = []
    tsne_data = []

    for ds_idx, sim_params in enumerate(DATASETS):
        scenario = sim_params["scenario"]
        K_true = sim_params["K"]
        gen_params = {k: v for k, v in sim_params.items() if k != "scenario"}
        gen_params["seed"] = gen_params["seed"] + task_id

        print(f"\n{'=' * 65}\nDataset {ds_idx} | {scenario} | K_true={K_true}\n{'=' * 65}")

        torch.manual_seed(gen_params["seed"])
        np.random.seed(gen_params["seed"])
        X_sim, c_true, _, _ = scflame_synthetic_separated(**gen_params)
        X_sim = X_sim.to(DEVICE)
        c_true_np = c_true.cpu().numpy()
        C_sim = X_sim.sum(dim=1).float()
        C_sim = C_sim / C_sim.mean()

        if args.save_datasets:
            ds_dir = os.path.join(out_dir, "datasets")
            pd.DataFrame(X_sim.cpu().numpy()).to_csv(
                os.path.join(ds_dir, f"dataset_{task_id}_{scenario}_counts.csv"), index=False)
            pd.DataFrame({"cluster": c_true_np}).to_csv(
                os.path.join(ds_dir, f"dataset_{task_id}_{scenario}_labels.csv"), index=False)

        for repeat in range(args.n_repeats):
            print(f"Starting repeat {repeat + 1}/{args.n_repeats} for {scenario}...")
            repeat_seed = (gen_params["seed"] * 100) + repeat
            torch.manual_seed(repeat_seed)
            np.random.seed(repeat_seed)

            # NBFA is fit once per repeat and shared across initialisation strategies
            nbfa_result = train_nb_fa(
                X_sim, q=sim_params["q"], C=C_sim,
                mc_samples=args.mc_samples, epochs=args.nbfa_epochs,
                lr=1e-2, disp=sim_params["r_true"], verbose=args.verbose,
            )
            if args.save_checkpoints:
                torch.save(nbfa_result, os.path.join(
                    out_dir, "checkpoints", f"nbfa_{task_id}_{scenario}_rep{repeat + 1}.pt"))

            if args.save_latents and repeat == 0:
                Z = nbfa_result["m"].detach().cpu().numpy()
                pd.DataFrame(Z).to_csv(
                    os.path.join(out_dir, f"nbfa_latents_{task_id}_{scenario}.csv"), index=False)
                gmm = EII_GMM(n_components=K_true, n_init=10, max_iter=100, tol=1e-4, random_state=repeat_seed)
                gmm.fit(Z)
                pd.DataFrame({"cluster": gmm.predict(Z)}).to_csv(
                    os.path.join(out_dir, f"nbfa_eii_gmm_alloc_{task_id}_{scenario}.csv"), index=False)

            for init_type, init_fn in INIT_TYPES.items():
                print(f"  [{scenario} R{repeat + 1}] Init: {INIT_LABELS[init_type]}")

                gmm_init = init_fn(nbfa_result, K_true, DEVICE, seed=repeat_seed + 500)

                final_result = train_scflame(
                    X_sim, nbfa_result, gmm_init, K_true,
                    epochs=args.scflame_epochs, mc_samples=args.mc_samples,
                    lr_latent=1e-3, latent_steps=3,
                    temp_annealing=True, max_temp=1.5,
                    L_steps=3, verbose=args.verbose,
                )
                if args.save_checkpoints:
                    torch.save(final_result, os.path.join(
                        out_dir, "checkpoints",
                        f"scflame_{task_id}_{scenario}_rep{repeat + 1}_{init_type}.pt"))

                c_pred_final = final_result["r"].argmax(dim=1).cpu().numpy()

                summary_rows.append({
                    "dataset_idx": ds_idx,
                    "scenario": scenario,
                    "repeat": repeat,
                    "K_true": K_true,
                    "K_final": len(np.unique(c_pred_final)),
                    "init_type": init_type,
                    "ari_final": adjusted_rand_score(c_true_np, c_pred_final),
                    "nmi_final": normalized_mutual_info_score(c_true_np, c_pred_final),
                    "acc_final": clustering_accuracy(c_true_np, c_pred_final),
                    "elbo_final": final_result["trace"]["elbo"][-1],
                    **{f"sim_{k}": v for k, v in sim_params.items()},
                })

                if args.save_latents and repeat == 0:
                    latents = final_result["m"].detach().cpu().numpy()
                    pd.DataFrame(latents).to_csv(
                        os.path.join(out_dir, f"scflame_latents_{task_id}_{scenario}_{init_type}.csv"),
                        index=False)
                    pd.DataFrame({"cluster": c_pred_final}).to_csv(
                        os.path.join(out_dir, f"scflame_alloc_{task_id}_{scenario}_{init_type}.csv"),
                        index=False)

                if not args.no_tsne and repeat == 0:
                    tsne_data.append(dict(
                        dataset_idx=ds_idx, scenario=scenario, init_type=init_type,
                        latents=final_result["m"].detach().cpu().numpy(),
                        true_labels=c_true_np, pred_labels=c_pred_final, K_true=K_true,
                    ))

    summary_path = os.path.join(out_dir, f"scflame_simulation_summary_{task_id}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    if tsne_data:
        print("\nGenerating t-SNE plots...")
        for entry in tsne_data:
            tsne_path = os.path.join(
                out_dir, f"tsne_{task_id}_{entry['scenario']}_{entry['init_type']}.png")
            plot_tsne(
                latents=entry["latents"], true_labels=entry["true_labels"],
                pred_labels=entry["pred_labels"], K_true=entry["K_true"],
                save_path=tsne_path, seed=args.seed,
            )
            print(f"  Saved: {tsne_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit scFLAME on simulated data across scenarios and init strategies.")
    p.add_argument("--out-dir", required=True, help="Directory to write results to")
    p.add_argument("--n-repeats", type=int, default=3, help="Number of independent repeats per scenario")
    p.add_argument("--nbfa-epochs", type=int, default=350, help="Epochs for the NBFA warm-start stage")
    p.add_argument("--scflame-epochs", type=int, default=250, help="Epochs for the joint scFLAME fit")
    p.add_argument("--mc-samples", type=int, default=10, help="Monte Carlo samples per ELBO evaluation")
    p.add_argument("--seed", type=int, default=1,
                    help="Random seed; overridden by $SLURM_ARRAY_TASK_ID when running under Slurm")
    p.add_argument("--save-datasets", action="store_true",
                    help="Save each simulated dataset's counts/labels to out-dir/datasets/")
    p.add_argument("--save-checkpoints", action="store_true",
                    help="Save nbfa_result/scflame_result .pt checkpoints under out-dir/checkpoints/")
    p.add_argument("--save-latents", action="store_true",
                    help="Save NBFA and scFLAME latents/allocations as CSVs")
    p.add_argument("--no-tsne", action="store_true", help="Skip generating t-SNE plots")
    p.add_argument("--verbose", action="store_true", help="Print per-epoch training progress")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())