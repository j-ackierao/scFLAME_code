"""
run_scflame.py
Fits scFLAME under three initialization strategies on one real dataset.
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

INIT_TYPES = {
    "eii_gmm": make_eii_gmm_init,
    "random": make_random_init,
    "random_far": make_random_far_init,
}
LABELS = {"eii_gmm": "NBFA+EII GMM", "random": "Random", "random_far": "Random far"}

def parse_args():
    p = argparse.ArgumentParser(description="scFLAME init-strategy benchmark on a real dataset")
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
    p.add_argument("--dispersions_path", required=True)
    p.add_argument("--libsize_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    os.makedirs(args.out_dir, exist_ok=True)

    loader = LOADERS[args.dataset]

    dat = loader(counts_path=args.counts_path, clusters_path=args.clusters_path,
                 dispersions_path=args.dispersions_path, libsize_path=args.libsize_path)
    X_np, c_true, K_true = dat["X_np"], dat["c_true"], dat["K_true"]
    dispersions, C = dat["dispersions"], dat["C"]
    print(f"N={X_np.shape[0]}, D={X_np.shape[1]}, K_true={K_true}")

    X = torch.tensor(X_np, dtype=torch.float32, device=device)

    summary_rows = []
    tsne_data = []

    for repeat in range(5):
        print(f"Starting Repeat {repeat + 1}/5...")
        repeat_seed = 21 + repeat
        torch.manual_seed(repeat_seed)
        np.random.seed(repeat_seed)

        nbfa_result = train_nb_fa(
            X, q=15, C=C, mc_samples=10, epochs=500,
            lr=1e-2, disp=dispersions, verbose=False,
        )

        for init_type, init_fn in INIT_TYPES.items():
            print(f"  [R{repeat + 1}] Init: {LABELS[init_type]}")
            K_init = K_true

            gmm_init = init_fn(nbfa_result, K_init, device, seed=repeat_seed + 500)

            final_result = train_scflame(
                X, nbfa_result, gmm_init, K_init,
                epochs=300, mc_samples=10,
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
    out_path = os.path.join(args.out_dir, f"scflame_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if tsne_data:
        tsne_path = os.path.join(args.out_dir, f"scflame_tsne_data.npz")
        np.savez(tsne_path, data=np.array(tsne_data, dtype=object), allow_pickle=True)
        print(f"Saved: {tsne_path}")


if __name__ == "__main__":
    main()