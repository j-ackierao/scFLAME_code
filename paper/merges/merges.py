# ============================================================
# merges.py
# ================
# Reproduces the PBMC 4k subtype discovery section.
 
# This is the paper-exact version of scripts/run_merge.py with the PBMC-specific
# settings actually used for the reported numbers (K_init sweep, q=15,
# L_reg=0.01).
 
# Expected input files under --data-dir:
#     counts.csv       cells x genes, already subset to the HVGs used in the
#                       paper (this script does not do its own HVG selection --
#                       see paper/preprocessing/ for how this file was made)
#     clusters.csv      columns "Barcode", "Cluster" (10x Genomics' own naming;
#                       not the "cell_id"/"celltype" convention scflame.utils
#                       .load_dataset expects, hence the small loader below)
#     dispersion.csv     columns "gene", "dispersion" (edgeR tagwise estimates)
#     library_sizes.csv  columns "cell_id", "tmm_lib_size" (TMM size factors)

# The Baron results in the Supplement used the same seeds and parameters, but with the 
# Baron-specific input files (counts.csv, clusters.csv, dispersion.csv, library_sizes.csv). 
# The Baron dataset was preprocessed and loaded in the same way as in realdata.py.
 
# ============================================================

from __future__ import annotations
 
import os
import sys
 
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scflame import DEVICE, train_nb_fa, train_scflame, make_eii_gmm_init
from scflame.merging import greedy_merge_full, m_step_cluster_params_full

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pbmc4k(data_dir: str):
    """Load the pre-selected HVG counts, 10x-style cluster labels, dispersions,
    and library sizes for the PBMC 4k dataset. Unlike scflame.utils.load_dataset,
    this does not do HVG selection (counts.csv is already subset) and expects
    10x's own "Barcode"/"Cluster" column names."""
    counts_df = pd.read_csv(os.path.join(data_dir, "counts.csv"), index_col=0)
    X = torch.tensor(counts_df.values.astype(np.float32), dtype=torch.float32).to(DEVICE)
 
    clusters_df = pd.read_csv(os.path.join(data_dir, "clusters.csv")).set_index("Barcode")
    clusters_df = clusters_df.loc[counts_df.index]
    c_true = clusters_df["Cluster"].astype(int).to_numpy()
 
    dispersions_csv = pd.read_csv(os.path.join(data_dir, "dispersion.csv"))
    dispersions = dispersions_csv.set_index("gene").loc[counts_df.columns]["dispersion"].values
 
    lib_sizes_csv = pd.read_csv(os.path.join(data_dir, "library_sizes.csv")).set_index("cell_id")
    C = lib_sizes_csv["tmm_lib_size"].values
 
    return X, counts_df.columns.tolist(), c_true, dispersions, C

# ============================================================
# Simulation settings
# ============================================================

K_INIT_VALUES = [8, 10, 15, 20]
 
NBFA_EPOCHS   = 500
NBMCFA_EPOCHS = 300
MC_SAMPLES    = 10
TOP_M         = None   # None = all pairs
K_FINAL       = 2      # final number of clusters
 
out_dir = "results/pbmc4k_merge"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)

# ============================================================
# Load data
# ============================================================

# Use SLURM task ID as seed (fallback to 0 if not running under Slurm)
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) - 1

X, gene_names, c_true_np, dispersions, C = load_pbmc4k("data/pbmc4k")
K_true = len(np.unique(c_true_np))

# ============================================================
# Run
# ============================================================

history_rows = []

# Fit NBFA once
torch.manual_seed(42 + task_id); np.random.seed(42 + task_id)
print("Fitting NBFA...")
C_tensor = torch.tensor(C, dtype=torch.float32).to(device)

nbfa_result = train_nb_fa(
    X, q=15, C=C_tensor,
    mc_samples=MC_SAMPLES, epochs=NBFA_EPOCHS,
    lr=1e-2, disp=torch.tensor(dispersions, dtype=torch.float32), verbose=True)
print("NBFA done.")

torch.save(nbfa_result, f"{out_dir}/checkpoints/nbfa_result_task_{task_id}.pt")
print("Saved NBFA checkpoint.")

alloc_cols = {}
stability_records = {}

for K_init in K_INIT_VALUES:
    print(f"\n{'='*65}")
    print(f"K_init={K_init} (K_true={K_true})")
    print(f"{'='*65}")

    torch.manual_seed(42 + task_id); np.random.seed(42 + task_id)
    gmm_init = make_eii_gmm_init(nbfa_result, K_init, device, seed=42 + task_id)

    initial_result = train_scflame(
        X, nbfa_result, gmm_init, K_init,
        epochs=NBMCFA_EPOCHS, mc_samples=MC_SAMPLES,
        lr_latent=1e-3, latent_steps=3,
        temp_annealing=True, max_temp=1.5,
        L_steps=3, L_reg = 0.01, verbose=False)

    torch.save(initial_result, f"{out_dir}/checkpoints/initial_result_task_{task_id}_Kinit_{K_init}.pt")

    c_pred_init = initial_result['r'].argmax(dim=1).cpu().numpy()
    ari_init    = adjusted_rand_score(c_true_np, c_pred_init)
    elbo_final = initial_result['trace']['elbo'][-1]
    print(f"  ARI after NBMCFA init: {ari_init:.3f}")
    print(f"  Final ELBO after NBMCFA init: {elbo_final:.3f}")

    m_init     = initial_result['m']
    log_s_init = initial_result['log_s']
    s_init     = torch.exp(log_s_init)
    r_init     = initial_result['r']

    nu_k_init, Sigma_k_init, pi_k_init = m_step_cluster_params_full(m_init, s_init, r_init)

    merge_input = {
        'r':       r_init,
        'm':       m_init,
        'log_s':   log_s_init,
        'nu_k':    nu_k_init,
        'Sigma_k': Sigma_k_init,
        'pi_k':    pi_k_init,
    }

    final_result, K_reached, history, allocations = greedy_merge_full(
        X, merge_input,
        mc_samples=MC_SAMPLES,
        top_m=TOP_M,
        verbose=True,
        c_true=c_true_np,
        K_true=K_true,
        criterion='icl',
        K_final=K_FINAL,
        checkpoint_dir=f"{out_dir}/checkpoints",
        checkpoint_tag=f"task_{task_id}_Kinit_new{K_init}")

    # stability inside the loop, per K_init
    for K_step, labels in allocations.items():
        alloc_cols[f'Kinit{K_init}_K{K_step}'] = labels

    c_pred_final = final_result['r'].argmax(dim=1).cpu().numpy()
    ari_final    = adjusted_rand_score(c_true_np, c_pred_final)
    print(f"  After merge: ARI={ari_final:.3f} | K={K_reached} (true={K_true})")

    for step in history:
        history_rows.append({
            'K_init': K_init,
            'ari_init': ari_init,
            **step,
        })

# ============================================================
# Save
# ============================================================

history_df = pd.DataFrame(history_rows)

history_df.to_csv(f"{out_dir}/merge_history_{task_id}.csv", index=False)
print(f"\nSaved: {out_dir}/merge_history_{task_id}.csv")

alloc_df = pd.DataFrame(alloc_cols)
alloc_df.index.name = 'cell'
alloc_df.to_csv(f"{out_dir}/cluster_allocations_{task_id}.csv")
print(f"Saved: {out_dir}/cluster_allocations_{task_id}.csv")

