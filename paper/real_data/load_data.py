"""
Data loading for real-data scFLAME comparisons.

One function per dataset each with the same shape of options:
    n_hvgs           - int, or None to use every gene in the counts file as-is
    normalize        - if True, library-size normalize (to the median
                        library size) and log1p the counts actually
                        returned.
    dispersions_path - if given, load and return per-gene dispersions
                        aligned to the returned gene order; None to skip
    libsize_path     - if given, load and return per-cell effective
                        library sizes aligned to the returned cell order;
                        None to skip
    device           - torch device for the returned X tensor
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── shared helpers ───────────────────────────────────────────────────────────

def _select_hvgs(counts_df: pd.DataFrame, n_hvgs: int) -> pd.DataFrame:
    """Return counts_df subset to the top n_hvgs columns by raw log1p variance."""
    log_df = np.log1p(counts_df.values)
    col_var = log_df.var(axis=0)
    top_idx = np.argsort(col_var)[-n_hvgs:]
    return counts_df.iloc[:, top_idx]


def _normalize(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Library-size normalize to the median library size, then log1p."""
    libsize = counts_df.values.sum(axis=1, keepdims=True)
    s0 = np.median(libsize)
    return np.log1p(counts_df / libsize * s0)


def _load_dispersions(path: str, gene_order) -> np.ndarray:
    disp_df = pd.read_csv(path)
    return disp_df.set_index("gene").loc[gene_order]["dispersion"].to_numpy()


def _load_library_sizes(path: str, cell_order, id_col: str = "cell_id", value_col: str = "tmm_lib_size") -> np.ndarray:
    lib_df = pd.read_csv(path)
    return lib_df.set_index(id_col).loc[cell_order][value_col].to_numpy()


def _prepare_X(counts_df: pd.DataFrame, n_hvgs, normalize: bool, device: str):
    """Shared HVG-selection on raw counts + optional normalize + tensor conversion."""
    df_hv = counts_df if n_hvgs is None else _select_hvgs(counts_df, n_hvgs)
    out_df = _normalize(df_hv) if normalize else df_hv

    X_np = out_df.values.astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    return X, df_hv  # df_hv carries the selected gene columns, at raw scale, for gene_order alignment


# ── PBMC 4k (10x) ────────────────────────────────────────────────────────────

def load_pbmc4k(counts_path: str, clusters_path: str,
                 dispersions_path: str = None, libsize_path: str = None,
                 n_hvgs: int = None, normalize: bool = False,
                 device: str = DEVICE):
    """
    counts_path: cells x genes raw counts CSV, already pre-filtered to
        5000 HVGs upstream (unlike uterus/zeisel, which are loaded from
        the full gene set and filtered here). n_hvgs defaults to None -
        no further selection is applied on top of that existing panel.
    clusters_path: 10x_clusters.csv with columns Barcode, Cluster.
    """
    counts_df = pd.read_csv(counts_path, index_col=0)
    X, df_hv = _prepare_X(counts_df, n_hvgs, normalize, device)

    clusters_df = pd.read_csv(clusters_path).set_index("Barcode").loc[counts_df.index]
    c_true = clusters_df["Cluster"].astype(int).to_numpy()
    K_true = len(np.unique(c_true))

    result = {"X": X, "c_true": c_true, "K_true": K_true, "genes": list(df_hv.columns), "cells": list(counts_df.index)}
    if dispersions_path is not None:
        result["dispersions"] = _load_dispersions(dispersions_path, df_hv.columns)
    if libsize_path is not None:
        result["C"] = _load_library_sizes(libsize_path, counts_df.index)
    return result


# ── Mouse uterus ────────────────────────────────────────────────

def load_uterus(counts_path: str, clusters_path: str,
                 dispersions_path: str = None, libsize_path: str = None,
                 n_hvgs: int = 5000, normalize: bool = False,
                 device: str = DEVICE):
    counts_df = pd.read_csv(counts_path, index_col=0)
    X, df_hv = _prepare_X(counts_df, n_hvgs, normalize, device)

    clusters_df = pd.read_csv(clusters_path).set_index("cell_id").loc[counts_df.index]
    c_true = clusters_df["cluster_id"].astype(str).str.extract(r"(\d+)$")[0].astype(int).to_numpy()
    K_true = len(np.unique(c_true))

    result = {"X": X, "c_true": c_true, "K_true": K_true, "genes": list(df_hv.columns), "cells": list(counts_df.index)}
    if dispersions_path is not None:
        result["dispersions"] = _load_dispersions(dispersions_path, df_hv.columns)
    if libsize_path is not None:
        result["C"] = _load_library_sizes(libsize_path, counts_df.index)
    return result

# ── Mouse uterus ────────────────────────────────────────────────

def load_uterus(counts_path: str, clusters_path: str,
                 dispersions_path: str = None, libsize_path: str = None,
                 n_hvgs: int = 5000, normalize: bool = False,
                 device: str = DEVICE):
    counts_df = pd.read_csv(counts_path, index_col=0)
    X, df_hv = _prepare_X(counts_df, n_hvgs, normalize, device)

    clusters_df = pd.read_csv(clusters_path).set_index("cell_id").loc[counts_df.index]
    c_true = clusters_df["cluster_id"].astype(str).str.extract(r"(\d+)$")[0].astype(int).to_numpy()
    K_true = len(np.unique(c_true))

    result = {"X": X, "c_true": c_true, "K_true": K_true, "genes": list(df_hv.columns), "cells": list(counts_df.index)}
    if dispersions_path is not None:
        result["dispersions"] = _load_dispersions(dispersions_path, df_hv.columns)
    if libsize_path is not None:
        result["C"] = _load_library_sizes(libsize_path, counts_df.index)
    return result

# ── Zeisel mouse brain ───────────────────────────────────────────────────────

def load_zeisel(counts_path: str, clusters_path: str,
                 dispersions_path: str = None, libsize_path: str = None,
                 n_hvgs: int = 5000, normalize: bool = False,
                 device: str = DEVICE):

    counts_df = pd.read_csv(counts_path, index_col=0)
    X, df_hv = _prepare_X(counts_df, n_hvgs, normalize, device)

    clusters_df = pd.read_csv(clusters_path).set_index("cell_id").loc[counts_df.index]
    c_true = clusters_df["celltype"].astype(int).to_numpy()
    K_true = len(np.unique(c_true))

    result = {"X": X, "c_true": c_true, "K_true": K_true, "genes": list(df_hv.columns), "cells": list(counts_df.index)}
    if dispersions_path is not None:
        result["dispersions"] = _load_dispersions(dispersions_path, df_hv.columns)
    if libsize_path is not None:
        result["C"] = _load_library_sizes(libsize_path, counts_df.index)
    return result

# ── TM spleen 10X/SS2 ───────────────────────────────────────────────────────
def load_tm_spleen(counts_path: str, clusters_path: str,
                 dispersions_path: str = None, libsize_path: str = None,
                 n_hvgs: int = 8000, normalize: bool = False,
                 device: str = DEVICE):

    counts_df = pd.read_csv(counts_path, index_col=0)
    X, df_hv = _prepare_X(counts_df, n_hvgs, normalize, device)

    clusters_df = pd.read_csv(clusters_path).set_index("cell_id").loc[counts_df.index]
    c_true = clusters_df["cluster_id"].astype(int).to_numpy()
    K_true = len(np.unique(c_true))

    result = {"X": X, "c_true": c_true, "K_true": K_true, "genes": list(df_hv.columns), "cells": list(counts_df.index)}
    if dispersions_path is not None:
        result["dispersions"] = _load_dispersions(dispersions_path, df_hv.columns)
    if libsize_path is not None:
        result["C"] = _load_library_sizes(libsize_path, counts_df.index)
    return result

# ── Segerstolpe pancreas ─────────────────────────────────────────────────────
 
def load_segerstolpe(counts_path: str, clusters_path: str,
                      dispersions_path: str = None, libsize_path: str = None,
                      n_hvgs: int = 5000, normalize: bool = False,
                      device: str = "cpu"):
    """
    NOTE: counts_df, clusters_df and the library-size CSV are all
    filtered/aligned here by POSITION (a boolean mask), assumes
    all three files share the exact same row order to begin with. Our
    preprocessing code ensures this, but if parts are edited, verify this.
    """
    df = pd.read_csv(counts_path, index_col=0)
    clusters_df = pd.read_csv(clusters_path)
    c_true = clusters_df["celltype"].astype(int).to_numpy()
 
    celltype_counts = pd.Series(c_true).value_counts()
    min_count = int(0.02 * len(c_true))
    valid_celltypes = celltype_counts[celltype_counts >= min_count].index
    mask = np.isin(c_true, valid_celltypes)
 
    df = df.iloc[mask]
    c_true = c_true[mask]
    K_true = len(valid_celltypes)
 
    X, df_hv = _prepare_X(df, n_hvgs, normalize, device)
 
    result = {"X": X, "c_true": c_true, "K_true": K_true, "genes": list(df_hv.columns), "cells": list(df.index)}
 
    if dispersions_path is not None:
        disp_df = pd.read_csv(dispersions_path)
        disp_df["gene"] = disp_df["gene"].str.replace("-", ".", regex=False).str.replace("+", ".", regex=False)
        result["dispersions"] = disp_df.set_index("gene").loc[df_hv.columns]["dispersion"].to_numpy()
 
    if libsize_path is not None:
        lib_df = pd.read_csv(libsize_path)
        C = lib_df.set_index("cell_id")["tmm_lib_size"].to_numpy()
        result["C"] = C[mask]
 
    return result
 
 
# ── Baron pancreas ───────────────────────────────────────────────────────────
 
def load_baron(counts_path: str, clusters_path: str,
               dispersions_path: str = None, libsize_path: str = None,
               n_hvgs: int = 5000, normalize: bool = False,
               device: str = "cpu"):
    df = pd.read_csv(counts_path, index_col=0)
 
    clusters_df = pd.read_csv(clusters_path).set_index("cell_id")
    counts_per_type = clusters_df["celltype"].value_counts()
    valid_types = counts_per_type[counts_per_type >= 50].index #2.5% of dataset
    clusters_df = clusters_df[clusters_df["celltype"].isin(valid_types)]
 
    df = df.loc[clusters_df.index]
 
    X, df_hv = _prepare_X(df, n_hvgs, normalize, device)
 
    clusters_df = clusters_df.loc[df_hv.index]
    c_true = clusters_df["celltype"].astype(int).to_numpy()
    K_true = len(np.unique(c_true))
 
    result = {"X": X, "c_true": c_true, "K_true": K_true, "genes": list(df_hv.columns), "cells": list(df_hv.index)}
 
    if dispersions_path is not None:
        disp_df = pd.read_csv(dispersions_path)
        disp_df["gene"] = disp_df["gene"].str.replace("-", ".", regex=False).str.replace("+", ".", regex=False)
        result["dispersions"] = disp_df.set_index("gene").loc[df_hv.columns]["dispersion"].to_numpy()
 
    if libsize_path is not None:
        lib_df = pd.read_csv(libsize_path).set_index("cell_id").loc[df_hv.index]
        result["C"] = lib_df["tmm_lib_size"].to_numpy()
 
    return result

# ── PBMC 68k ─────────────────────────────────────────────────────────────────
 
def load_pbmc68k(data_dir: str, clusters_path: str,
                  dispersions_path: str = None, libsize_path: str = None,
                  device: str = DEVICE):
    """
    data_dir: directory containing matrix.mtx, genes.tsv, barcodes.tsv,
    in standard 10x format (as written by preprocess_pbmc68k_real.R) -
    already HVG-filtered upstream (5000 genes), no further selection here.
    """
    adata = sc.read_10x_mtx(data_dir, var_names="gene_ids", make_unique=True)
 
    X_np = adata.X.toarray().astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    genes, barcodes = list(adata.var_names), list(adata.obs_names)
 
    clusters_df = pd.read_csv(clusters_path).set_index("cell_id").loc[barcodes]
    c_true = clusters_df["cluster"].astype(int).to_numpy()
    K_true = len(np.unique(c_true))
    print(f"K_true={K_true}")
 
    result = {"X": X, "c_true": c_true, "K_true": K_true, "genes": genes, "cells": barcodes}
    if dispersions_path is not None:
        result["dispersions"] = _load_dispersions(dispersions_path, genes)
    if libsize_path is not None:
        result["C"] = _load_library_sizes(libsize_path, barcodes)
    return result
 
 
# ── Pancreas (4-study merge, batch labels) ──────────────────────────────────
 
def load_pancreas(counts_path: str, clusters_path: str,
                   dispersions_path: str = None, libsize_path: str = None,
                   device: str = DEVICE):
    """
    clusters_path: metadata CSV with columns cell_id, cluster_id, batch_id.
    Returns batch_true/batch_no in addition to the usual fields.
    No HVG filtering; use full subset of ~6k genes.
    """
    counts_df = pd.read_csv(counts_path, index_col=0)
    X_np = counts_df.values.astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
 
    clusters_df = pd.read_csv(clusters_path).set_index("cell_id").loc[counts_df.index]
    c_true, celltype_labels = pd.factorize(clusters_df["cluster_id"])
    batch_true, protocol_labels = pd.factorize(clusters_df["batch_id"])
    c_true, batch_true = c_true.astype(int), batch_true.astype(int)
    K_true, batch_no = len(np.unique(c_true)), len(np.unique(batch_true))
 
    result = {
        "X": X, "c_true": c_true, "K_true": K_true,
        "batch_true": batch_true, "batch_no": batch_no,
        "celltype_labels": list(celltype_labels), "protocol_labels": list(protocol_labels),
        "genes": list(counts_df.columns), "cells": list(counts_df.index),
    }
    if dispersions_path is not None:
        result["dispersions"] = _load_dispersions(dispersions_path, counts_df.columns)
    if libsize_path is not None:
        result["C"] = _load_library_sizes(libsize_path, counts_df.index)
    return result

# ── LUAD (batch labels) ──────────────────────────────────────────────────────
 
def load_luad(counts_path: str, clusters_path: str,
              dispersions_path: str = None, libsize_path: str = None,
              n_hvgs: int = 5000, normalize: bool = False,
              device: str = DEVICE):
    """
    clusters_path: metadata CSV with columns cell_id, celltype, protocol.
    Returns batch_true/batch_no in addition to the usual fields.
    """
    counts_df = pd.read_csv(counts_path, index_col=0)
    X, df_hv = _prepare_X(counts_df, n_hvgs, normalize, device)
 
    clusters_df = pd.read_csv(clusters_path).set_index("cell_id").loc[counts_df.index]
    c_true, celltype_labels = pd.factorize(clusters_df["celltype"])
    batch_true, protocol_labels = pd.factorize(clusters_df["protocol"])
    c_true, batch_true = c_true.astype(int), batch_true.astype(int)
    K_true, batch_no = len(np.unique(c_true)), len(np.unique(batch_true))
 
    result = {
        "X": X, "c_true": c_true, "K_true": K_true,
        "batch_true": batch_true, "batch_no": batch_no,
        "celltype_labels": list(celltype_labels), "protocol_labels": list(protocol_labels),
        "genes": list(df_hv.columns), "cells": list(counts_df.index),
    }
    if dispersions_path is not None:
        result["dispersions"] = _load_dispersions(dispersions_path, df_hv.columns)
    if libsize_path is not None:
        result["C"] = _load_library_sizes(libsize_path, counts_df.index)
    return result