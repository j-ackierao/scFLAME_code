"""
preprocess_scanpy.py
====================
Python/Scanpy alternative to preprocess.R: turn an AnnData (.h5ad) file or 10x
Genomics output into the input files scFLAME expects, with the same QC steps.

--input can be:
    an .h5ad file
    a 10x .h5 file (e.g. filtered_feature_bc_matrix.h5)
    a 10x matrix directory (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz), or a Cell
    Ranger outs/ directory containing filtered_feature_bc_matrix/
For 10x input only Gene Expression features are kept (antibody/CRISPR capture dropped),
and cell-type labels or batches can be added with --metadata.

Differences from preprocess.R (no edgeR needed):
    library sizes   total counts per cell (computed before HVG selection), instead of
                    TMM-adjusted library sizes. Written to the tmm_lib_size column so
                    the scFLAME loader reads it unchanged.
    dispersions     edgeR-style NB dispersion phi per gene (Var = mu + phi * mu^2), the
                    same convention as edgeR's tagwise.dispersion. Estimated by method of 
                    moments with the library sizes as offsets, then shrunk towards an 
                    abundance trend (see estimate_dispersions). With thousands of cells, 
                    edgeR's own shrinkage is also small, so the two are similar.
    HVGs            scanpy's "seurat_v3" flavour (the same VST method as Seurat's
                    FindVariableFeatures), which needs the scikit-misc package.

Unlike preprocess.R, counts.csv holds the HVGs only, so counts and dispersions cover
the same genes. Pass --n-hvgs to the run scripts no larger than --n-hvgs here.

Writes to --out-dir:
    counts.csv          cells x genes raw counts, first column = cell ID
    library_sizes.csv   cell_id, tmm_lib_size (= total counts)
    dispersion.csv      gene, dispersion (phi)
    clusters.csv        cell_id, celltype (integer code), [batch_id]   (only if --label-key)
    celltype_key.csv    integer code -> original label                  (only if --label-key)
    processed.h5ad      filtered AnnData (all kept genes, HVGs flagged in var["highly_variable"],
                        raw counts in layers["counts"]) for downstream_scanpy.py

Usage:
    python scripts/preprocess_scanpy.py --input my_data.h5ad --out-dir data/my_data \\
        --label-key cell_type --counts-layer counts
    python scripts/preprocess_scanpy.py --input sample1/outs --out-dir data/sample1 \\
        --metadata annotations.csv --label-key cell_type
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def get_counts(adata: sc.AnnData, layer: str | None) -> sp.csr_matrix:
    """Raw counts as a sparse matrix, checking they really are counts."""
    X = adata.layers[layer] if layer is not None else adata.X
    X = sp.csr_matrix(X)
    sample = X.data[: 100_000]
    if sample.size and (sample.min() < 0 or not np.allclose(sample, np.round(sample))):
        where = f"layers['{layer}']" if layer else "adata.X"
        raise SystemExit(
            f"{where} doesn't look like raw counts (negative or non-integer values). "
            "scFLAME models raw counts: pass the layer that holds them with --counts-layer."
        )
    return X


def load_input(path: str, metadata: str | None) -> sc.AnnData:
    """Read an .h5ad, a 10x .h5 file, or a 10x matrix directory; optionally join per-cell metadata."""
    if os.path.isdir(path):
        for sub in ("", "filtered_feature_bc_matrix", os.path.join("outs", "filtered_feature_bc_matrix")):
            d = os.path.join(path, sub)
            if any(os.path.exists(os.path.join(d, f)) for f in ("matrix.mtx", "matrix.mtx.gz")):
                print(f"Reading 10x matrix directory {d}")
                adata = sc.read_10x_mtx(d, var_names="gene_symbols", gex_only=True)
                break
        else:
            raise SystemExit(f"No matrix.mtx(.gz) found in {path} or its filtered_feature_bc_matrix/")
    elif path.endswith(".h5ad"):
        adata = sc.read_h5ad(path)
    elif path.endswith(".h5"):
        print(f"Reading 10x .h5 file {path}")
        adata = sc.read_10x_h5(path, gex_only=True)
    else:
        raise SystemExit("--input must be an .h5ad file, a 10x .h5 file, or a 10x matrix directory")

    if metadata is not None:
        meta = pd.read_csv(metadata, index_col=0)  # first column = cell barcode
        shared = adata.obs_names.intersection(meta.index)
        if len(shared) == 0:
            raise SystemExit("No barcodes in --metadata match the data. Check the first column holds "
                             "barcodes in the same format (e.g. with or without the '-1' suffix).")
        if len(shared) < adata.n_obs:
            print(f"--metadata covers {len(shared)} of {adata.n_obs} cells; keeping only those cells.")
        adata = adata[shared].copy()
        adata.obs = adata.obs.join(meta.loc[shared], rsuffix="_meta")
    return adata


def estimate_dispersions(X: sp.csr_matrix, lib_sizes: np.ndarray,
                         prior_df: float = 10.0, n_bins: int = 20,
                         min_disp: float = 1e-4, max_disp: float = 1e3) -> np.ndarray:
    """NB dispersion phi per gene (Var = mu + phi * mu^2), edgeR convention.

    1. Offsets s_i = library size / mean library size; gene rate m_g = sum_i x_ig / sum_i s_i,
       so the expected count is mu_ig = s_i * m_g.
    2. Method of moments per gene:
           phi_g = [ sum_i (x_ig - mu_ig)^2 - sum_i mu_ig ] / sum_i mu_ig^2
       computed from sparse sums so the matrix is never densified.
    3. Trend: median log-dispersion in bins of average abundance, interpolated.
    4. Empirical-Bayes-style shrinkage on the log scale, as edgeR does with prior.df:
           log phi_g <- (df * log phi_g + prior_df * log trend_g) / (df + prior_df),  df = n_cells - 1
       Genes whose moment estimate is <= 0 (no overdispersion detected) are set to the trend.
    """
    n = X.shape[0]
    s = lib_sizes / lib_sizes.mean()
    sum_x = np.asarray(X.sum(axis=0)).ravel()
    sum_x2 = np.asarray(X.multiply(X).sum(axis=0)).ravel()
    sum_sx = np.asarray(X.T @ s).ravel()
    sum_s, sum_s2 = s.sum(), (s ** 2).sum()

    m = sum_x / sum_s
    sum_sq_resid = sum_x2 - 2 * m * sum_sx + m ** 2 * sum_s2
    sum_mu = m * sum_s
    sum_mu2 = m ** 2 * sum_s2
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_raw = (sum_sq_resid - sum_mu) / sum_mu2

    ok = np.isfinite(phi_raw) & (phi_raw > 0)
    log_abund = np.log(m + 1e-8)
    log_phi = np.log(np.clip(phi_raw, min_disp, max_disp))

    # abundance trend from genes with a usable estimate
    edges = np.unique(np.quantile(log_abund[ok], np.linspace(0, 1, n_bins + 1)))
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = ok & (log_abund >= lo) & (log_abund <= hi)
        if in_bin.sum() >= 5:
            centres.append(np.median(log_abund[in_bin]))
            medians.append(np.median(log_phi[in_bin]))
    if centres:
        trend = np.interp(log_abund, centres, medians)
    else:  # too few genes for a trend: use the common (pooled) dispersion
        common = max((sum_sq_resid - sum_mu).sum() / sum_mu2.sum(), min_disp)
        trend = np.full_like(log_phi, np.log(common))

    df = n - 1
    log_phi = np.where(ok, (df * log_phi + prior_df * trend) / (df + prior_df), trend)
    return np.clip(np.exp(log_phi), min_disp, max_disp)


def write_counts_csv(X: sp.csr_matrix, cell_ids, gene_names, path: str, chunk: int = 2000) -> None:
    """Dense cells x genes CSV, written in chunks so large datasets fit in memory."""
    for start in range(0, X.shape[0], chunk):
        block = pd.DataFrame(X[start:start + chunk].toarray().astype(np.int64),
                             index=cell_ids[start:start + chunk], columns=gene_names)
        block.to_csv(path, mode="w" if start == 0 else "a", header=start == 0, index_label="")


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    adata = load_input(args.input, args.metadata)
    adata.var_names_make_unique()
    adata.layers["counts"] = get_counts(adata, args.counts_layer)
    adata.X = adata.layers["counts"].copy()
    print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

    # ---- Cell QC ----
    adata.var["mt"] = adata.var_names.str.match(args.mito_pattern)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[(adata.obs["n_genes_by_counts"] > args.min_genes_per_cell)
                  & (adata.obs["pct_counts_mt"] < args.max_pct_mito)].copy()

    # ---- Gene filtering ----
    sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)

    # ---- Optional: drop rare cell types ----
    if args.label_key is not None:
        if args.label_key not in adata.obs:
            raise SystemExit(f"--label-key '{args.label_key}' is not a column of adata.obs")
        props = adata.obs[args.label_key].value_counts(normalize=True)
        keep = props.index[props >= args.min_celltype_frac]
        adata = adata[adata.obs[args.label_key].isin(keep)].copy()
    sc.pp.filter_genes(adata, min_counts=1)  # cell filtering can leave all-zero genes
    print(f"After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    # ---- Library sizes + dispersions (all genes, before HVG selection, as in preprocess.R) ----
    X = sp.csr_matrix(adata.layers["counts"])
    lib_sizes = np.asarray(X.sum(axis=1)).ravel().astype(float)
    dispersions = estimate_dispersions(X, lib_sizes)

    # ---- Highly variable genes ----
    sc.pp.highly_variable_genes(
        adata, n_top_genes=min(args.n_hvgs, adata.n_vars), flavor="seurat_v3",
        layer="counts", batch_key=args.batch_key,
    )
    hvg = adata.var["highly_variable"].to_numpy()
    print(f"Selected {hvg.sum()} highly variable genes")

    # ---- Write outputs ----
    cell_ids = adata.obs_names.to_numpy()
    gene_names = adata.var_names[hvg].to_numpy()
    write_counts_csv(X[:, hvg], cell_ids, gene_names, os.path.join(args.out_dir, "counts.csv"))
    pd.DataFrame({"cell_id": cell_ids, "tmm_lib_size": lib_sizes}).to_csv(
        os.path.join(args.out_dir, "library_sizes.csv"), index=False)
    pd.DataFrame({"gene": gene_names, "dispersion": dispersions[hvg]}).to_csv(
        os.path.join(args.out_dir, "dispersion.csv"), index=False)

    if args.label_key is not None:
        labels = adata.obs[args.label_key].astype("category")
        codes = labels.cat.codes.to_numpy() + 1  # 1-based, like as.integer(as.factor()) in R
        clusters = pd.DataFrame({"cell_id": cell_ids, "celltype": codes})
        if args.batch_key is not None:
            clusters["batch_id"] = adata.obs[args.batch_key].astype("category").cat.codes.to_numpy()
        clusters.to_csv(os.path.join(args.out_dir, "clusters.csv"), index=False)
        pd.DataFrame({"celltype": np.arange(1, len(labels.cat.categories) + 1),
                      "label": labels.cat.categories}).to_csv(
            os.path.join(args.out_dir, "celltype_key.csv"), index=False)
    else:
        print("No --label-key: no clusters.csv written, so run scFLAME with --n-clusters.")
        if args.batch_key is not None:
            print("Note: batch_id is stored in clusters.csv, which needs --label-key; batch not written.")

    adata.write_h5ad(os.path.join(args.out_dir, "processed.h5ad"))
    print(f"Done. {adata.n_obs} cells x {hvg.sum()} genes written to {args.out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare scFLAME input files from an .h5ad file or 10x output.")
    p.add_argument("--input", "--adata", dest="input", required=True,
                   help="An .h5ad file, a 10x .h5 file, or a 10x matrix / Cell Ranger outs directory")
    p.add_argument("--metadata", default=None,
                   help="Optional CSV of per-cell metadata (first column = barcode), e.g. labels or batch")
    p.add_argument("--out-dir", required=True, help="Directory for counts.csv etc. (the --data-dir for scFLAME)")
    p.add_argument("--counts-layer", default=None,
                   help="Layer holding raw counts, e.g. 'counts' (default: adata.X; ignored for 10x, "
                        "which is always raw counts)")
    p.add_argument("--label-key", default=None,
                   help="adata.obs column with cell-type labels, if you have them")
    p.add_argument("--batch-key", default=None, help="adata.obs column with batch, for batch correction")
    p.add_argument("--mito-pattern", default="^MT-", help="Regex for mitochondrial genes ('^mt-' for mouse)")
    p.add_argument("--min-genes-per-cell", type=int, default=200)
    p.add_argument("--max-pct-mito", type=float, default=10)
    p.add_argument("--min-cells-per-gene", type=int, default=10)
    p.add_argument("--min-celltype-frac", type=float, default=0.01,
                   help="Drop cell types below this fraction of cells (only with --label-key)")
    p.add_argument("--n-hvgs", type=int, default=5000, help="Number of highly variable genes to write")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())