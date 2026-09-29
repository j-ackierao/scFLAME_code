## ============================================================
## Dataset for real-data scFLAME comparisons (R functions)
## ============================================================
##
## Each loader returns:
##
##   $counts_mat   genes x cells raw count matrix
##   $true_labels  factor vector aligned with cells
##   $sce          SingleCellExperiment object
##   $K_true       number of true cell types
##   $genes        gene names
##   $cells        cell names
##
## ============================================================


## ------------------------------------------------------------
## Helper: select highly variable genes
## ------------------------------------------------------------

.select_hvgs <- function(counts_mat, n_hvgs) {

  log_mat <- log1p(counts_mat)

  gene_vars <- apply(log_mat, 1, var)

  ## Highest-variance genes
  hvg_idx <- order(
    gene_vars,
    decreasing = TRUE
  )[seq_len(min(n_hvgs, length(gene_vars)))]

  counts_mat[hvg_idx, , drop = FALSE]
}


## ------------------------------------------------------------
## Helper: construct SingleCellExperiment
## ------------------------------------------------------------

.make_sce <- function(counts_mat, true_labels) {

  ## Make absolutely sure labels are a factor
  true_labels <- as.factor(true_labels)

  ## Check that the number of labels matches cells
  if (length(true_labels) != ncol(counts_mat)) {
    stop(
      "Number of true labels (",
      length(true_labels),
      ") does not match number of cells (",
      ncol(counts_mat),
      ")."
    )
  }

  sce <- SingleCellExperiment(
    assays = list(
      counts = counts_mat
    )
  )

  colData(sce)$CellType <- true_labels

  sce
}


## ============================================================
## PBMC 4k
## ============================================================

load_pbmc4k <- function(counts_path,
                        clusters_path,
                        n_hvgs = NULL) {

  df <- read.csv(
    counts_path,
    row.names = 1,
    check.names = FALSE
  )

  counts_mat <- t(as.matrix(df))

  storage.mode(counts_mat) <- "numeric"

  if (!is.null(n_hvgs)) {
    counts_mat <- .select_hvgs(
      counts_mat,
      n_hvgs
    )
  }

  clusters_df <- read.csv(
    clusters_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  cell_idx <- match(
    colnames(counts_mat),
    clusters_df$Barcode
  )

  if (any(is.na(cell_idx))) {
    stop(
      "Some cells in counts_path were not found in clusters_path."
    )
  }

  true_labels <- as.factor(
    clusters_df$Cluster[cell_idx]
  )

  sce <- .make_sce(
    counts_mat,
    true_labels
  )

  K_true <- length(unique(true_labels))

  list(
    counts_mat = counts_mat,
    true_labels = true_labels,
    sce = sce,
    K_true = K_true,
    genes = rownames(counts_mat),
    cells = colnames(counts_mat)
  )
}


## ============================================================
## Mouse uterus
## ============================================================

load_uterus <- function(counts_path,
                        clusters_path,
                        n_hvgs = 5000) {

  ## -----------------------------
  ## Counts
  ## -----------------------------

  df <- read.csv(
    counts_path,
    row.names = 1,
    check.names = FALSE
  )

  counts_mat <- t(as.matrix(df))

  storage.mode(counts_mat) <- "numeric"

  counts_mat <- .select_hvgs(
    counts_mat,
    n_hvgs
  )

  clusters_df <- read.csv(
    clusters_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  cell_idx <- match(
    colnames(counts_mat),
    clusters_df$cell_id
  )

  if (any(is.na(cell_idx))) {
    stop(
      "Some cells in counts_path were not found in clusters_path."
    )
  }

  cluster_strings <- as.character(
    clusters_df$cluster_id[cell_idx]
  )

  ## Extract trailing integer
  true_labels <- sub(
    ".*([0-9]+)$",
    "\\1",
    cluster_strings
  )

  true_labels <- as.factor(true_labels)

  sce <- .make_sce(
    counts_mat,
    true_labels
  )

  K_true <- length(unique(true_labels))

  list(
    counts_mat = counts_mat,
    true_labels = true_labels,
    sce = sce,
    K_true = K_true,
    genes = rownames(counts_mat),
    cells = colnames(counts_mat)
  )
}


## ============================================================
## Zeisel mouse brain
## ============================================================

load_zeisel <- function(counts_path,
                        clusters_path,
                        n_hvgs = 5000) {

  df <- read.csv(
    counts_path,
    row.names = 1,
    check.names = FALSE
  )

  counts_mat <- t(as.matrix(df))

  storage.mode(counts_mat) <- "numeric"

  counts_mat <- .select_hvgs(
    counts_mat,
    n_hvgs
  )

  clusters_df <- read.csv(
    clusters_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  cell_idx <- match(
    colnames(counts_mat),
    clusters_df$cell_id
  )

  if (any(is.na(cell_idx))) {
    stop(
      "Some cells in counts_path were not found in clusters_path."
    )
  }

  true_labels <- as.factor(
    clusters_df$celltype[cell_idx]
  )

  sce <- .make_sce(
    counts_mat,
    true_labels
  )

  K_true <- length(unique(true_labels))

  list(
    counts_mat = counts_mat,
    true_labels = true_labels,
    sce = sce,
    K_true = K_true,
    genes = rownames(counts_mat),
    cells = colnames(counts_mat)
  )
}


## ============================================================
## TM spleen 10X / SS2
## ============================================================

load_tm_spleen <- function(counts_path,
                           clusters_path,
                           n_hvgs = 8000) {

  df <- read.csv(
    counts_path,
    row.names = 1,
    check.names = FALSE
  )

  counts_mat <- t(as.matrix(df))

  storage.mode(counts_mat) <- "numeric"

  counts_mat <- .select_hvgs(
    counts_mat,
    n_hvgs
  )

  clusters_df <- read.csv(
    clusters_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  cell_idx <- match(
    colnames(counts_mat),
    clusters_df$cell_id
  )

  if (any(is.na(cell_idx))) {
    stop(
      "Some cells in counts_path were not found in clusters_path."
    )
  }

  true_labels <- as.factor(
    clusters_df$cluster_id[cell_idx]
  )

  sce <- .make_sce(
    counts_mat,
    true_labels
  )

  K_true <- length(unique(true_labels))

  list(
    counts_mat = counts_mat,
    true_labels = true_labels,
    sce = sce,
    K_true = K_true,
    genes = rownames(counts_mat),
    cells = colnames(counts_mat)
  )
}


## ============================================================
## Segerstolpe pancreas
## ============================================================

load_segerstolpe <- function(counts_path,
                             clusters_path,
                             n_hvgs = 5000) {

  df <- read.csv(
    counts_path,
    row.names = 1,
    check.names = FALSE
  )

  counts_mat <- t(as.matrix(df))

  storage.mode(counts_mat) <- "numeric"

  clusters_df <- read.csv(
    clusters_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  true_labels <- as.factor(
    clusters_df$celltype
  )

  celltype_counts <- table(true_labels)

  min_count <- ceiling(
    0.02 * length(true_labels)
  )

  valid_celltypes <- names(
    celltype_counts[
      celltype_counts >= min_count
    ]
  )

  keep_cells <- true_labels %in% valid_celltypes

  cat(
    "Keeping",
    sum(keep_cells),
    "of",
    length(keep_cells),
    "cells\n"
  )

  cat(
    "Remaining cell types:",
    paste(valid_celltypes, collapse = ", "),
    "\n"
  )

  counts_mat <- counts_mat[
    ,
    keep_cells,
    drop = FALSE
  ]

  true_labels <- droplevels(
    true_labels[keep_cells]
  )

  counts_mat <- .select_hvgs(
    counts_mat,
    n_hvgs
  )

  sce <- .make_sce(
    counts_mat,
    true_labels
  )

  K_true <- length(unique(true_labels))

  list(
    counts_mat = counts_mat,
    true_labels = true_labels,
    sce = sce,
    K_true = K_true,
    genes = rownames(counts_mat),
    cells = colnames(counts_mat)
  )
}


## ============================================================
## Baron pancreas
## ============================================================

load_baron <- function(counts_path,
                       clusters_path,
                       n_hvgs = 5000) {

  df <- read.csv(
    counts_path,
    row.names = 1,
    check.names = FALSE
  )

  counts_mat <- t(as.matrix(df))

  storage.mode(counts_mat) <- "numeric"

  clusters_df <- read.csv(
    clusters_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  true_labels <- as.factor(
    clusters_df$celltype
  )

  counts_per_type <- table(
    true_labels
  )

  valid_types <- names(
    counts_per_type[
      counts_per_type >= 50
    ]
  )

  keep_cells <- true_labels %in% valid_types

  counts_mat <- counts_mat[
    ,
    keep_cells,
    drop = FALSE
  ]

  true_labels <- droplevels(
    true_labels[keep_cells]
  )

  counts_mat <- .select_hvgs(
    counts_mat,
    n_hvgs
  )

  sce <- .make_sce(
    counts_mat,
    true_labels
  )

  K_true <- length(unique(true_labels))

  list(
    counts_mat = counts_mat,
    true_labels = true_labels,
    sce = sce,
    K_true = K_true,
    genes = rownames(counts_mat),
    cells = colnames(counts_mat)
  )
}

## ============================================================
## Pancreas
## ============================================================

load_tm_spleen <- function(counts_path,
                           clusters_path,
                           n_hvgs = 8000) {

  df <- read.csv(
    counts_path,
    row.names = 1,
    check.names = FALSE
  )

  counts_mat <- t(as.matrix(df))

  storage.mode(counts_mat) <- "numeric"

  counts_mat <- .select_hvgs(
    counts_mat,
    n_hvgs
  )

  clusters_df <- read.csv(
    clusters_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  cell_idx <- match(
    colnames(counts_mat),
    clusters_df$cell_id
  )

  if (any(is.na(cell_idx))) {
    stop(
      "Some cells in counts_path were not found in clusters_path."
    )
  }

  true_labels <- as.factor(
    clusters_df$cluster_id[cell_idx]
  )

  sce <- .make_sce(
    counts_mat,
    true_labels
  )

  K_true <- length(unique(true_labels))

  list(
    counts_mat = counts_mat,
    true_labels = true_labels,
    sce = sce,
    K_true = K_true,
    genes = rownames(counts_mat),
    cells = colnames(counts_mat)
  )
}