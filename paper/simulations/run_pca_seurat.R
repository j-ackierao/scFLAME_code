## ============================================================
## run_pca_seurat.R
## Benchmarks Seurat (Louvain) (fixed and tuned K), PCA + k-means, 
## on one simulated dataset with *_rawcounts.csv / *_clusters.csv
## ============================================================

suppressPackageStartupMessages({
  library(argparse)
  library(Matrix)
  library(Seurat)
  library(SingleCellExperiment)
  library(mclust)
  library(aricode)
  library(clue)
})

cluster_accuracy <- function(y_true, y_pred) {
  tab <- table(y_true, y_pred)
  n <- max(nrow(tab), ncol(tab))
  tab_sq <- matrix(0, n, n)
  tab_sq[1:nrow(tab), 1:ncol(tab)] <- tab
  assignment <- solve_LSAP(tab_sq, maximum = TRUE)
  sum(tab_sq[cbind(1:n, assignment)]) / length(y_true)
}
 
## Load a simulated dataset from its raw-counts/clusters CSV pair.
load_sim_dataset <- function(counts_path, clusters_path) {
  df <- read.csv(counts_path, row.names = 1)          # cells x genes, as written by the simulation scripts
  counts_mat <- as.matrix(df)
  storage.mode(counts_mat) <- "numeric"
  counts_mat <- t(counts_mat)                          # -> genes x cells
 
  clusters_df <- read.csv(clusters_path)
  true_labels <- as.factor(clusters_df$celltype)
 
  sce <- SingleCellExperiment(assays = list(counts = counts_mat))
  colData(sce)$CellType <- true_labels
 
  list(sce = sce, counts_mat = counts_mat, true_labels = true_labels,
       K_true = length(unique(true_labels)))
}

parser <- ArgumentParser(description = "Baseline clustering benchmark on one simulated dataset")
parser$add_argument("--counts_path", type = "character", required = TRUE,
                     help = "*_rawcounts.csv, cells x genes")
parser$add_argument("--clusters_path", type = "character", required = TRUE,
                     help = "*_clusters.csv with a `celltype` column")
parser$add_argument("--out_csv", type = "character", required = TRUE)
parser$add_argument("--replicate", type = "integer", default = NA,
                     help = "Dataset index (task_id); only used for the top-level seed. Defaults to $SLURM_ARRAY_TASK_ID - 1")
opt <- parser$parse_args()

task_id <- if (is.na(opt$replicate)) {
  as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", unset = "0")) - 1
} else {
  opt$replicate
}
set.seed(42 + task_id)

n_reps <- 3
n_pcs  <- 20

## -----------------------------
## LOAD DATA
## -----------------------------
dat <- load_sim_dataset(opt$counts_path, opt$clusters_path)
counts_mat  <- dat$counts_mat
true_labels <- dat$true_labels
K_true      <- dat$K_true

dir.create(dirname(opt$out_csv), recursive = TRUE, showWarnings = FALSE)

## -----------------------------
## RUN METHODS
## -----------------------------
results <- list()

for (rep in seq_len(n_reps)) {

  set.seed(42 + rep)
  message("rep ", rep)

  ## =============================
  ## 1. Seurat (Louvain)
  ## =============================
  seu <- CreateSeuratObject(counts = counts_mat)

  seu <- NormalizeData(seu)
  seu <- FindVariableFeatures(seu)
  seu <- ScaleData(seu)
  seu <- RunPCA(seu)

  seu <- FindNeighbors(seu)
  seu <- FindClusters(seu)

  clusters <- Idents(seu)

  ari <- adjustedRandIndex(clusters, true_labels)
  nmi <- NMI(clusters, true_labels)
  acc <- cluster_accuracy(true_labels, clusters)

  results[[length(results) + 1]] <- data.frame(
    rep = rep,
    method = "seurat_louvain",
    resolution = NA,
    K_true = K_true,
    K_final = length(unique(clusters)),
    ari = ari,
    nmi = nmi,
    accuracy = acc
  )

  ## =============================
  ## 2. PCA + KMEANS
  ## =============================
  log_counts <- log1p(t(counts_mat))  # cells x genes

  pca <- prcomp(log_counts, rank. = n_pcs)
  emb <- pca$x

  set.seed(42 + rep)
  km <- kmeans(emb, centers = K_true, nstart = 10)
  clusters <- km$cluster

  ari <- adjustedRandIndex(clusters, true_labels)
  nmi <- NMI(clusters, true_labels)
  acc <- cluster_accuracy(true_labels, clusters)

  results[[length(results) + 1]] <- data.frame(
    rep = rep,
    method = "pca_kmeans",
    resolution = NA,
    K_true = K_true,
    K_final = length(unique(clusters)),
    ari = ari,
    nmi = nmi,
    accuracy = acc
  )

  ## =============================
  ## 3. Seurat (tuned resolution for K_true)
  ## =============================
  seu_k <- CreateSeuratObject(counts = counts_mat)

  seu_k <- NormalizeData(seu_k)
  seu_k <- FindVariableFeatures(seu_k)
  seu_k <- ScaleData(seu_k)
  seu_k <- RunPCA(seu_k)
  seu_k <- FindNeighbors(seu_k)

  # grid of resolutions to try
  res_grid <- seq(0.1, 2, by = 0.1)

  best_clusters <- NULL
  best_diff <- Inf
  best_res <- NA

  for (res in res_grid) {
    tmp <- FindClusters(seu_k, resolution = res, verbose = FALSE)
    clusters_tmp <- Idents(tmp)
    k_tmp <- length(unique(clusters_tmp))

    diff <- abs(k_tmp - K_true)

    if (diff < best_diff) {
      best_diff <- diff
      best_clusters <- clusters_tmp
      best_res <- res
    }

    # perfect match -> stop early
    if (k_tmp == K_true) break
  }

  clusters <- best_clusters

  ari <- adjustedRandIndex(clusters, true_labels)
  nmi <- NMI(clusters, true_labels)
  acc <- cluster_accuracy(true_labels, clusters)

  results[[length(results) + 1]] <- data.frame(
    rep = rep,
    method = "seurat_tunedK",
    resolution = best_res,
    K_true = K_true,
    K_final = length(unique(clusters)),
    ari = ari,
    nmi = nmi,
    accuracy = acc
  )
}

## -----------------------------
## SAVE
## -----------------------------
summary_df <- do.call(rbind, results)
write.csv(summary_df, opt$out_csv, row.names = FALSE)

message("Saved: ", opt$out_csv)