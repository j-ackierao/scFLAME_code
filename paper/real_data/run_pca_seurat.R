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

source("load_data.R")

cluster_accuracy <- function(y_true, y_pred) {
  tab <- table(y_true, y_pred)
  n <- max(nrow(tab), ncol(tab))
  tab_sq <- matrix(0, n, n)
  tab_sq[1:nrow(tab), 1:ncol(tab)] <- tab
  assignment <- solve_LSAP(tab_sq, maximum = TRUE)
  sum(tab_sq[cbind(1:n, assignment)]) / length(y_true)
}

## -----------------------------
## Config
## -----------------------------

counts_path <- "rawcounts.csv" # change to relevant raw counts file
clusters_path <- "clusters.csv" # change to relevant clusters file

n_reps <- 5
n_pcs <- 20

out_dir <- "real_data/results"

## -----------------------------
## Load data
## -----------------------------

data <- load_segerstolpe( #Change to desired function
  counts_path = counts_path,
  clusters_path = clusters_path
)

counts_mat <- data$counts_mat
true_labels <- data$true_labels
sce <- data$sce
K_true <- data$K_true


## -----------------------------
## Run
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
  res_grid <- seq(0.05, 1, by = 0.05)

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
write.csv(summary_df, paste0(out_dir, "pca_seurat_summary.csv"), row.names = FALSE)

message("Saved: ", paste0(out_dir, "pca_seurat_summary.csv"))