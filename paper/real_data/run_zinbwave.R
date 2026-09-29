library(Matrix)
library(Seurat)
library(SingleCellExperiment)
library(mclust)
library(aricode)
library(clue)
library(zinbwave)
library(matrixStats)
library(BiocParallel)

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

  ## -----------------------------
  ## Fit ZINB-WaVE
  ## -----------------------------

  vars <- rowVars(log1p(counts_mat))

  hvg <- names(
    sort(vars, decreasing = TRUE)[1:1000]
  )

  sce_sub <- sce[hvg, ]

  message("Fitting ZINB-WaVE...")

  sce_zinb <- zinbwave(
    sce_sub,
    K = n_pcs,
    epsilon = 1000,
    BPPARAM = bp,
    verbose = FALSE
  )

  latent_W <- reducedDim(
    sce_zinb,
    "zinbwave"
  )

  ## -----------------------------
  ## ZINB-WaVE + K-Means
  ## -----------------------------

  message("Clustering: KMeans")

  set.seed(42 + rep)

  km_zinb <- kmeans(
    latent_W,
    centers = K_true,
    nstart = 25
  )

  clusters_km <- km_zinb$cluster

  results[[length(results) + 1]] <- data.frame(
    rep = rep,
    method = "zinbwave_kmeans",
    resolution = NA,
    K_true = K_true,
    K_final = length(unique(clusters_km)),
    ari = adjustedRandIndex(
      clusters_km,
      true_labels
    ),
    nmi = NMI(
      clusters_km,
      true_labels
    ),
    accuracy = cluster_accuracy(
      true_labels,
      clusters_km
    )
  )

  if (rep == 1) {
    clusters_km_first <- clusters_km
    latent_first <- latent_W
  }
}

## -----------------------------
## SAVE
## -----------------------------
summary_df <- do.call(rbind, results)
write.csv(
    summary_df, 
    file.path(out_dir, "zinbwave_summary.csv"), 
    row.names = FALSE
)

## -----------------------------
## Save latent space from first repeat
## -----------------------------
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

latent_df <- as.data.frame(latent_first)
rownames(latent_df) <- colnames(sce)
colnames(latent_df) <- paste0("z", seq_len(ncol(latent_df)))

write.csv(
  latent_df,
  file.path(out_dir, "zinbwave_latents.csv"),
  row.names = TRUE
)

## -----------------------------
## Save clusters from first repeat
## -----------------------------
cluster_km_df <- data.frame(
  cell_id = colnames(sce),
  cluster = clusters_km_first
)

write.csv(
  cluster_km_df,
  file.path(out_dir, "zinbwave_kmeans_clusters.csv"),
  row.names = FALSE
)
