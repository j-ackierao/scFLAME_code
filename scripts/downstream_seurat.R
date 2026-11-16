## ============================================================
## downstream_seurat.R
## Bring scFLAME results into Seurat for analysis of the clustering:
##   1. pick the best repeat (highest ELBO)
##   2. add scFLAME clusters to the metadata
##   3. add the scFLAME latent factors as a reduction (like PCA) and run UMAP on them
##   4. compare scFLAME clusters with an existing annotation (if there is one)
##   5. dot plot of scFLAME's top marker genes per cluster
##
## Run after:  preprocess.R  ->  python scripts/run_realdata.py --save-top-genes ...
## ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(ggplot2)
})

## -----------------------------
## Configuration
## -----------------------------
seurat_path   <- "data/baron/processed.rds"  # Seurat object written by preprocess.R (or your own)
results_dir   <- "results/baron"             # --out-dir used for run_realdata.py
repeat_to_use <- NULL                        # NULL = repeat with the highest ELBO; or e.g. 2
label_column  <- "celltype"                  # existing annotation to compare against, or NULL
n_top_genes   <- 5                           # marker genes per cluster in the dot plot
plot_dir      <- file.path(results_dir, "plots")

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
seu <- readRDS(seurat_path)
if (!is.null(label_column) && !label_column %in% colnames(seu@meta.data)) {
  warning("label_column '", label_column, "' not in the Seurat metadata; skipping comparisons.")
  label_column <- NULL
}

## Put a results table (with a cell_id column) in the same cell order as the Seurat object
align_to_seurat <- function(df, what) {
  idx <- match(colnames(seu), df$cell_id)
  if (!anyNA(idx)) return(df[idx, , drop = FALSE])
  if (nrow(df) == ncol(seu)) {
    warning(what, ": cell IDs don't match the Seurat object, assuming the same cell order. ",
            "Check that the Seurat object is the one counts.csv was written from.")
    return(df)
  }
  stop(what, ": ", sum(is.na(idx)), " Seurat cells are missing from the scFLAME output ",
       "(", nrow(df), " cells vs ", ncol(seu), " in Seurat). ",
       "Use the Seurat object counts.csv was written from.")
}

## -----------------------------
## 1. Pick a repeat
## -----------------------------
## Chosen by the run with the highest ELBO.
summary_df <- read.csv(file.path(results_dir, "scflame_summary.csv"))
summary_df$rep <- summary_df[["repeat"]] + 1           # summary is 0-indexed, files are rep1, rep2, ...
if (is.null(repeat_to_use)) repeat_to_use <- summary_df$rep[which.max(summary_df$elbo_final)]
message("Using repeat ", repeat_to_use, " (ELBO = ",
        signif(summary_df$elbo_final[summary_df$rep == repeat_to_use], 6), ")")
print(summary_df)

## -----------------------------
## 2. Clusters
## -----------------------------
alloc <- align_to_seurat(read.csv(file.path(results_dir, "scflame_allocations.csv"),
                                  check.names = FALSE), "scflame_allocations.csv")
pred_col <- paste0("rep", repeat_to_use, "_pred")

seu$scflame_cluster <- factor(alloc[[pred_col]], levels = sort(unique(alloc[[pred_col]])))

## -----------------------------
## 3. Latent factors -> reduction -> UMAP
## -----------------------------
latent_file <- file.path(results_dir, sprintf("scflame_latents_rep%d.csv", repeat_to_use))
latents <- align_to_seurat(read.csv(latent_file, check.names = FALSE), basename(latent_file))
emb <- as.matrix(latents[, setdiff(colnames(latents), "cell_id")])
colnames(emb) <- paste0("scFLAME_", seq_len(ncol(emb)))
rownames(emb) <- colnames(seu)

seu[["scflame_latent"]] <- CreateDimReducObject(
  embeddings = emb, key = "scFLAME_", assay = DefaultAssay(seu)
)
seu <- RunUMAP(seu, reduction = "scflame_latent", dims = seq_len(ncol(emb)),
               reduction.name = "scflame_umap", reduction.key = "scflameUMAP_", verbose = FALSE)

p_clusters <- DimPlot(seu, reduction = "scflame_umap", group.by = "scflame_cluster",
                      label = TRUE, repel = TRUE) + ggtitle("scFLAME clusters")
if (!is.null(label_column)) {
  p_labels <- DimPlot(seu, reduction = "scflame_umap", group.by = label_column,
                      label = TRUE, repel = TRUE) + ggtitle(label_column)
  ggsave(file.path(plot_dir, "umap_clusters_vs_labels.pdf"), p_clusters + p_labels,
         width = 14, height = 6)
} else {
  ggsave(file.path(plot_dir, "umap_clusters.pdf"), p_clusters, width = 7.5, height = 6)
}

## -----------------------------
## 4. Compare with an existing annotation
## -----------------------------
## Each column (scFLAME cluster) shows how its cells split across the existing labels.
if (!is.null(label_column)) {
  comp <- as.data.frame(table(label = seu@meta.data[[label_column]],
                              cluster = seu$scflame_cluster)) %>%
    group_by(cluster) %>%
    mutate(frac = Freq / sum(Freq)) %>%
    ungroup()
  p_comp <- ggplot(comp, aes(cluster, label, fill = frac)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = ifelse(Freq > 0, Freq, "")), size = 2.5) +
    scale_fill_gradient(low = "white", high = "#B30000", name = "Fraction of\ncluster") +
    labs(x = "scFLAME cluster", y = label_column) +
    theme_minimal()
  ggsave(file.path(plot_dir, "cluster_vs_label_heatmap.pdf"), p_comp, width = 8, height = 6)
}

## -----------------------------
## 5. Top marker genes per cluster
## -----------------------------
## scFLAME scores each gene by how much a cluster's latent centre shifts that gene's expression
## away from the average cell. The dot plot then shows the observed expression of those genes.
genes_file <- file.path(results_dir, "scflame_top_genes.csv")
if (file.exists(genes_file)) {
  top_genes <- read.csv(genes_file, check.names = FALSE)
  if (!all(c("cluster", "gene") %in% colnames(top_genes))) {
    stop("Expected 'cluster' and 'gene' columns in scflame_top_genes.csv; found: ",
         paste(colnames(top_genes), collapse = ", "), ". Edit the column names here.")
  }
  if ("repeat" %in% colnames(top_genes)) top_genes <- top_genes[top_genes[["repeat"]] == repeat_to_use, ]
  top_genes <- top_genes %>% group_by(cluster) %>% slice_head(n = n_top_genes) %>% ungroup()

  genes <- unique(top_genes$gene)
  missing <- setdiff(genes, rownames(seu))
  if (length(missing) > 0) {
    message(length(missing), " marker genes aren't in the Seurat object and are dropped: ",
            paste(head(missing, 10), collapse = ", "))
  }
  genes <- intersect(genes, rownames(seu))

  seu <- NormalizeData(seu, verbose = FALSE)   # preprocess.R saves raw counts only
  p_dot <- DotPlot(seu, features = genes, group.by = "scflame_cluster") +
    RotatedAxis() + labs(x = NULL, y = "scFLAME cluster")
  ggsave(file.path(plot_dir, "top_genes_dotplot.pdf"), p_dot,
         width = max(8, 0.25 * length(genes) + 3), height = 5, limitsize = FALSE)
} else {
  message("No scflame_top_genes.csv found; rerun run_realdata.py with --save-top-genes for the dot plot.")
}

saveRDS(seu, file.path(results_dir, "seurat_with_scflame.rds"))
message("Plots in ", plot_dir, "; Seurat object saved to ",
        file.path(results_dir, "seurat_with_scflame.rds"))