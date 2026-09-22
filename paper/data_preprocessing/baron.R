## ============================================================
## baron.R
## Baron pancreas: produces the model-input files and a saved Seurat object.
## Note: this is distinct from the simulation-side preprocess_baron.R
## (a function sourced by simulate_splat.R / simulate_zinbwave.R),
## which applies an additional variance-based HVG cutoff. This script is for
## real-data files that feed the model directly.
## ============================================================

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scRNAseq)
  library(scater)
  library(Seurat)
  library(edgeR)
})

outdir <- "processed"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
min_cells_per_gene <- 10     # genes expressed in fewer than this many cells are excluded
min_celltype_frac <- 0.02    # cell types below this fraction of all cells are excluded

## -----------------------------
## Load + subset to donor GSM2230757
## -----------------------------
sce <- BaronPancreasData(which = "human")
sce_donor1 <- sce[, colData(sce)$donor == "GSM2230757"]
sce_donor1 <- sce_donor1[rowSums(counts(sce_donor1)) > 10, ]

## -----------------------------
## Cell QC
## -----------------------------
stats <- perCellQCMetrics(sce_donor1)
qc <- quickPerCellQC(stats)
sce_donor1 <- sce_donor1[, !qc$discard]

## -----------------------------
## Gene filtering
## -----------------------------
keep_genes <- rowSums(counts(sce_donor1) > 0) >= min_cells_per_gene
sce_donor1 <- sce_donor1[keep_genes, ]

## -----------------------------
## Drop rare cell types (<2% of all cells)
## -----------------------------
cell_counts <- table(sce_donor1$label)
cell_props <- prop.table(cell_counts)
valid_labels <- names(cell_props)[cell_props >= min_celltype_frac]
sce_filtered <- sce_donor1[, sce_donor1$label %in% valid_labels]
sce_filtered$label <- factor(sce_filtered$label)

## -----------------------------
## Build + save Seurat object
## -----------------------------
seurat_baron <- CreateSeuratObject(counts = counts(sce_filtered))
seurat_baron$celltype <- sce_filtered$label
saveRDS(seurat_baron, file.path(outdir, "seurat_baron_processed.rds"))

## -----------------------------
## TMM normalization + dispersion
## -----------------------------
dge <- DGEList(counts = counts(sce_filtered))
dge <- calcNormFactors(dge, method = "TMM")
effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
dge <- estimateDisp(dge)

## -----------------------------
## Write outputs
## -----------------------------
write.csv(t(counts(sce_filtered)), file.path(outdir, "baron_rawcounts.csv"))

write.csv(
  data.frame(cell_id = colnames(sce_filtered), tmm_lib_size = effective_lib_size),
  file.path(outdir, "baron_tmm_effective_library_sizes.csv"), row.names = FALSE
)
write.csv(
  data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
  file.path(outdir, "baron_edgeR_tagwise_dispersion.csv"), row.names = FALSE
)
write.csv(
  data.frame(cell_id = colnames(sce_filtered), celltype = as.integer(as.factor(sce_filtered$label))),
  file.path(outdir, "baron_clusters.csv"), row.names = FALSE
)

message("Done. ", ncol(sce_filtered), " cells x ", nrow(sce_filtered), " genes")