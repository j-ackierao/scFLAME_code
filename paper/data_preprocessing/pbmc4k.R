## ============================================================
## preprocess_pbmc_real.R
## PBMC 4k (10x Genomics).
## Cluster labels and count matrix can be downloaded from 
## https://www.10xgenomics.com/datasets/4-k-pbm-cs-from-a-healthy-donor-2-standard-2-1-0
## We downloaded 'Gene / cell matrix (filtered)' and 'Clustering analysis' - labels
## from graph based clustering were used for comparison, and we renamed this file 10x_clusters.csv.
## ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(edgeR)
})
 
set.seed(1)
 
outdir <- "data"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
 
archive_path   <- "pbmc4k_filtered_gene_bc_matrices.tar.gz"
data_dir       <- "./filtered_gene_bc_matrices/GRCh38/"
clusters_path  <- "10x_clusters.csv"
min_cells      <- 3     # gene kept if expressed in >= this many cells (matches original 10x paper filtering)
min_features   <- 200   # cell kept if >= this many genes detected
max_pct_mito   <- 5
n_hvgs         <- 5000
 
## -----------------------------
## Load
## -----------------------------
untar(archive_path)
pbmc.data <- Read10X(data.dir = data_dir)
 
pbmc <- CreateSeuratObject(
  counts = pbmc.data,
  project = "pbmc4k",
  min.cells = min_cells,
  min.features = min_features
)
 
## -----------------------------
## Cell QC
## -----------------------------
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
pbmc <- subset(pbmc, subset = nFeature_RNA > min_features & percent.mt < max_pct_mito)
 
## -----------------------------
## Attach cluster labels from the external reference file
## -----------------------------
pbmc_truelabels <- read.csv(clusters_path)
rownames(pbmc_truelabels) <- pbmc_truelabels$Barcode
pbmc_truelabels <- pbmc_truelabels[colnames(pbmc), , drop = FALSE]
 
if (any(is.na(pbmc_truelabels$Cluster))) {
  warning(sum(is.na(pbmc_truelabels$Cluster)),
          " cells had no matching barcode in ", clusters_path,
          " - check barcode formatting (e.g. '-1' suffixes) between the two sources.")
}
 
pbmc$CellType <- as.factor(pbmc_truelabels$Cluster)
 
## -----------------------------
## Save Seurat object
## -----------------------------
saveRDS(pbmc, file.path(outdir, "seurat_pbmc_processed.rds"))
 
## -----------------------------
## TMM normalization + dispersion - computed on the full gene set
## -----------------------------
counts_mat <- GetAssayData(pbmc, layer = "counts")
dge <- DGEList(counts = counts_mat)
dge <- calcNormFactors(dge, method = "TMM")
effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
dge <- estimateDisp(dge)

## -----------------------------
## 5000 HVGs: unlike other datasets, we select HVGs after TMM/dispersion here, and before running models
## -----------------------------
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = n_hvgs)
 
## -----------------------------
## Write outputs
## -----------------------------
write.csv(t(as.matrix(counts_mat)), file.path(outdir, "pbmc_rawcounts.csv"))
 
write.csv(
  data.frame(cell_id = colnames(pbmc), tmm_lib_size = effective_lib_size),
  file.path(outdir, "pbmc_tmm_effective_library_sizes.csv"), row.names = FALSE
)
write.csv(
  data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
  file.path(outdir, "pbmc_edgeR_tagwise_dispersion.csv"), row.names = FALSE
)
write.csv(
  data.frame(cell_id = colnames(pbmc), celltype = as.integer(pbmc$CellType)),
  file.path(outdir, "pbmc_clusters.csv"), row.names = FALSE
)
 
message("Done. ", ncol(pbmc), " cells x ", nrow(pbmc), " genes")