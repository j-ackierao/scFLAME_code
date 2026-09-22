## ============================================================
## preprocess_zeisel.R
## Zeisel mouse brain cortex. Data from scRNAseq package.
## QC steps adapted from the OSCA workflow book:
## https://bioconductor.org/books/3.13/OSCA.workflows/zeisel-mouse-brain-strt-seq.html
## (Amezquita et al., "Orchestrating Single-Cell Analysis with Bioconductor",
## Nature Methods 2020; OSCA book content licensed CC-BY 4.0)
## ============================================================

suppressMessages({
  library(scRNAseq)
  library(scater)
  library(scran)
  library(Seurat)
  library(edgeR)
})

set.seed(42)

outdir <- "data"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

## -----------------------------
## Load + collapse duplicate gene loci
## -----------------------------
sce.zeisel <- ZeiselBrainData()
sce.zeisel <- aggregateAcrossFeatures(
  sce.zeisel, id = sub("_loc[0-9]+$", "", rownames(sce.zeisel))
)

## -----------------------------
## QC filtering (spike-in + mito based)
## -----------------------------
stats <- perCellQCMetrics(sce.zeisel, subsets = list(
  Mt = rowData(sce.zeisel)$featureType == "mito"
))
qc <- quickPerCellQC(stats, percent_subsets = c("altexps_ERCC_percent", "subsets_Mt_percent"))
sce.zeisel <- sce.zeisel[, !qc$discard]

## -----------------------------
## Build Seurat object for downstream use / saving
## -----------------------------
seurat_zeisel <- CreateSeuratObject(counts = counts(sce.zeisel))
seurat_zeisel$celltype <- colData(sce.zeisel)$level1class
seurat_zeisel$celltype_fine <- colData(sce.zeisel)$level2class

## -----------------------------
## TMM normalization + dispersion
## -----------------------------
counts_mat <- GetAssayData(seurat_zeisel, layer = "counts")
dge <- DGEList(counts = counts_mat)
dge <- calcNormFactors(dge, method = "TMM")
effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
dge <- estimateDisp(dge)

## -----------------------------
## Write outputs
## -----------------------------
write.csv(
  data.frame(cell_id = colnames(counts_mat), tmm_lib_size = effective_lib_size),
  file.path(outdir, "zeisel_tmm_effective_library_sizes.csv"), row.names = FALSE
)
write.csv(
  data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
  file.path(outdir, "zeisel_edgeR_tagwise_dispersion.csv"), row.names = FALSE
)
write.csv(
  t(as.matrix(counts_mat)),
  file.path(outdir, "zeisel_rawcounts.csv")
)
write.csv(
  data.frame(
    cell_id = colnames(seurat_zeisel),
    cluster_id = as.integer(as.factor(seurat_zeisel$celltype)),
    celltype = seurat_zeisel$celltype,
    celltype_fine = seurat_zeisel$celltype_fine
  ),
  file.path(outdir, "zeisel_clusters.csv"), row.names = FALSE
)
saveRDS(seurat_zeisel, file.path(outdir, "seurat_zeisel_processed.rds"))

message("Done. ", ncol(seurat_zeisel), " cells x ", nrow(seurat_zeisel), " genes")