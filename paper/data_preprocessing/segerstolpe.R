## ============================================================
## segerstolpe.R
## Segerstolpe pancreas: produces the model-input files and a saved Seurat object.
## Also caches the QC'd SCE as seger.rds, which is the file simulations
## by generate_data_splat.R / generate_data_zinbwave.R expects as input.
##
## QC and normalization steps adapted from the OSCA workflow book:
## https://bioconductor.org/books/3.21/OSCA.workflows/segerstolpe-human-pancreas-smart-seq2.html
## (Amezquita et al., "Orchestrating Single-Cell Analysis with Bioconductor",
## Nature Methods 2020; OSCA book content licensed CC-BY 4.0)
## ============================================================

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scater)
  library(scran)
  library(scRNAseq)
  library(AnnotationHub)
  library(Seurat)
  library(edgeR)
})

outdir <- "data"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
min_celltype_frac <- 0.025 

## -----------------------------
## Load + map gene symbols to Ensembl IDs
## -----------------------------
sce.seger <- SegerstolpePancreasData()
edb <- AnnotationHub()[["AH73881"]]
symbols <- rowData(sce.seger)$symbol
ens.id <- mapIds(edb, keys = symbols, keytype = "SYMBOL", column = "GENEID")
ens.id <- ifelse(is.na(ens.id), symbols, ens.id)

keep <- !duplicated(ens.id)
sce.seger <- sce.seger[keep, ]
rownames(sce.seger) <- ens.id[keep]

## -----------------------------
## Tidy metadata + cell-type labels
## -----------------------------
emtab.meta <- colData(sce.seger)[, c("cell type", "disease", "individual", "single cell well quality")]
colnames(emtab.meta) <- c("CellType", "Disease", "Donor", "Quality")
colData(sce.seger) <- emtab.meta

sce.seger$CellType <- gsub(" cell", "", sce.seger$CellType)
sce.seger$CellType <- paste0(
  toupper(substr(sce.seger$CellType, 1, 1)),
  substring(sce.seger$CellType, 2)
)

## -----------------------------
## QC: drop low-quality cells (author-flagged) and outliers
## (donors H5/H6 excluded from the outlier reference set - low/no
## spike-ins, per the OSCA book)
## -----------------------------
low.qual <- sce.seger$Quality != "OK"

stats <- perCellQCMetrics(sce.seger)
qc <- quickPerCellQC(
  stats,
  percent_subsets = "altexps_ERCC_percent",
  batch = sce.seger$Donor,
  subset = !sce.seger$Donor %in% c("H6", "H5")
)
sce.seger <- sce.seger[, !(qc$discard | low.qual)]

## -----------------------------
## Drop rare cell types
## -----------------------------
cell_counts <- table(sce.seger$CellType)
cell_props <- prop.table(cell_counts)
keep_types <- names(cell_props)[cell_props >= min_celltype_frac]
sce.seger <- sce.seger[, sce.seger$CellType %in% keep_types]
sce.seger$CellType <- factor(sce.seger$CellType)

## -----------------------------
## Build + save Seurat object
## -----------------------------
seurat.seger <- CreateSeuratObject(counts = counts(sce.seger))
seurat.seger$celltype <- sce.seger$CellType
saveRDS(seurat.seger, file.path(outdir, "seger.rds")) # Used in generating simulation data

## -----------------------------
## TMM normalization + dispersion
## -----------------------------
dge <- DGEList(counts = counts(sce.seger))
dge <- calcNormFactors(dge, method = "TMM")
effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
dge <- estimateDisp(dge)

## -----------------------------
## Write outputs
## -----------------------------
write.csv(t(counts(sce.seger)), file.path(outdir, "seger_rawcounts.csv"))

write.csv(
  data.frame(cell_id = colnames(sce.seger), tmm_lib_size = effective_lib_size),
  file.path(outdir, "seger_tmm_effective_library_sizes.csv"), row.names = FALSE
)
write.csv(
  data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
  file.path(outdir, "seger_edgeR_tagwise_dispersion.csv"), row.names = FALSE
)
write.csv(
  data.frame(cell_id = colnames(sce.seger), celltype = as.integer(as.factor(sce.seger$CellType))),
  file.path(outdir, "seger_clusters.csv"), row.names = FALSE
)

message("Done. ", ncol(sce.seger), " cells x ", nrow(sce.seger), " genes")