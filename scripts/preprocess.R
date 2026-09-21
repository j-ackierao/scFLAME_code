## ============================================================
## preprocess_template.R
## Generic scRNA-seq preprocessing template: QC filtering, gene
## filtering, optional rare cell-type removal, HVG selection, and
## the standard set of input files scFLAME expects to be run.
## ============================================================

library(Seurat)
library(edgeR)

## -----------------------------
## Configuration
## -----------------------------
counts_path       <- "path/to/counts.rds"   # dgCMatrix, or anything Read10X()/readRDS() returns
metadata_path     <- NULL                   # optional data.frame of per-cell metadata (e.g. cell type), or NULL
mito_pattern      <- "^MT-"                 # use "^mt-" for mouse
min_genes_per_cell <- 200                   # nFeature_RNA lower bound
max_pct_mito      <- 10                     # percent.mt upper bound
min_cells_per_gene <- 10                    # gene kept if expressed in >= this many cells
min_celltype_frac <- 0.01                   # drop cell types below this fraction of cells (skipped if no label column)
label_column      <- "celltype"             # column in metadata to use for the above, if present
n_hvgs            <- 5000
outdir            <- "data"
tag               <- "dataset"

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

## -----------------------------
## Load counts + build the Seurat object
## -----------------------------
counts_matrix <- readRDS(counts_path)        # swap for Read10X(), read.csv etc. as needed
meta <- if (!is.null(metadata_path)) readRDS(metadata_path) else NULL

seurat_obj <- CreateSeuratObject(counts = counts_matrix, meta.data = meta)

## -----------------------------
## Cell QC
## -----------------------------
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = mito_pattern)
seurat_obj <- subset(
  seurat_obj,
  subset = nFeature_RNA > min_genes_per_cell & percent.mt < max_pct_mito
) # other QC metrics may be added here as needed

## -----------------------------
## Gene filtering
## -----------------------------
counts_mat <- GetAssayData(seurat_obj, layer = "counts")
genes_to_keep <- rownames(counts_mat)[Matrix::rowSums(counts_mat > 0) >= min_cells_per_gene]
seurat_obj <- subset(seurat_obj, features = genes_to_keep)

## -----------------------------
## Optional: drop rare cell types (only if a label column exists)
## -----------------------------
if (!is.null(label_column) && label_column %in% colnames(seurat_obj@meta.data)) {
  cell_props <- prop.table(table(seurat_obj@meta.data[[label_column]]))
  keep_types <- names(cell_props)[cell_props >= min_celltype_frac]
  seurat_obj <- subset(seurat_obj, cells = colnames(seurat_obj)[seurat_obj@meta.data[[label_column]] %in% keep_types])
}

## -----------------------------
## TMM normalization + tagwise dispersion (edgeR) - run before HVG selection for more accurate library size estimates
## -----------------------------
counts_mat <- GetAssayData(seurat_obj, layer = "counts")
dge <- DGEList(counts = counts_mat)
dge <- calcNormFactors(dge, method = "TMM")
effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
dge <- estimateDisp(dge)
# Alternatively, total library size can be used as a normalization factor, but TMM is generally preferred.

## -----------------------------
## Optional: highly variable genes (can be filtered later when running scFLAME)
## -----------------------------
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = n_hvgs)
seurat_obj <- subset(seurat_obj, features = VariableFeatures(seurat_obj))
dge <- dge[rownames(seurat_obj), ]

## -----------------------------
## Write outputs
## -----------------------------
write.csv(
  t(as.matrix(counts_mat)),
  file.path(outdir, paste0(tag, "_rawcounts.csv"))
)

write.csv(
  data.frame(cell_id = colnames(seurat_obj), tmm_lib_size = effective_lib_size),
  file.path(outdir, paste0(tag, "_tmm_effective_library_sizes.csv")),
  row.names = FALSE
)

write.csv(
  data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
  file.path(outdir, paste0(tag, "_edgeR_tagwise_dispersion.csv")),
  row.names = FALSE
)

if (!is.null(label_column) && label_column %in% colnames(seurat_obj@meta.data)) {
  write.csv(
    data.frame(
      cell_id = colnames(seurat_obj),
      celltype = as.integer(as.factor(seurat_obj@meta.data[[label_column]]))
    ),
    file.path(outdir, paste0(tag, "_clusters.csv")),
    row.names = FALSE
  )
}

saveRDS(seurat_obj, file.path(outdir, paste0(tag, "_processed.rds")))

message("Done. ", ncol(seurat_obj), " cells x ", nrow(seurat_obj), " genes written to ", outdir)