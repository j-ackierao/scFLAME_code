## ============================================================
## preprocess_mouse_uterus.R
## Mouse Cell Atlas uterus (two batches). Data can be downloaded from
## https://figshare.com/s/865e694ad06d5857db4b; files required
## for this script are MCA_CellAssignments.csv and MCA_BatchRemove_dge.zip, 
## which contains Uterus1_rm.batch_dge.txt.gz and Uterus2_rm.batch_dge.txt.gz.
## ============================================================

suppressMessages({
  library(Seurat)
  library(edgeR)
})

set.seed(42)

outdir <- "data"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

min_genes_per_cell <- 250
min_cells_per_gene <- 50
max_pct_mito <- 5
min_celltype_frac <- 0.03

## -----------------------------
## Load + merge batches
## -----------------------------
mat.raw1 <- read.table(gzfile("data/Uterus1_rm.batch_dge.txt.gz"), header = TRUE)
mat.raw2 <- read.table(gzfile("data/Uterus2_rm.batch_dge.txt.gz"), header = TRUE)

celltypes <- read.csv("data/MCA_CellAssignments.csv")
celltypes_uterus <- celltypes[celltypes$Tissue == "Uterus", ]
rownames(celltypes_uterus) <- celltypes_uterus$Cell.name

seurat.uterus1 <- CreateSeuratObject(counts = mat.raw1, project = "Uterus1")
seurat.uterus2 <- CreateSeuratObject(counts = mat.raw2, project = "Uterus2")
seurat.uterus1$batch <- "batch1"
seurat.uterus2$batch <- "batch2"

seurat.uterus <- merge(seurat.uterus1, y = seurat.uterus2,
                        add.cell.ids = c("U1", "U2"), project = "Uterus")

## -----------------------------
## QC filtering
## -----------------------------
seurat.uterus[["percent.mt"]] <- PercentageFeatureSet(seurat.uterus, pattern = "^mt-")
seurat.uterus <- subset(seurat.uterus, subset = nFeature_RNA >= min_genes_per_cell)

seurat.uterus <- JoinLayers(seurat.uterus)
gene_counts <- Matrix::rowSums(GetAssayData(seurat.uterus, layer = "counts") > 0)
genes_to_keep <- names(gene_counts)[gene_counts >= min_cells_per_gene]
seurat.uterus <- subset(seurat.uterus, features = genes_to_keep)

seurat.uterus <- subset(seurat.uterus, subset = percent.mt < max_pct_mito)

## -----------------------------
## Attach cell-type annotations, keep cells present in the metadata file
## -----------------------------
cells_clean <- gsub("^U1_|^U2_", "", colnames(seurat.uterus))
common_cells <- intersect(cells_clean, rownames(celltypes_uterus))
keep_cells <- colnames(seurat.uterus)[cells_clean %in% common_cells]
seurat.uterus <- subset(seurat.uterus, cells = keep_cells)

cells_clean <- gsub("^U1_|^U2_", "", colnames(seurat.uterus))
seurat.uterus$CellType <- celltypes_uterus[cells_clean, "Annotation"]
seurat.uterus$ClusterID <- celltypes_uterus[cells_clean, "ClusterID"]
seurat.uterus$batch_metadata <- celltypes_uterus[cells_clean, "Batch"]

## -----------------------------
## Drop rare cell types
## -----------------------------
cell_counts <- table(seurat.uterus$CellType)
keep_types <- names(cell_counts)[cell_counts >= min_celltype_frac * ncol(seurat.uterus)]
seurat.uterus <- subset(seurat.uterus, subset = CellType %in% keep_types)

## -----------------------------
## TMM normalization + dispersion
## -----------------------------
counts_mat <- GetAssayData(seurat.uterus, layer = "counts")
dge <- DGEList(counts = counts_mat)
dge <- calcNormFactors(dge, method = "TMM")
effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
dge <- estimateDisp(dge)

## -----------------------------
## Write outputs
## -----------------------------
write.csv(
  data.frame(cell_id = colnames(counts_mat), tmm_lib_size = effective_lib_size),
  file.path(outdir, "uterus_tmm_effective_library_sizes.csv"), row.names = FALSE
)
write.csv(
  data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
  file.path(outdir, "uterus_edgeR_tagwise_dispersion.csv"), row.names = FALSE
)
write.csv(
  t(as.matrix(GetAssayData(seurat.uterus, layer = "counts"))),
  file.path(outdir, "uterus_rawcounts.csv")
)
write.csv(
  data.frame(
    cell_id = colnames(seurat.uterus),
    cluster_id = as.integer(as.factor(seurat.uterus$CellType)),
    celltype = seurat.uterus$CellType,
    batch = seurat.uterus$batch,
    batch_metadata = seurat.uterus$batch_metadata
  ),
  file.path(outdir, "uterus_clusters.csv"), row.names = FALSE
)
saveRDS(seurat.uterus, file.path(outdir, "seurat_uterus_processed.rds"))

message("Done. ", ncol(seurat.uterus), " cells x ", nrow(seurat.uterus), " genes across ",
        length(unique(seurat.uterus$batch)), " batches")