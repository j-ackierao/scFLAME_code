## ============================================================
## preprocess_tm_spleen.R
## Tabula Muris spleen, 3-month timepoint, both 10x and SS2.
## ============================================================

suppressMessages({
  library(TabulaMurisSenisData)
  library(Seurat)
  library(edgeR)
})

set.seed(42)

outdir <- "data"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

process_spleen <- function(sce, tag, min_genes_per_cell, min_pct_expr = 0.01, n_hvgs = 8000) {
  seurat_obj <- CreateSeuratObject(
    counts = as(counts(sce), "dgCMatrix"),
    meta.data = as.data.frame(colData(sce))
  )

  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^mt-")
  seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > min_genes_per_cell & percent.mt < 5)

  counts_mat <- GetAssayData(seurat_obj, layer = "counts")
  genes_to_keep <- rownames(counts_mat)[Matrix::rowSums(counts_mat > 0) >= 3]
  seurat_obj <- subset(seurat_obj, features = genes_to_keep)

  seurat_obj$ground_truth <- as.character(seurat_obj$cell_ontology_class)
  cell_props <- prop.table(table(seurat_obj$ground_truth))
  keep_types <- names(cell_props)[cell_props > min_pct_expr]
  seurat_obj <- subset(seurat_obj, subset = ground_truth %in% keep_types)

  seurat_obj <- NormalizeData(seurat_obj)
  ## HVGs selected before TMM/dispersion here (unlike the generic template) for runtime - 
  ## full gene-set TMM fitting was slow for one of these datasets. Library size / dispersion
  ## here reflect the HVG-filtered set, not the full transcriptome. 
  seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = n_hvgs)
  seurat_obj <- subset(seurat_obj, features = VariableFeatures(seurat_obj))

  counts_mat <- GetAssayData(seurat_obj, layer = "counts")
  dge <- DGEList(counts = counts_mat)
  dge <- calcNormFactors(dge, method = "TMM")
  effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
  dge <- estimateDisp(dge)

  write.csv(
    data.frame(cell_id = colnames(counts_mat), tmm_lib_size = effective_lib_size),
    file.path(outdir, paste0(tag, "_tmm_effective_library_sizes.csv")), row.names = FALSE
  )
  write.csv(
    data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
    file.path(outdir, paste0(tag, "_edgeR_tagwise_dispersion.csv")), row.names = FALSE
  )
  write.csv(
    t(as.matrix(GetAssayData(seurat_obj, layer = "counts"))),
    file.path(outdir, paste0(tag, "_rawcounts.csv"))
  )
  write.csv(
    data.frame(
      cell_id = colnames(seurat_obj),
      cluster_id = as.integer(as.factor(seurat_obj$ground_truth)),
      celltype = seurat_obj$ground_truth
    ),
    file.path(outdir, paste0(tag, "_clusters.csv")), row.names = FALSE
  )
  saveRDS(seurat_obj, file.path(outdir, paste0(tag, "_processed.rds")))

  message(tag, ": ", ncol(seurat_obj), " cells x ", nrow(seurat_obj), " genes")
  seurat_obj
}

spleen_ss2 <- TabulaMurisSenisFACS(tissues = "Spleen")[[1]]
spleen_10x <- TabulaMurisSenisDroplet(tissues = "Spleen")[[1]]

spleen_benchmark_ss2 <- spleen_ss2[, spleen_ss2$age == "3m"]
spleen_benchmark_10x <- spleen_10x[, spleen_10x$age == "3m"]

seurat_ss2 <- process_spleen(spleen_benchmark_ss2, "tm_spleen_ss2", min_genes_per_cell = 500)
seurat_10x <- process_spleen(spleen_benchmark_10x, "tm_spleen_10x", min_genes_per_cell = 250)