## ============================================================
## pbmc68k.R
## Fresh 68k PBMC (10x Genomics "pbmc68k" donor A).
##
## NOTE on normalization: unlike the other datasets in this repo,
## library size here is a simple total-count size factor (colSums,
## centred at 1.0), not TMM - calcNormFactors() triggers a dense-
## matrix coercion on the full 68k x ~20k count matrix that isn't
## practical here. This deviation is documented in the paper.
## Dispersions are likewise estimated on a random 5000-cell
## subsample of the HVG-filtered matrix for tractability.
##
## Download the raw data from https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0.
## Cluster labels: We reproduced the original authors' clustering - k-means
## (k=10) on the top 50 PCs from the top 1000 HVGs - by running their
## own script, main_process_68k_pbmc.R, from
## https://github.com/10XGenomics/single-cell-3prime-paper and 
## saved to a csv with columns "barcode" and "cluster".
## ============================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(edgeR)
})

set.seed(42)  # governs the random 5000-cell subsample used for dispersion estimation

archive_path <- "~/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
raw_data_dir <- "./68k_pbmc_processed/"     # where the 10x matrix is extracted to and read from
annot_path   <- "68k_pbmc_kmeans10_clusters.csv"  # output of 10XGenomics/single-cell-3prime-paper's main_process_68k_pbmc.R
annot_barcode_col <- "barcode"   # confirm this matches the actual column name in saved CSV
annot_cluster_col <- "cluster"   # confirm this matches the actual column name in saved CSV

outdir <- "data"               # separate from raw_data_dir - never write filtered output back into the raw input directory
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

min_features <- 200
max_pct_mito <- 5
n_hvgs <- 5000
n_dispersion_subsample <- 5000

## -----------------------------
## Load + basic QC
## -----------------------------
untar(archive_path)
pbmc.data <- Read10X(data.dir = raw_data_dir)

pbmc68 <- CreateSeuratObject(
  counts = pbmc.data,
  project = "pbmc68k",
  min.cells = 3,
  min.features = min_features
)

pbmc68[["percent.mt"]] <- PercentageFeatureSet(pbmc68, pattern = "^MT-")
pbmc68 <- subset(pbmc68, subset = nFeature_RNA > min_features & percent.mt < max_pct_mito)

## -----------------------------
## Attach cluster labels reproduced via the original authors' code
## -----------------------------
cluster_annot <- read.csv(annot_path, stringsAsFactors = FALSE)
matched_annot <- cluster_annot[match(colnames(pbmc68), cluster_annot[[annot_barcode_col]]), ]
 
if (any(is.na(matched_annot[[annot_cluster_col]]))) {
  warning(sum(is.na(matched_annot[[annot_cluster_col]])),
          " cells had no matching barcode in ", annot_path)
}
 
pbmc68$true_label <- matched_annot[[annot_cluster_col]]

## -----------------------------
## Remove the smallest cluster - this only represents 97 of the 68k cells, and is a very small minority of the total population.
## -----------------------------
cluster_counts <- table(pbmc68$true_label)
cluster_to_drop <- names(cluster_counts)[which.min(cluster_counts)]
message(sprintf(
  "Removing smallest cluster '%s' (%d of %d cells)",
  cluster_to_drop, min(cluster_counts), sum(cluster_counts)
))
pbmc68 <- subset(pbmc68, subset = true_label != cluster_to_drop)

## -----------------------------
## Library size, computed on the full (pre-HVG) gene set
## -----------------------------
all_genes_counts <- GetAssayData(pbmc68, layer = "counts")
raw_lib_sizes <- Matrix::colSums(all_genes_counts)
libsize_factors <- raw_lib_sizes / mean(raw_lib_sizes)

write.csv(
  data.frame(cell_id = colnames(pbmc68), libsize_lib_size = libsize_factors),
  file.path(outdir, "pbmc68k_libsize_effective_library_sizes.csv"),
  row.names = FALSE
)

## -----------------------------
## HVG selection + export (standard 10x mtx format: matrix.mtx,
## genes.tsv with ensembl+symbol columns, barcodes.tsv - so
## sc.read_10x_mtx() can load this directly downstream)
## -----------------------------
pbmc68 <- FindVariableFeatures(pbmc68, selection.method = "vst", nfeatures = n_hvgs)
hvg_names <- VariableFeatures(pbmc68)
raw_counts <- all_genes_counts[hvg_names, ]
 
writeMM(obj = raw_counts, file = file.path(outdir, "matrix.mtx"))
write.table(
  data.frame(ensembl = rownames(raw_counts), symbol = rownames(raw_counts)),
  file = file.path(outdir, "genes.tsv"),
  col.names = FALSE, row.names = FALSE, quote = FALSE, sep = "\t"
)
write.table(
  colnames(raw_counts),
  file = file.path(outdir, "barcodes.tsv"),
  col.names = FALSE, row.names = FALSE, quote = FALSE
)

## -----------------------------
## Dispersion, estimated on a random subsample of cells for
## tractability (edgeR defaults; libsize_factors reused rather than
## a fresh TMM fit, consistent with the library-size step above)
## -----------------------------
sampled_idx <- sample(seq_len(ncol(raw_counts)), size = n_dispersion_subsample, replace = FALSE)
counts_subset <- raw_counts[, sampled_idx]

dge_subset <- DGEList(counts = counts_subset)
dge_subset$samples$lib.size <- raw_lib_sizes[sampled_idx]
dge_subset$samples$norm.factors <- libsize_factors[sampled_idx]
dge_subset <- estimateDisp(dge_subset)

write.csv(
  data.frame(gene = rownames(raw_counts), dispersion = dge_subset$tagwise.dispersion),
  file.path(outdir, "pbmc68k_edgeR_tagwise_dispersion.csv"),
  row.names = FALSE
)

message("Done. ", ncol(pbmc68), " cells x ", nrow(all_genes_counts), " genes (",
        nrow(raw_counts), " HVGs written); dispersion from a ", n_dispersion_subsample, "-cell subsample")