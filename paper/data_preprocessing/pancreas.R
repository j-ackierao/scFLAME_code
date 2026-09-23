## ============================================================
## pancreas.R
## Combines four public human pancreas scRNA-seq studies:
##   GSE81076    CEL-seq    (Grun et al. 2016)
##   GSE85241    CEL-seq2   (Muraro et al. 2016)
##   GSE86473    Smart-seq2 (Lawlor et al. 2017)
##   E-MTAB-5061 Smart-seq2 (Segerstolpe et al. 2016)
##
## Adapted from https://github.com/MarioniLab/MNN2017/tree/master/Pancreas
## (Haghverdi et al., "Batch effects in single-cell RNA-sequencing
## data are corrected by matching mutual nearest neighbors", Nature
## Biotechnology 2018), This script requires their download script
## `DownloadData.sh` to be run first to populate RawData/ with the raw
## GSE86473 feature-count files and the E-MTAB-5061/GEO downloads
## that this script fetches directly.
##
## Deviations from the original MNN2017 pipeline are described in the
## manuscript ('Methods').
## ============================================================

suppressPackageStartupMessages({
  library(scran)
  library(scater)
  library(SingleCellExperiment)
  library(biomaRt)
  library(edgeR)
  library(Rtsne)
  library(cluster)
})

set.seed(42) 

rawdir <- "RawData"   # populate via MNN2017's download script before running this
outdir <- "data/pancreas"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

## -----------------------------
## Shared helpers
## -----------------------------
filter_sparse <- function(df) {
  gene_sparsity <- rowSums(df == 0) / ncol(df)
  df <- df[gene_sparsity < 0.9, ] # genes with >90% zero counts are dropped
  cell_sparsity <- colSums(df == 0) / nrow(df)
  df[, cell_sparsity < 0.8] # cells with >80% zero counts are dropped
}

deconvolution_normalize <- function(counts_mat) {
  sce <- SingleCellExperiment(list(counts = as.matrix(counts_mat)))
  clusters <- quickCluster(sce, min.size = 120)
  sce <- computeSumFactors(sce, sizes = c(10, 20, 40, 60), positive = TRUE,
                            assay.type = "counts", clusters = clusters)
  logNormCounts(sce)
}

## find_hvg(): unchanged from the MNN2017 pipeline 
find_hvg <- function(dataframe, p.threshold = 1e-2, return.p = FALSE) {
  require(MASS); require(limSolve); require(statmod)
  gene.names <- rownames(dataframe)
  means <- rowMeans(dataframe, na.rm = TRUE)
  vars <- apply(dataframe, 1, var, na.rm = TRUE)
  cv2 <- vars / (means^2)

  recip.means <- 1 / means
  recip.means[is.infinite(recip.means)] <- 0
  useForFit <- recip.means <= 0.1

  fit <- glmgam.fit(cbind(a0 = 1, a1tilde = recip.means[!useForFit]), cv2[!useForFit])
  a0 <- unname(fit$coefficients["a0"])
  a1 <- unname(fit$coefficients["a1tilde"])

  d.f <- ncol(dataframe) - 1
  a.fit <- (a1 / means) + a0
  varFitRatio <- vars / (a.fit * means^2)

  pvals <- pchisq(varFitRatio * d.f, d.f, lower.tail = FALSE)
  pvals[is.na(pvals)] <- 1.0
  adj.pvals <- p.adjust(pvals, method = "fdr")
  HVG <- adj.pvals <= p.threshold

  if (return.p) HVG <- cbind(HVG, adj.pvals)
  HVG
}

## marker-gene PAM cell typing, shared by GSE81076/GSE85241
assign_marker_celltypes <- function(norm_df, hvg_ids, meta, tag) {
  hvg <- norm_df[norm_df$gene_id %in% hvg_ids, 1:(ncol(norm_df) - 1)]

  tsne <- Rtsne(t(hvg), perplexity = 30)
  map <- data.frame(tsne$Y)
  colnames(map) <- c("Dim1", "Dim2")
  map$Sample <- colnames(hvg)
  uber <- merge(map, meta, by = "Sample")

  marker_genes <- rownames(norm_df)[grepl(
    rownames(norm_df),
    pattern = "(^GCG)|(^KRT19)|(^INS)|(^SST)|(^PPY)|(^PRSS1)|(^COL1A1)|(^GHRL)|(^ESAM)"
  )]
  marker_exprs <- norm_df[marker_genes, 1:(ncol(norm_df) - 1)]
  marker_df <- data.frame(t(marker_exprs))
  marker_df$Sample <- rownames(marker_df)

  marker.pam <- pam(marker_df[, seq_len(ncol(marker_df) - 1)], 9)
  marker_cluster <- data.frame(markClust = as.factor(marker.pam$clustering),
                                Sample = rownames(marker_df))

  uber <- merge(uber, marker_df, by = "Sample")
  uber <- merge(uber, marker_cluster, by = "Sample")

  message(tag, ": marker-cluster x expression cross-tab")
  print(table(uber$markClust))

  ## Cluster -> cell type assignment by highest-expressing marker gene.
  ## NOTE: cluster numbers below are dataset-specific - re-verify these if
  ## re-running from a different random seed or scran version, since
  ## PAM cluster IDs are arbitrary labels, not stable across runs.
  uber$CellType <- ""
  if (tag == "GSE81076") {
    uber$CellType[uber$markClust %in% c(5)] <- "Alpha"
    uber$CellType[uber$markClust %in% c(9)] <- "PP"
    uber$CellType[uber$markClust == 3] <- "Beta"
    uber$CellType[uber$markClust == 7] <- "Delta"
    uber$CellType[uber$markClust %in% c(1)] <- "Acinar"
    uber$CellType[uber$markClust %in% c(2, 4, 6)] <- "Ductal"
    uber$CellType[uber$markClust %in% c(8) & uber$COL1A1 >= 2] <- "Mesenchyme"
  } else if (tag == "GSE85241") {
    uber$CellType[uber$markClust %in% c(7)] <- "PP"
    uber$CellType[uber$markClust %in% c(4, 8)] <- "Beta"
    uber$CellType[uber$markClust %in% c(1, 6)] <- "Alpha"
    uber$CellType[uber$markClust %in% c(3)] <- "Delta"
    uber$CellType[uber$markClust %in% c(5)] <- "Acinar"
    uber$CellType[uber$markClust %in% c(2)] <- "Ductal"
    uber$CellType[uber$markClust %in% c(9)] <- "Mesenchyme"
  }
  uber
}

## ============================================================
## GSE81076
## ============================================================
process_gse81076 <- function() {
  path <- file.path(rawdir, "GSE81076_D2_3_7_10_17.txt.gz")
  if (!file.exists(path)) {
    download.file("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81076/suppl/GSE81076%5FD2%5F3%5F7%5F10%5F17%2Etxt%2Egz", path)
  }
  df <- read.table(path, sep = "\t", h = TRUE, stringsAsFactors = FALSE)
  ndim <- ncol(df)

  donor <- unlist(regmatches(colnames(df)[2:ndim], gregexpr("D[0-9]{1,2}", colnames(df)[2:ndim])))
  plate <- unlist(lapply(strsplit(unlist(lapply(strsplit(colnames(df)[2:ndim], split = "D[0-9]{1,2}", perl = TRUE),
                                                 function(x) x[2])), split = "_", fixed = TRUE), function(x) x[1]))
  meta <- data.frame(Sample = colnames(df)[2:ndim], Donor = donor, Plate = plate,
                      Protocol = "CELseq", Study = "GSE81076")

  colnames(df) <- gsub("X", "gene", colnames(df))
  df$gene <- gsub("__chr[0-9]+", "", df$gene)
  df <- df[!duplicated(df$gene), ]
  rownames(df) <- df$gene
  df <- df[, -1]

  df.nz <- filter_sparse(df)
  df.nz <- apply(df.nz, 2, as.integer)
  rownames(df.nz) <- rownames(filter_sparse(df))

  write.csv(t(df.nz), file.path(outdir, "GSE81076_raw_counts.csv"), row.names = TRUE)
  write.table(meta, file.path(outdir, "GSE81076_metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  sce <- deconvolution_normalize(df.nz)
  norm <- data.frame(logcounts(sce))
  norm$gene_id <- rownames(df.nz)
  write.table(norm, file.path(outdir, "GSE81076_SFnorm.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  hvg <- find_hvg(norm[, 1:(ncol(norm) - 1)], p.threshold = 1e-2, return.p = TRUE)
  hvg_df <- data.frame(HVG = hvg[, 1], pval = hvg[, 2], gene_id = rownames(norm))
  write.table(hvg_df, file.path(outdir, "GSE81076-HVG.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  uber <- assign_marker_celltypes(norm, hvg_df$gene_id[hvg_df$HVG], meta, "GSE81076")
  write.table(uber, file.path(outdir, "GSE81076_marker_metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
}

## ============================================================
## GSE85241
## ============================================================
process_gse85241 <- function() {
  path <- file.path(rawdir, "GSE85241_cellsystems_dataset_4donors_updated.csv")
  if (!file.exists(path)) {
    download.file("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE85nnn/GSE85241/suppl/GSE85241%5Fcellsystems%5Fdataset%5F4donors%5Fupdated%2Ecsv%2Egz", path)
  }
  df <- read.table(path, sep = "\t", h = TRUE, stringsAsFactors = FALSE)
  df$gene_id <- rownames(df)
  colnames(df) <- gsub("X", "gene_id", colnames(df))

  n <- ncol(df) - 1
  donor <- unlist(lapply(strsplit(colnames(df)[1:n], ".", fixed = TRUE), function(x) x[1]))
  plate <- unlist(lapply(strsplit(colnames(df)[1:n], ".", fixed = TRUE), function(x) x[2]))
  meta <- data.frame(Sample = colnames(df)[1:n], Donor = donor, Plate = plate,
                      Protocol = "CELseq2", Study = "GSE85241")
  write.table(meta, file.path(outdir, "GSE85241_metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  df$gene_id <- gsub("__chr[0-9X]+", "", df$gene_id)
  df <- df[!duplicated(df$gene_id), ]
  rownames(df) <- df$gene_id
  df <- df[, 1:n]

  df.nz <- filter_sparse(df)
  df.nz <- apply(df.nz, 2, as.integer)
  rownames(df.nz) <- rownames(filter_sparse(df))

  write.csv(t(df.nz), file.path(outdir, "GSE85241_raw_counts.csv"), row.names = TRUE)

  sce <- deconvolution_normalize(df.nz)
  norm <- data.frame(logcounts(sce))
  norm$gene_id <- rownames(df.nz)
  write.table(norm, file.path(outdir, "GSE85241_SFnorm.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  hvg <- find_hvg(norm[, 1:(ncol(norm) - 1)], p.threshold = hvg_p_threshold, return.p = TRUE)
  hvg_df <- data.frame(HVG = hvg[, 1], pval = hvg[, 2], gene_id = rownames(norm))
  write.table(hvg_df, file.path(outdir, "GSE85241-HVG.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  uber <- assign_marker_celltypes(norm, hvg_df$gene_id[hvg_df$HVG], meta, "GSE85241")
  write.table(uber, file.path(outdir, "GSE85241_marker_metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
}

## ============================================================
## GSE86473
## Raw per-cell-type feature-count files and the experimental design
## TSV are expected under RawData/ 
## ============================================================
process_gse86473 <- function() {
  cts <- lapply(c("alpha", "beta", "delta", "PP"), function(ct) {
    read.table(file.path(rawdir, paste0(ct, "-feature_counts.tsv.gz")), sep = "\t", h = TRUE, stringsAsFactors = FALSE)
  })
  df <- Reduce(function(x, y) merge(x, y, by = "gene_id"), cts)
  colnames(df) <- tolower(unlist(lapply(strsplit(colnames(df), ".", fixed = TRUE), function(x) x[1])))

  meta <- read.table(file.path(rawdir, "GSE86473_experimental_design.tsv"), sep = "\t", h = TRUE, stringsAsFactors = FALSE)
  meta$Sample <- tolower(meta$Sample)
  meta$Study <- "GSE86473"
  meta$CellType <- paste0(toupper(substr(meta$CellType, 1, 1)), substring(meta$CellType, 2))
  write.table(meta, file.path(outdir, "GSE86473_metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  rownames(df) <- df$gene_id
  ensembl <- useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl", GRCh = 37)
  gene_symbol <- getBM(attributes = c("ensembl_gene_id", "external_gene_name"),
                        filters = "ensembl_gene_id", mart = ensembl, values = df$gene_id)
  dat <- merge(df, gene_symbol, by.x = "gene_id", by.y = "ensembl_gene_id")
  df <- as.data.frame(append(dat[, -1], list(gene_id = as.character(dat$external_gene_name)), after = 0))
  df <- df[!duplicated(df$gene_id), ]
  rownames(df) <- df$gene_id
  df <- df[, 2:(ncol(df) - 1)]

  df.nz <- filter_sparse(df)
  df.nz <- apply(df.nz, 2, as.integer)
  rownames(df.nz) <- rownames(filter_sparse(df))

  write.csv(t(df.nz), file.path(outdir, "GSE86473_raw_counts.csv"), row.names = TRUE)

  ## No deconvolution here (unlike GSE81076/GSE85241): this dataset
  ## already has author-provided cell-type labels, so HVG-based
  ## marker clustering is never run on it.
  sce <- SingleCellExperiment(list(counts = as.matrix(df.nz)))
  sce <- logNormCounts(sce)
  norm <- data.frame(logcounts(sce))
  norm$gene_id <- rownames(df.nz)
  write.table(norm, file.path(outdir, "GSE86473_SFnorm.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  hvg <- find_hvg(norm[, 1:(ncol(norm) - 1)], p.threshold = hvg_p_threshold, return.p = TRUE)
  hvg_df <- data.frame(HVG = hvg[, 1], pval = hvg[, 2], gene_id = rownames(norm))
  write.table(hvg_df, file.path(outdir, "GSE86473-HVG.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
}

## ============================================================
## E-MTAB-5061
## ============================================================
process_emtab5061 <- function() {
  archive <- file.path(rawdir, "EMTAB5061_rpkm_counts.txt.zip")
  if (!file.exists(archive)) {
    download.file("https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-5061/files/E-MTAB-5061.processed.1.zip", archive)
  }
  unzipped <- file.path(rawdir, "pancreas_refseq_rpkms_counts_3514sc.txt")
  if (!file.exists(unzipped)) system(paste("unzip", archive, "-d", rawdir))

  df <- read.table(unzipped, h = FALSE, sep = "\t", stringsAsFactors = FALSE)
  col_names <- unlist(read.table(unzipped, h = FALSE, sep = "\t", stringsAsFactors = FALSE, comment.char = "", nrows = 1))

  df <- df[, c(1, 3517:ncol(df))]
  colnames(df) <- gsub("#samples", "gene_id", col_names)

  sdrf_path <- file.path(rawdir, "E-MTAB-5061.sdrf.txt")
  if (!file.exists(sdrf_path)) {
    download.file("https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-5061/E-MTAB-5061.sdrf.txt", sdrf_path)
  }
  sdrf <- read.table(sdrf_path, h = TRUE, sep = "\t", stringsAsFactors = FALSE)

  meta <- sdrf[, c("Assay.Name", "Characteristics.cell.type.", "Characteristics.individual.")]
  colnames(meta) <- c("Sample", "CellType", "Donor")
  meta$Study <- "E-MTAB-5061"
  meta$Protocol <- "SmartSeq2"

  remove_cells <- unique(sdrf$Assay.Name[sdrf$Characteristics.single.cell.well.quality. == "low quality cell"])
  df <- df[, !colnames(df) %in% remove_cells]
  meta <- meta[!meta$Sample %in% remove_cells, ]
  meta$CellType <- gsub(" cell", "", meta$CellType)
  meta$CellType <- paste0(toupper(substr(meta$CellType, 1, 1)), substring(meta$CellType, 2))
  write.table(meta, file.path(outdir, "E-MTAB-5061_metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  df <- df[!duplicated(df$gene_id), ]
  rownames(df) <- df$gene_id
  df <- df[, -1]

  df.nz <- filter_sparse(df)
  df.nz <- apply(df.nz, 2, as.integer)
  rownames(df.nz) <- rownames(filter_sparse(df))

  write.csv(t(df.nz), file.path(outdir, "E-MTAB-5061_raw_counts.csv"), row.names = TRUE)

  ## No deconvolution here either, for the same reason as GSE86473 -
  ## already-labeled dataset, no HVG-based cell typing needed.
  sce <- SingleCellExperiment(list(counts = as.matrix(df.nz)))
  sce <- logNormCounts(sce)
  norm <- data.frame(logcounts(sce))
  norm$gene_id <- rownames(df.nz)
  write.table(norm, file.path(outdir, "E-MTAB-5061_SFnorm.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  hvg <- find_hvg(norm[, 1:(ncol(norm) - 1)], p.threshold = hvg_p_threshold, return.p = TRUE)
  hvg_df <- data.frame(HVG = hvg[, 1], pval = hvg[, 2], gene_id = rownames(norm))
  write.table(hvg_df, file.path(outdir, "E-MTAB-5061-HVG.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
}

process_gse81076()
process_gse85241()
process_gse86473()
process_emtab5061()

## ============================================================
## Merge across all four studies, harmonize labels, compute TMM +
## dispersion on the final intersected matrix.
## ============================================================
studies <- c("GSE81076", "GSE85241", "GSE86473", "E-MTAB-5061")
meta_files <- c(
  "GSE81076" = "GSE81076_marker_metadata.tsv", "GSE85241" = "GSE85241_marker_metadata.tsv",
  "GSE86473" = "GSE86473_metadata.tsv", "E-MTAB-5061" = "E-MTAB-5061_metadata.tsv"
)
counts_files <- setNames(paste0(studies, "_raw_counts.csv"), studies)
norm_files <- setNames(paste0(studies, "_SFnorm.tsv"), studies)

raw_list <- list()
clusters_list <- list()

for (s in studies) {
  raw_cxg <- read.csv(file.path(outdir, counts_files[s]), row.names = 1, check.names = FALSE)
  norm <- read.table(file.path(outdir, norm_files[s]), header = TRUE, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  gene_ids <- norm$gene_id

  if (ncol(raw_cxg) != length(gene_ids)) {
    stop(s, ": raw count matrix has ", ncol(raw_cxg), " genes but SFnorm has ", length(gene_ids))
  }
  colnames(raw_cxg) <- gene_ids
  raw_gxc <- t(as.matrix(raw_cxg))

  meta <- read.table(file.path(outdir, meta_files[s]), h = TRUE, sep = "\t", stringsAsFactors = FALSE)
  meta <- meta[meta$Sample %in% colnames(raw_gxc) & meta$CellType != "", ]
  raw_gxc <- raw_gxc[, meta$Sample, drop = FALSE]
  storage.mode(raw_gxc) <- "integer"

  clusters_list[[s]] <- data.frame(cell_id = meta$Sample, cluster_id = meta$CellType, batch_id = s)
  raw_list[[s]] <- raw_gxc
  message(s, ": ", nrow(raw_gxc), " genes x ", ncol(raw_gxc), " cells")
}

clusters_all <- do.call(rbind, clusters_list)
clusters_all$cluster_id[clusters_all$cluster_id == "PP"] <- "Gamma"
clusters_all$cluster_id[grepl("Mesenchyme|Co-ex|Endo|Epsi|Mast|MHC|Uncl|Not|PSC", clusters_all$cluster_id)] <- "other"
write.csv(clusters_all, file.path(outdir, "pancreas_all_clusters.csv"), row.names = FALSE)

common_genes <- Reduce(intersect, lapply(raw_list, rownames))
merged <- do.call(cbind, lapply(raw_list, function(x) x[common_genes, , drop = FALSE]))
write.csv(t(merged), file.path(outdir, "pancreas_all_4studies_raw_counts.csv"), row.names = TRUE)

dge_all <- DGEList(counts = merged)
dge_all <- calcNormFactors(dge_all, method = "TMM")
dge_all <- estimateDisp(dge_all)

write.csv(
  data.frame(gene = rownames(dge_all), dispersion = dge_all$tagwise.dispersion),
  file.path(outdir, "pancreas_all_4studies_edgeR_tagwise_dispersion.csv"), row.names = FALSE
)
write.csv(
  data.frame(cell_id = colnames(merged),
             tmm_lib_size = dge_all$samples$lib.size * dge_all$samples$norm.factors),
  file.path(outdir, "pancreas_all_4studies_tmm_effective_library_sizes.csv"), row.names = FALSE
)

message("Done. ", ncol(merged), " cells x ", nrow(merged), " genes across ", length(studies), " studies")

## R version 4.6.0, package versions used in this script:
## scran          scater        SingleCellExperiment              biomaRt                edgeR                Rtsne              cluster 
## "1.40.0"      "1.40.1"             "1.34.0"             "2.68.0"             "4.10.1"               "0.17"            "2.1.8.2" 