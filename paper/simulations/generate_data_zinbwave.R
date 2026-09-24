## ============================================================
## generate_data_zinbwave.R
## Simulate data from the ZINBWave simulation model, using either the Baron or Segerstolpe 
## datasets as reference parameters. Reproduces same simulation as the manuscript.
## This script is intended to be run as a SLURM array job, with each task simulating a different replicate.
## Replicates 0-19 were used to generate the 20 replicates in the manuscript; we set SLURM_ARRAY_TASK_ID to 1-20.
## We recommend for every set of parameters to save data in a different folder.
## We ran with 5 CPU cores; this can be adjusted. BiocParallel is set to take 4 workers in this script.
##
## Usage:
##   Rscript simulate_zinbwave.R --dataset baron --dropout_mid 0.5 --replicate 0 --out_dir baron_sim_data
##   Rscript simulate_zinbwave.R --dataset segerstolpe --dropout_mid 1.0 --replicate 3 \
##       --seger_path data/segerstolpe/seger.rds --out_dir seger_sim_data
## ============================================================

library(argparse)
library(scRNAseq)
library(SingleCellExperiment)
library(zinbwave)
library(BiocParallel)
library(Matrix)
library(matrixStats)
library(scater)
library(edgeR)

## -----------------------------
## Argument parsing
## -----------------------------
parser <- ArgumentParser(description = "ZINBWave simulation with a dropout-parameter ablation")

parser$add_argument("--dataset", type = "character", default = "baron",
                     choices = c("baron", "segerstolpe"),
                     help = "Reference dataset to fit zinbwave params from [default %(default)s]")
parser$add_argument("--rds_path", type = "character",
                     default = "data/baron/baron_pancreas.rds",
                     help = "Path to the cached Baron SCE .rds or Segerstolpe SCE .rds [default %(default)s].
                     The baron_pancreas.rds is the result of BaronPancreasData() in the SingleCellExperiment package.
                     For the seger.rds, this file is the ouptut of our own upstream Segerstolpe preprocessing pipeline 
                     - this script only loads and filters it further.")
parser$add_argument("--dropout", type = "logical", default = FALSE,
                     help = "Manually inject 50% technical dropout.")
parser$add_argument("--replicate", type = "integer", default = 0,
                     help = paste("Replicate index; seed is 42 + replicate."))
parser$add_argument("--outdir", type = "character", default = "sim_data",
                     help = "Output directory for simulated data [default %(default)s]")


opt <- parser$parse_args()

de_prob     <- as.numeric(strsplit(opt$de_prob, ",")[[1]])
group_prob  <- as.numeric(strsplit(opt$group_prob, ",")[[1]])
batch_cells <- as.integer(strsplit(opt$batch_cells, ",")[[1]])

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)

load_and_preprocess_baron <- function(path) {
  sce <- readRDS(path)
 
  sce_donor1 <- sce[, colData(sce)$donor == "GSM2230757"]
  sce_donor1 <- sce_donor1[rowSums(counts(sce_donor1)) > 10, ]
 
  stats <- perCellQCMetrics(sce_donor1)
  qc <- quickPerCellQC(stats)
  sce_donor1 <- sce_donor1[, !qc$discard]
 
  keep_genes <- rowSums(counts(sce_donor1) > 0) >= 5
  sce_donor1 <- sce_donor1[keep_genes, ]
 
  cell_counts <- table(sce_donor1$label)
  valid_labels <- names(cell_counts[cell_counts >= 3])
  sce_donor1 <- sce_donor1[, sce_donor1$label %in% valid_labels]
 
  group <- as.factor(colData(sce_donor1)[["label"]])
  keep_cells <- !is.na(group) & nzchar(as.character(group))
  sce_donor1 <- sce_donor1[, keep_cells]
  group <- droplevels(group[keep_cells])
 
  tab <- sort(table(group), decreasing = TRUE)
  keep_groups <- names(tab)[tab >= 100]
  sce_donor1 <- sce_donor1[, group %in% keep_groups]
  group <- droplevels(group[group %in% keep_groups])
  message("[baron] Retained groups: ", paste(levels(group), collapse = ", "))
 
  cts <- counts(sce_donor1)
  keep_genes <- rowSums(cts > 0) >= 10
  sce_donor1 <- sce_donor1[keep_genes, ]
 
  log_cts <- log1p(as.matrix(counts(sce_donor1)))
  gene_var <- rowVars(log_cts)
  n_top <- min(2000, nrow(sce_donor1))
  top_genes <- order(gene_var, decreasing = TRUE)[seq_len(n_top)]
  sce_donor1 <- sce_donor1[top_genes, ]
  rm(log_cts, gene_var, top_genes); gc()
 
  counts(sce_donor1) <- as.matrix(counts(sce_donor1))
  colData(sce_donor1)$group <- group
 
  message("[baron] Using ", nrow(sce_donor1), " genes and ", ncol(sce_donor1), " cells")
  sce_donor1
}

load_and_preprocess_segerstolpe <- function(path) {
  sce.seger <- readRDS(path)
 
  cell_counts <- table(sce.seger$CellType)
  keep_types <- names(cell_counts)[cell_counts >= 50]
  sce.seger <- sce.seger[, sce.seger$CellType %in% keep_types]
  sce.seger$CellType <- factor(sce.seger$CellType)
 
  group <- sce.seger$CellType
  keep_cells <- !is.na(group) & nzchar(as.character(group))
  sce.seger <- sce.seger[, keep_cells]
  group <- droplevels(group[keep_cells])
  message("[segerstolpe] Retained cell types: ", paste(levels(group), collapse = ", "))
 
  keep_genes <- rowSums(counts(sce.seger) > 0) >= 10
  sce.seger <- sce.seger[keep_genes, ]
 
  log_cts <- log1p(as.matrix(counts(sce.seger)))
  gene_var <- rowVars(log_cts)
  n_top <- min(2000, nrow(sce.seger))
  top_genes <- order(gene_var, decreasing = TRUE)[seq_len(n_top)]
  sce.seger <- sce.seger[top_genes, ]
  rm(log_cts, gene_var, top_genes); gc()
 
  counts(sce.seger) <- as.matrix(counts(sce.seger))
  colData(sce.seger)$group <- group
 
  message("[segerstolpe] Using ", nrow(sce.seger), " genes and ", ncol(sce.seger), " cells")
  sce.seger
}

sce_ref <- switch(
  opt$dataset,
  baron = load_and_preprocess_baron(
    opt$rds_path,
  ),
  segerstolpe = load_and_preprocess_segerstolpe(
    opt$rds_path,
  ),
  stop("Unknown --dataset: ", opt$dataset)
)
# 2000 top genes are used for the ZINBWave fit to reduce memory usage and speed up the fit.

## -----------------------------
## Fit ZINBWaVE model
## -----------------------------

set.seed(42 + opt$replicate)

bp <- MulticoreParam(workers = 4, progressbar = TRUE)

register(bp)

message("Using BiocParallel backend: ", class(bp)[1], " with ", bpnworkers(bp), " workers")

## Intercept + cell-type labels in X so group structure is modeled directly.
group <- as.factor(colData(sce_ref)[["label"]])
X <- model.matrix(~ group)
colnames(X) <- make.names(colnames(X))

## Minimal gene-level design: intercept only
V <- matrix(1, nrow = nrow(sce_ref), ncol = 1)
colnames(V) <- "Intercept"

## Library-size offset for the mean model
libsize <- colSums(counts(sce_ref))

## Number of latent factors; K=0 keeps only the group structure.
K <- 0

message("Starting zinbFit ... this can take a long time.")
fit <- zinbFit(
  Y = sce_ref,
  X = ~ group,
  K = 0,
  which_assay = "counts",
  commondispersion = TRUE,
  zeroinflation = TRUE,
  verbose = TRUE,
  BPPARAM = bp
)

## -----------------------------
## Simulate from the fitted model
## -----------------------------
if (opt$dropout) {
  message("Injecting 50% technical dropout into the simulated data.")
  zinb_zeroFraction <- zinbSim(fit)$zeroFraction
  print(zinb_zeroFraction)

  pi_mat <- getPi(fit)
  summary(as.vector(pi_mat))

  pi_new <- pmin(pi_mat * 1.5, 0.99)

  mu <- getMu(fit)
  theta <- getTheta(fit)

  sim_counts <- matrix(0, nrow(mu), ncol(mu))

  for(i in seq_len(nrow(mu))) {
    for(j in seq_len(ncol(mu))) {
      
      is_zero <- rbinom(1, 1, pi_new[i,j])
      
      if(is_zero == 1) {
        sim_counts[i,j] <- 0
      } else {
        sim_counts[i,j] <- rnbinom(1, mu = mu[i,j], size = theta[j])
      }
    }
  }
} else{
  sim_counts <- zinbSim(fit)$counts
}

## Convert to genes x cells to make an SCE
sim_counts <- t(sim_counts)
storage.mode(sim_counts) <- "integer"

sim_sce <- SingleCellExperiment(
  assays = list(counts = as(sim_counts, "dgCMatrix"))
)

## Carry over row/col metadata
rownames(sim_sce) <- rownames(sce_ref)
colnames(sim_sce) <- paste0("sim_", seq_len(ncol(sim_sce)))

colData(sim_sce)$group <- group
colData(sim_sce)$source_group <- group
metadata(sim_sce)$fit_group_column <- "label"
metadata(sim_sce)$zinbwave_fit <- list(
  n_genes = nrow(sce_ref),
  n_cells = ncol(sce_ref),
  K = K,
  X_columns = colnames(X)
)

dropout.rate <- (sum(counts_mat == 0) - sum(truecounts_mat == 0)) / sum(truecounts_mat > 0)
message("Dropout rate: ", round(dropout.rate, 4))
message("Overall zero proportion: ", round(sum(counts_mat == 0) / length(counts_mat), 4))
message("Simulated dimensions: ", nrow(sim_sce), " genes x ", ncol(sim_sce), " cells")
message("Simulated group sizes:")
print(table(colData(sim_sce)$group))

if (opt$dropout) {
  custom_zero_fraction <- mean(sim_counts == 0)
  metadata(sim_sce)$dropout <- list(
  zinbSim_default = zinb_zeroFraction,
  scaled_simulation = custom_zero_fraction,
  scaling_factor = 1.5

  message("Dropout comparison:")
  message("  zinbSim default:   ", round(zinb_zeroFraction, 4))
  message("  scaled simulation: ", round(custom_zero_fraction, 4))
)
} else {
  custom_zero_fraction <- NA
}

## -----------------------------
## Write outputs
## -----------------------------
counts(sim_sce) <- as.matrix(counts(sim_sce))

raw_count <- t(counts(sim_sce))  # cells x genes
write.csv(raw_count, file = file.path(opt$outdir, paste0("zinbwave_simulated_", opt$replicate, "_rawcounts.csv")))

dge <- DGEList(counts = counts(sim_sce))
dge <- calcNormFactors(dge, method = "TMM")
effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
sim_sce$tmm_lib_size <- effective_lib_size

write.csv(
  data.frame(cell_id = colnames(sim_sce), tmm_lib_size = sim_sce$tmm_lib_size),
  file.path(opt$outdir, paste0("zinbwave_simulated_", opt$replicate, "_tmm_effective_library_sizes.csv")),
  row.names = FALSE
)

dge <- estimateDisp(dge)
write.csv(
  data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
  file.path(opt$outdir, paste0("zinbwave_simulated_", opt$replicate, "_edgeR_tagwise_dispersion.csv")),
  row.names = FALSE
)

write.csv(
  data.frame(cell_id = colnames(sim_sce), celltype = as.integer(as.factor(sim_sce$group))),
  file.path(opt$outdir, paste0("zinbwave_simulated_", opt$replicate, "_clusters.csv")),
  row.names = FALSE
)

if (opt$dropout){
    dropout_df <- data.frame(
    zinbSim_zeroFraction = zinb_zeroFraction,
    custom_zeroFraction  = custom_zero_fraction
  )

  write.csv(
    dropout_df,
    file.path(opt$outdir, paste0("zinbwave_simulated_", opt$replicate, "_dropout.csv")),
    row.names = FALSE
  )
}

message("Done. Outputs written to ", opt$outdir, " with tag '", opt$replicate, "'")