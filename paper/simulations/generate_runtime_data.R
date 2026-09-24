## ============================================================
## generate_runtime_data.R
## Generates one Splat-simulated dataset of a given size (--n_cells),
## fit to Baron pancreas data, for the runtime/ARI scaling study.
##
## Cells are generated in fixed-size batches (--cells_per_batch,
## default 5000) and concatenated, purely to bound peak memory during
## splatSimulate().
## ============================================================

suppressPackageStartupMessages({
  library(scRNAseq)
  library(SingleCellExperiment)
  library(splatter)
  library(Matrix)
  library(argparse)
})

source("generate_data_splat.R")   # load_and_preprocess_baron()

parser <- ArgumentParser(description = "Generate one Splat scaling-study dataset")
parser$add_argument("--n_cells", type = "integer", required = TRUE,
                     help = "Total number of cells to simulate")
parser$add_argument("--cells_per_batch", type = "integer", default = 5000,
                     help = "Cells per splatSimulate() call, for memory bounding [default %(default)s]")
parser$add_argument("--n_genes", type = "integer", default = 4000)
parser$add_argument("--group_prob", type = "character", default = "0.4,0.25,0.15,0.1,0.1")
parser$add_argument("--de_prob", type = "character", default = "0.3,0.2,0.1,0.2,0.2")
parser$add_argument("--de_facLoc", type = "double", default = 0.2)
parser$add_argument("--baron_path", type = "character", default = "data/baron_pancreas.rds")
parser$add_argument("--replicate", type = "integer", default = NA,
                     help = "Replicate index; seed is 42 + replicate. Defaults to $SLURM_ARRAY_TASK_ID - 1 if unset [default: env-derived]")
parser$add_argument("--outdir", type = "character", default = "sim_data/run_times")

opt <- parser$parse_args()
if (is.na(opt$replicate)) {
  opt$replicate <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", unset = "0")) - 1
}
dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)

set.seed(42 + opt$replicate)

## -----------------------------
## Fit Splat params to Baron (shared preprocessing - see preprocess_baron.R)
## -----------------------------
sce_donor1 <- load_and_preprocess_baron(opt$baron_path)
counts(sce_donor1) <- as.matrix(counts(sce_donor1))
params <- splatEstimate(sce_donor1)

## -----------------------------
## Simulate in fixed-size batches until n_cells is reached, then trim
## to exactly n_cells.
## -----------------------------
n_batches <- ceiling(opt$n_cells / opt$cells_per_batch)
sim_list <- vector("list", n_batches)

for (i in seq_len(n_batches)) {
  message("Simulating batch ", i, " of ", n_batches)
  sim_list[[i]] <- splatSimulate(
    params = params,
    batchCells = opt$cells_per_batch,
    nGenes = 4000,
    group.prob = c(0.4,0.25,0.15,0.1,0.1),
    method = "groups",
    de.prob = c(0.3,0.2,0.1,0.2,0.2),
    de.facLoc = 0.2,
    verbose = FALSE
  )
  gc()
}

counts_all <- do.call(cbind, lapply(sim_list, counts))
counts_all <- counts_all[, seq_len(opt$n_cells), drop = FALSE]  # trim to exact n_cells
cluster_labels <- unlist(lapply(sim_list, function(x) colData(x)$Group))[seq_len(opt$n_cells)]

cts <- as(counts_all, "CsparseMatrix")
cell_ids <- paste0("cell_", seq_len(ncol(cts)))
colnames(cts) <- cell_ids

out_prefix <- file.path(opt$outdir, sprintf("n%d_%d", opt$n_cells, opt$replicate))

Matrix::writeMM(cts, paste0(out_prefix, "_counts.mtx"))

## Use a raw total-count library size (colSums), not TMM - purely inspecting run-time
write.csv(
  data.frame(cell_id = cell_ids, raw_lib_size = Matrix::colSums(cts)),
  paste0(out_prefix, "_libsize.csv"), row.names = FALSE
)

write.csv(
  data.frame(cell_id = cell_ids, celltype = as.integer(as.factor(cluster_labels))),
  paste0(out_prefix, "_clusters.csv"), row.names = FALSE
)

message("Done. Wrote ", opt$n_cells, "-cell dataset to ", out_prefix, "_*")