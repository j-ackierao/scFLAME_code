## ============================================================
## generate_data_splat.R
## Simulate data from the Splat simulation model, using either the Baron or Segerstolpe 
## datasets as reference parameters. Reproduces same simulation as the manuscript.
## This script is intended to be run as a SLURM array job, with each task simulating a different replicate.
## Replicates 0-19 were used to generate the 20 replicates in the manuscript; we set SLURM_ARRAY_TASK_ID to 1-20.
## We recommend for every set of parameters to save data in a different folder.
##
## Usage:
##   Rscript simulate_splat.R --dataset baron --dropout_mid 0.5 --replicate 0 --out_dir baron_sim_data
##   Rscript simulate_splat.R --dataset segerstolpe --dropout_mid 1.0 --replicate 3 \
##       --seger_path data/segerstolpe/seger.rds --out_dir seger_sim_data
## ============================================================

library(argparse)
library(SingleCellExperiment)
library(splatter)
library(BiocParallel)
library(Matrix)
library(matrixStats)
library(scater)
library(edgeR)

## -----------------------------
## Argument parsing
## -----------------------------
parser <- ArgumentParser(description = "Splat simulation with a dropout-parameter ablation")

parser$add_argument("--dataset", type = "character", default = "baron",
                     choices = c("baron", "segerstolpe"),
                     help = "Reference dataset to fit splat params from [default %(default)s]")
parser$add_argument("--rds_path", type = "character",
                     default = "data/baron/baron_pancreas.rds",
                     help = "Path to the cached Baron SCE .rds or Segerstolpe SCE .rds [default %(default)s].
                     The baron_pancreas.rds is the result of BaronPancreasData() in the SingleCellExperiment package.
                     For the seger.rds, this file is the ouptut of our own upstream Segerstolpe preprocessing pipeline 
                     - this script only loads and filters it further.")
parser$add_argument("--dropout_type", type = "character", default = NA,
                     choices = c("none", "experiment", "batch", "group", "cell"),
                     help = "splatter dropout.type; omit for no dropout.")
parser$add_argument("--dropout_mid", type = "double", default = NA,
                     help = "Override splatter dropout.mid; omit to keep the data-driven fit")
parser$add_argument("--de.facScale", type = "double", default = NA,
                     help = "Override splatter de.facScale for clustering signal strength; omit to keep the data-driven fit")
parser$add_argument("--de_prob", type = "character", default = "0.3,0.2,0.1,0.2,0.2",
                     help = "Comma-separated de.prob vector [default %(default)s]")
parser$add_argument("--group_prob", type = "character", default = "0.4,0.25,0.15,0.1,0.1",
                     help = "Comma-separated group.prob vector [default %(default)s]")
parser$add_argument("--batch_facLoc", type = "double", default = 0,
                     help = "batch.facLoc; 0 disables batch effects [default %(default)s]")
parser$add_argument("--batch_facScale", type = "double", default = 0,
                     help = "batch.facScale; 0 disables batch effects [default %(default)s]")
parser$add_argument("--batch_cells", type = "character", default = "1000",
                     help = "Comma-separated batchCells vector, e.g. '1000' or '400,350,250' [default %(default)s]")
parser$add_argument("--replicate", type = "integer", default = 0,
                     help = paste("Replicate index; seed is 42 + replicate."))
parser$add_argument("--outdir", type = "character", default = "sim_data",
                     help = "Output directory for simulated data [default %(default)s]")


opt <- parser$parse_args()

de_prob     <- as.numeric(strsplit(opt$de_prob, ",")[[1]])
group_prob  <- as.numeric(strsplit(opt$group_prob, ",")[[1]])
batch_cells <- as.integer(strsplit(opt$batch_cells, ",")[[1]])

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)

## -----------------------------
## Load + preprocess the chosen reference dataset (deterministic, no RNG)
## -----------------------------
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
### TWO OPTIONS HAVE BEEN TAKEN OUT, MAKE SURE THESE STAY OOUT

## -----------------------------
## Fit Splat parameters then apply any ablation overrides on top of the fit.
## -----------------------------
params <- splatEstimate(sce_ref)

if (!is.na(opt$dropout_mid))   params <- setParam(params, "dropout.mid",   opt$dropout_mid)
if (!is.na(opt$dropout_type)) params <- setParam(params, "dropout.type", opt$dropout_type)
if (!is.na(opt$de_facScale))  params <- setParam(params, "de.facScale",  opt$de_facScale)

## -----------------------------
## Simulate data
## -----------------------------

set.seed(42 + opt$replicate)

sim_sce <- splatSimulate(
  params = params,
  batchCells = batch_cells,
  nGenes = 4000,
  group.prob = group_prob,
  method = "groups",
  de.prob = de_prob,
  de.facLoc = 0.2,
  batch.facLoc = opt$batch_facLoc,
  batch.facScale = opt$batch_facScale,
  verbose = FALSE
)

sim_sce <- logNormCounts(sim_sce)
sim_sce <- runPCA(sim_sce)

counts_mat     <- assay(sim_sce, "counts")
truecounts_mat <- assay(sim_sce, "TrueCounts")

dropout.rate <- (sum(counts_mat == 0) - sum(truecounts_mat == 0)) / sum(truecounts_mat > 0)
message("Dropout rate: ", round(dropout.rate, 4))
message("Overall zero proportion: ", round(sum(counts_mat == 0) / length(counts_mat), 4))
message("Simulated dimensions: ", nrow(sim_sce), " genes x ", ncol(sim_sce), " cells")
message("Simulated group sizes:")
print(table(colData(sim_sce)$Group))


## -----------------------------
## Write outputs
## -----------------------------
counts(sim_sce) <- as.matrix(counts(sim_sce))

raw_count <- t(counts(sim_sce))  # cells x genes
write.csv(raw_count, file = file.path(opt$outdir, paste0("splat_simulated_", opt$replicate, "_rawcounts.csv")))

dge <- DGEList(counts = counts(sim_sce))
dge <- calcNormFactors(dge, method = "TMM")
effective_lib_size <- dge$samples$lib.size * dge$samples$norm.factors
sim_sce$tmm_lib_size <- effective_lib_size

write.csv(
  data.frame(cell_id = colnames(sim_sce), tmm_lib_size = sim_sce$tmm_lib_size),
  file.path(opt$outdir, paste0("splat_simulated_", opt$replicate, "_tmm_effective_library_sizes.csv")),
  row.names = FALSE
)

dge <- estimateDisp(dge)
write.csv(
  data.frame(gene = rownames(dge), dispersion = dge$tagwise.dispersion),
  file.path(opt$outdir, paste0("splat_simulated_", opt$replicate, "_edgeR_tagwise_dispersion.csv")),
  row.names = FALSE
)

write.csv(
  data.frame(cell_id = colnames(sim_sce), celltype = as.integer(as.factor(sim_sce$Group))),
  file.path(opt$outdir, paste0("splat_simulated_", opt$replicate, "_clusters.csv")),
  row.names = FALSE
)

message("Done. Outputs written to ", opt$outdir, " with tag '", opt$replicate, "'")