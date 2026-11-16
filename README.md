# scFLAME

scFLAME is a probabilistic model for single-cell RNA-seq that clusters cells and learns a low-dimensional embedding of them **at the same time**, working directly on raw counts. It is used in place of the usual normalise → PCA → neighbours → Leiden steps, and its results load straight back into Seurat or Scanpy.

For each cell you get:
- **a cluster**
- **latent factors**, a small number of values per cell that you use like PCs (UMAP, neighbours, trajectories, …).

For each cluster you get:
- **marker genes**, ranked by how much the cluster shifts each gene's expression away from the average cell.

Optionally, scFLAME can **over-cluster and merge** clusters step by step. This gives a hierarchy (dendrogram) of subtypes, which is useful when you don't know how many populations to expect.

Paper: see the preprint on [bioRXiv](https://www.biorxiv.org/content/10.64898/2026.09.04.749363v1)

## Installation

```
pip install -r requirements.txt
```

Preprocessing and plotting have their own optional dependencies. Install the set for the side you use:
- **R:** Seurat, edgeR, ggplot2, dplyr, stringr
- **Python:** scanpy, scikit-misc

Note R preprocessing and plotting was used for the paper.

## Quick start

### 1. Preprocess

This step runs QC, selects highly variable genes, and writes the input files scFLAME needs. Original analyses were all run in R for the paper.

```
# Python / Scanpy: .h5ad, 10x .h5, or a 10x / Cell Ranger output directory
python scripts/preprocess_scanpy.py --input sample1/outs --out-dir data/sample1 \
    --metadata annotations.csv --label-key cell_type

# R / Seurat: edit the configuration block at the top, then
Rscript scripts/preprocess.R
```

Cell-type labels are optional. If you have them, scFLAME also reports how well its clusters agree with them.

### 2. Fit scFLAME

```
python scripts/run_realdata.py --data-dir data/sample1 --out-dir results/sample1 \
    --n-clusters 10 --save-top-genes
```

Without `--n-clusters`, K is set to the number of labels in `clusters.csv`. If you don't know K, see [Choosing the number of clusters](#choosing-the-number-of-clusters).

### 3. Look at the results

```
# Seurat: edit the paths at the top, then
Rscript scripts/downstream_seurat.R

# Scanpy
python scripts/downstream_scanpy.py --adata data/sample1/processed.h5ad \
    --results-dir results/sample1 --label-key cell_type
```

Both scripts do the same things:
- pick the best fit
- add the clusters to the cell metadata
- store the latent factors as a reduction (`scflame_latent` in Seurat, `obsm["X_scflame"]` in Scanpy) and run UMAP on them
- save plots to `results/sample1/plots/`

The saved plots are:
- a UMAP coloured by cluster, next to your existing labels
- a heatmap of how each cluster splits across your labels
- a dot plot of each cluster's top marker genes

The object is saved with everything added, so you can carry on with your usual analysis.

## Choosing the number of clusters

There are two options:

1. **Fit at a fixed K** with `run_realdata.py --n-clusters K`. This is the simplest option. You can try a few values and compare them.
2. **Over-cluster and merge** with `run_merge.py`. Start from more clusters than you expect (`--k-init`) and merge the most similar pair repeatedly until `--k-final` remain:

   ```
   python scripts/run_merge.py --data-dir data/sample1 --out-dir results/sample1_merge \
       --k-init 15 --k-final 2 --criterion icl
   ```

   The clusters at every K along the way are saved, so a single run gives you all resolutions. Draw the hierarchy with `scripts/plot_merge_tree.R`, which can name the leaves by the majority cell type in each starting cluster. The height of each join shows how much model fit is lost by that merge.

   To load the clusters at a chosen K into Scanpy, add `--merge-dir results/sample1_merge --merge-kinit 15 --merge-k 8` to `downstream_scanpy.py`. In R, `plot_merge_tree.R` draws the matching UMAP if you give it a Seurat object.

## Input files

`--data-dir` must contain these files. Both preprocessing scripts write them for you.

| File | Contents | Required? |
|---|---|---|
| `counts.csv` | Raw counts, cells × genes. The first column is the cell ID. | Yes |
| `library_sizes.csv` | Columns `cell_id`, `tmm_lib_size` (edgeR TMM library sizes from `preprocess.R`, or total counts per cell from `preprocess_scanpy.py`) | Optional |
| `dispersion.csv` | Columns `gene`, `dispersion` (per-gene negative binomial dispersion φ, edgeR convention) | Optional |
| `clusters.csv` | Columns `cell_id`, `celltype` (integer-coded labels), and optionally `batch_id` for batch correction | Optional with `--n-clusters` |

The counts must be **raw** (integers), not normalised or log-transformed.

`--n-hvgs` in the run scripts should be no larger than the number of genes in `counts.csv`.

## Outputs

`run_realdata.py` writes the following to `--out-dir`:

| File | Contents |
|---|---|
| `scflame_summary.csv` | One row per repeat: number of clusters, ELBO, and ARI/NMI/accuracy if labels were given |
| `scflame_allocations.csv` | For each cell: its cluster in each repeat |
| `scflame_latents_rep{n}.csv` | The latent factors of each cell for repeat *n* |
| `scflame_top_genes.csv` | Top marker genes per cluster (with `--save-top-genes`) |
| `tsne.png` | Quick t-SNE of the latent factors |

**Which repeat should I use?** scFLAME is fitted several times from different starting points (`--n-repeats`, default 3). Use the repeat with the **highest ELBO**, which is the best fit to the data. The downstream scripts select it automatically.

`run_merge.py` writes the following:
- `merge_history_{seed}.csv`: each merge step
- `cluster_allocations_{seed}.csv`: the clusters at every K
- `merge_latents_Kinit{K}_{seed}.csv`: the latent factors from each `--k-init` run

## Main options

Most users only need these:

| Option | Default | What it does |
|---|---|---|
| `--n-clusters` | number of labels | Number of clusters K to fit (`run_realdata.py`) |
| `--k-init`, `--k-final` | 10 15, 2 | Starting and final K for merging (`run_merge.py`) |
| `--latent-dim` | 10 | Number of latent factors (analogous to the number of PCs). 15 is used for real data in the paper |
| `--n-hvgs` | 3000 / 5000 | Number of highly variable genes used |
| `--n-repeats` | 3 | Number of independent fits |
| `--save-top-genes` | off | Save marker genes per cluster |

Everything else (`--nbfa-epochs`, `--scflame-epochs`, `--mc-samples`, …) can be left at its default.

## Repository layout

```
scflame/                  The model (you should not need to touch this to use scFLAME)
  model.py                  NBFA warm-up and scFLAME training
  merging.py                Greedy cluster merging
  utils.py                  Data loading, likelihoods and marker-gene scoring
  viz.py                    t-SNE plotting helpers
scripts/                  Everything needed to run scFLAME on your own data
  preprocess.R              Preprocessing in R / Seurat (edgeR library sizes and dispersions)
  preprocess_scanpy.py      Preprocessing in Python / Scanpy (.h5ad or 10x input)
  run_realdata.py           Fit scFLAME at a given K
  run_merge.py              Over-cluster and merge
  downstream_seurat.R       Load results into Seurat and plot them
  downstream_scanpy.py      Load results into Scanpy and plot them
  plot_merge_tree.R         Dendrogram of the merge hierarchy
  run_simulation.py         Simulation study from the paper
paper/                    Scripts that reproduce the analyses and figures in the paper.
                          These are kept for reproducibility and are less tidy than scripts/.
                          To use scFLAME on your own data, start from scripts/ instead.
```

## Docker

To simplify setup, a docker container has been provided. For help with setting docker up, please consult the [documentation](https://docs.docker.com/desktop/).

Once you are setup with Docker, copy `docker run -it --rm -v "$PWD/scflame:/app/scflame" scflame` into your preferred terminal application.

If using Windows Powershell, replace the above `$PWD` with `${PWD}`.

Inside docker you will be in a Linux terminal and will be able to execute the Python or R scripts to run the model, according to their respective instructions.

For example:
```
$> python scripts/run_merge.py --data-dir data/baron --out-dir results/baron --k-init 10 15 --k-final 2 --criterion icl
```

Important things to note:

Docker containers do not retain information. Exiting the image will result in the loss of all data or changes unless moved out of the container.

Docker runs Linux inside. It uses WSL (Windows Subsystem for Linux) to achieve this on Windows machines. After closing docker, memory usage may stay
at a very high usage, as simply closing Docker Desktop does not release the VMs memory back. To force close, in your terminal enter `WSL --shutdown`.