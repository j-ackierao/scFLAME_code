# scFLAME

Code for scFLAME: a probabilistic generative model for scRNA-seq for joint dimensionality reduction and clustering.

## Layout

```
scflame/
  utils.py     Required probabilistic functions and scoring functions
  model.py     NBFA warm-up and main scFLAME training function:
  merging.py   Optional post-hoc greedy cluster merging
  viz.py       t-SNE plotting helpers
scripts/
  run_realdata.py   Fit scFLAME to one dataset at a given K and report ARI/NMI/ACC
  run_merge.py      scFLAME with greedy merge applied
```

## Installation

```
pip install -r requirements.txt
```

## Usage

Fit scFLAME directly at a given K:

```
python scripts/run_realdata.py --data-dir data/segerstolpe --out-dir results/segerstolpe
```

Over-cluster and merge down to derive a hierarchy of clusters:

```
python scripts/run_merge.py --data-dir data/baron --out-dir results/baron \
    --k-init 10 15 --k-final 2 --criterion icl
```

Both scripts expect `counts.csv`, `clusters.csv`, `dispersion.csv`, and
`library_sizes.csv` under `--data-dir` -- see the docstring at the top of
each script for the exact expected columns.