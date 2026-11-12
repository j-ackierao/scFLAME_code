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
python scripts/run_merge.py --data-dir data/baron --out-dir results/baron --k-init 10 15 --k-final 2 --criterion icl
```

Both scripts expect `counts.csv`, `clusters.csv`, `dispersion.csv`, and
`library_sizes.csv` under `--data-dir` -- see the docstring at the top of
each script for the exact expected columns.

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

