# A Mechanistic Interpretation of Prescribed-Parameter Convolutional Kernels

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23672114.svg)](https://doi.org/10.5281/zenodo.23672114)

Companion code for *A Mechanistic Interpretation of Prescribed-Parameter Convolutional Kernels*, being prepared as a new version of the preprint [arXiv:2502.17061](https://arxiv.org/abs/2502.17061). This repository supports that forthcoming revision. The experiments also form part of Chapter 3 of the associated thesis.

The experiments examine how the number and length of convolutional kernels affect effective dimensionality (ED), distance correlation (dCor), and classification accuracy.

The implementation adapts ROCKET to retain **proportion of positive values (PPV) features only**. After mean-centring, each kernel is normalised to unit Euclidean norm. This fixes its total squared weight, or energy, across kernel lengths, controlling a source of variation in convolution-response scale.

## Installation

Use Python 3.10. From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The direct dependency versions in `requirements.txt` were captured from the local Python 3.10.22 environment during release preparation. A fresh installation of these requirements was verified with Python 3.10.22 during release preparation. They are not a record of the environment used for the original experiment runs. Numba compiles the convolution routines on first use, so the first transform takes additional time.

For a fully pinned set of direct and indirect dependencies validated on Linux
with Python 3.10.22, install `requirements-lock.txt` in place of
`requirements.txt`. Other platforms have not been validated.

Run all commands below from the repository root.

## Repository structure

```text
src/
  rocket.py                         PPV-only, unit-energy ROCKET transform
  metrics.py                        ED and distance correlation
  data_loader.py                    UCR dataset list and loading
experiments/
  run_dimensionality.py              Experiment runner and accuracy evaluation
  configs/
    experiment_num_kernels.yaml      Kernel-count experiment
    experiment_kernel_length.yaml    Kernel-length experiment
analysis/
  plot_metrics_vs_kernels.py         Mean ED and dCor versus kernel count
  plot_metrics_vs_lengths.py         Mean ED and dCor versus kernel length
  plot_accuracy_vs_lengths.py        Accuracy table and training/test ranks
data/
  metadata.csv                      Dataset lengths and class counts
results/
  metrics_vs_kernels.csv             Kernel-count measurements
  metrics_vs_lengths.csv             Kernel-length measurements
reports/                            Generated figures
```

## Data and experimental protocol

The loader uses the 112-dataset UCR classification collection listed in `src/data_loader.py`. It loads the standard training and test partitions through aeon and downloads datasets into `data/` when necessary. Downloaded datasets are separate from the supplied dataset metadata.

For every dataset and configuration:

1. Generate one random kernel bank using the training series length, and apply that same bank to both splits.
2. Standardise each PPV feature using its training-set mean and standard deviation. Constant training features use a scale factor of one. Apply the fitted transformation to the test features.
3. Fit `RidgeClassifierCV` on the standardised training features, selecting among 10 logarithmically spaced regularisation values from `1e-3` to `1e3`, with accuracy as the selection score. Record training and test accuracy.
4. Calculate ED and dCor on the standardised training representation. Test observations are not used to calculate these geometric metrics.

ED is the covariance participation ratio, `(sum eigenvalues)^2 / sum(eigenvalues^2)`. dCor compares the original training series with their standardised PPV representations using double-centred Euclidean distance matrices.

The convolution uses randomly selected zero padding or no padding. The exact circular-translation-invariance result in the chapter assumes periodic boundaries; it is not an exact invariance guarantee for these finite-boundary experiments.

## Run the experiments

### Number of kernels

```bash
python experiments/run_dimensionality.py --config experiments/configs/experiment_num_kernels.yaml
```

The default configuration evaluates 100, 500, 1,000, 2,000, 5,000 and 10,000 kernels on all 112 datasets. Each kernel length is sampled uniformly from `{7, 9, 11}`. Results are written to `results/metrics_vs_kernels.csv`.

### Kernel length

```bash
python experiments/run_dimensionality.py --config experiments/configs/experiment_kernel_length.yaml
```

The default configuration evaluates odd lengths from 3 to 27, with 10,000 kernels at each length. All kernels within a configuration have the same length. Results are written to `results/metrics_vs_lengths.csv`.

The runner permits configurations with kernel length greater than series length. The reported kernel-length analysis instead uses a **fixed cohort of 109 datasets with series length at least 27**. The plotting script reads lengths from `data/metadata.csv` and excludes all configurations for SmoothSubspace, Chinatown and ItalyPowerDemand. The same cohort is used for the accuracy comparison.

The runner writes progress after each configuration. Starting another run with the same experiment name overwrites the corresponding CSV; it does not resume a previous run. Preserve the existing result files before rerunning if you want to retain them.

## Generate the geometric figures

Using the result CSVs and `data/metadata.csv`:

```bash
python analysis/plot_metrics_vs_kernels.py --input results/metrics_vs_kernels.csv
python analysis/plot_metrics_vs_lengths.py --input results/metrics_vs_lengths.csv
```

Figures are saved under `reports/metrics_vs_kernels/` and `reports/metrics_vs_lengths/`. Each script produces absolute and relative ED and dCor figures. The optional class-count-stratified figures are skipped with the supplied metadata because its `NumberClasses` column is not recognised by the existing plot scripts.

The curves show **means across datasets**. Relative metrics are first divided by the maximum observed for each dataset within the corresponding experiment. Shading shows pointwise 95% bootstrap confidence intervals for the means across datasets. These intervals describe variation across datasets, not variation across repeated random kernel banks.

## Reproduce the accuracy table

The following example calculates the kernel-length table from the CSV, using the same 109-dataset cohort. Rank one denotes the highest accuracy, and ties receive average ranks within each dataset.

```bash
python - <<'PY'
import pandas as pd

lengths = pd.read_csv('data/metadata.csv').set_index('Dataset')['Length']
df = pd.read_csv('results/metrics_vs_lengths.csv')
df = df.loc[(df['num_kernels'] == 10000) & (df['dataset'].map(lengths) >= 27)].copy()
df['kernel_length'] = df['kernel_lengths'].str.extract(r'\[(\d+)\]', expand=False).astype(int)

for split in ['Train', 'Test']:
    df[f'{split}_rank'] = df.groupby('dataset')[f'Accuracy_{split}'].rank(
        ascending=False, method='average'
    )

table = df.groupby('kernel_length')[
    ['Train_rank', 'Test_rank', 'Accuracy_Train', 'Accuracy_Test']
].mean().sort_index()
print(table.round({'Train_rank': 2, 'Test_rank': 2,
                   'Accuracy_Train': 4, 'Accuracy_Test': 4}).to_string())
PY
```

To save the accuracy table and generate the training/test mean-rank figure:

```bash
python analysis/plot_accuracy_vs_lengths.py
```

Outputs are saved to `reports/accuracy_vs_lengths/`. This script uses the same
109-dataset cohort and average-rank treatment of ties as the table above.

## Notes on running the analyses

The supplied result CSVs contain the measurements used in the manuscript and can be used directly to reproduce the analyses. Each experiment uses one random kernel bank per dataset and configuration, so numerical results may vary between runs. Bootstrap confidence intervals may also vary slightly between plotting runs.

The analyses use the full training sets. ED and pairwise-distance calculations can require substantial memory and computation for larger datasets.

## Software author and citation

The software author is **Jorge Marco Blanco**. Citation metadata is provided in
[`CITATION.cff`](CITATION.cff). The archived release is **v1.0.0**, published on 26 October 2026.
Cite the version DOI: [10.5281/zenodo.23672114](https://doi.org/10.5281/zenodo.23672114).

The upstream ROCKET authors and their work are acknowledged below.

## License

This distribution uses the GNU General Public License, version 3
(`GPL-3.0-only`). See [`LICENSE`](LICENSE) for the terms and
[`NOTICE.md`](NOTICE.md) for upstream attribution and modification notices.

## Attribution

The ROCKET adaptation is based on the implementation by Angus Dempster, François Petitjean and Geoffrey I. Webb:

- [Original ROCKET repository](https://github.com/angus924/rocket)
- Dempster, A., Petitjean, F. and Webb, G. I. (2020). *ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels*. Data Mining and Knowledge Discovery, 34, 1454–1495. [DOI](https://doi.org/10.1007/s10618-020-00701-z).

The adaptation retains PPV only, supports configurable kernel lengths, and normalises centred kernel weights to unit energy. The reference implementation and this adaptation therefore define different feature constructions.
