# Cross-study machine-learning benchmark for nanolubricant friction and wear

Code and dataset for a study-disjoint (leave-one-study-out) evaluation of
machine-learning models that predict friction and wear response of nanoparticle
lubricant additives, with a feasibility-filtered NSGA-II design screen.

> **Manuscript status.** This repository accompanies a manuscript currently under
> review. The manuscript text, figures and result tables are **intentionally not
> included**. Running the pipeline regenerates every numerical output locally into
> `results/` (which is git-ignored). Before making the repository public, review
> `CITATION.cff` and `.zenodo.json` (authors, ORCIDs, title, year).

## What this repository does

The pipeline takes a curated dataset of published nanolubricant experiments and:

1. **prepares** target-specific modelling frames and log response ratios;
2. **benchmarks** six estimators (study-balanced mean baseline, ridge, elastic
   net, random forest, XGBoost, and a five-seed neural-network ensemble) under
   **study-balanced leave-one-study-out (LOSO)** validation with fold-internal
   preprocessing and a single study-balanced RMSE estimand;
3. quantifies a **matched leakage contrast** (grouped vs row-random 5-fold);
4. reports **grouped drop-column** feature relevance (descriptive);
5. builds an **empirical residual (stress) envelope** for out-of-study error;
6. runs an **exploratory NSGA-II** search with a feasibility/support filter.

All estimands, hyper-parameters and the feature set are defined once in
`nanolube/common.py`.

## Repository layout

```
nanolubricant-ml-benchmark/
├── data/
│   ├── nanolubricant_dataset.csv     # 196 verified datapoints from 60 studies
│   ├── data_sources.csv              # per-study DOI attribution
│   └── data_dictionary.md            # column-by-column description
├── nanolube/                         # clean, maintained pipeline (import as a package)
│   ├── common.py                     # config, loaders, preprocessing, models
│   ├── data_prep.py                  # modelling frames + log response ratios
│   ├── benchmark_loso.py             # study-balanced LOSO benchmark + metrics
│   ├── ann_ensemble.py               # five-seed ANN ensemble
│   ├── paired_bootstrap.py           # paired study-cluster bootstrap; within-study coupling
│   ├── leakage_contrast.py           # grouped vs row-random matched contrast
│   ├── importance_dropcol.py         # grouped drop-column relevance (Holm)
│   ├── residual_envelope.py          # study-level cluster CV+ envelope
│   └── nsga2_screen.py               # NSGA-II + feasibility/support filter
├── archive/                          # original working scripts (verbatim, for transparency)
├── results/                          # regenerated outputs (git-ignored)
├── run_all.py                        # run the whole pipeline
├── requirements.txt
├── LICENSE                           # MIT (code)
└── LICENSE-DATA.md                   # CC BY 4.0 (data)
```

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline from the repository root:

```bash
python run_all.py            # full run (NSGA-II uses 10 seeds)
python run_all.py --quick    # faster smoke run (NSGA-II uses 2 seeds)
```

Or run any stage on its own (outputs land in `results/`):

```bash
python -m nanolube.data_prep
python -m nanolube.benchmark_loso
python -m nanolube.residual_envelope
python -m nanolube.ann_ensemble
python -m nanolube.paired_bootstrap
python -m nanolube.leakage_contrast
python -m nanolube.importance_dropcol
python -m nanolube.nsga2_screen
```

`data_prep` must be run first (the other modules read the modelling frames it
writes). `run_all.py` handles the ordering for you.

## Clean module ↔ original script

| Clean module (`nanolube/`) | Original script (`archive/`) |
|---|---|
| `common.py` | `common2.py` |
| `data_prep.py` | `build_dataset2.py`, `manifest.py` |
| `benchmark_loso.py` | `pipe5.py`, `stats5.py` |
| `ann_ensemble.py` | `fix6a.py` |
| `paired_bootstrap.py` | `fix5b.py` |
| `leakage_contrast.py` | `fix5c.py` |
| `importance_dropcol.py` | `dropcol5.py`, `fix6b.py` |
| `residual_envelope.py` | `conformal5.py` |
| `nsga2_screen.py` | `nsga5_run.py`, `nsga5_agg.py` |

## Reproducibility

All random seeds (model, splitter, bootstrap and optimiser) are fixed in the
code. The reported numbers were produced with Python 3.10, scikit-learn 1.7.2 and
XGBoost 3.2.0 (see `requirements.txt`). Small differences across BLAS builds or
library patch versions are possible but do not affect the conclusions.

## Data

See `data/README.md` and `data/data_dictionary.md`. The dataset holds only
extracted experimental values and their provenance — no model outputs.

## License

Code: **MIT** (`LICENSE`). Data: **CC BY 4.0** (`LICENSE-DATA.md`).

## Citing

If you use this code or data, please cite the archive via `CITATION.cff` and
credit the original experimental studies listed in `data/data_sources.csv`.
