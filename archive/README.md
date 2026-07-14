# archive/ — original working scripts

These are the actual iteration scripts used during the study, kept verbatim for
full transparency. **The maintained, runnable entry point is the `nanolube/`
package** (see the top-level `README.md`); the clean pipeline reproduces every
quantity these scripts computed.

Each script here is pure computation (no manuscript text, no precomputed results
are stored). They were run from a single working directory and refer to
intermediate files by their original names (for example `master_final.csv`,
`model_cof2.csv`, `oof_cof5.csv`), so they are **not wired to run standalone**
from this repository — they are provided for inspection and provenance.

| Script | Role |
|---|---|
| `common2.py` | Shared feature set, preprocessing and estimator definitions. |
| `build_dataset2.py` | Build target-specific modelling frames + log response ratios. |
| `manifest.py` | Filtering cascade, feature dictionary, provenance counts. |
| `pipe5.py` | Weighted LOSO + row-random out-of-fold predictions. |
| `stats5.py` | Study-balanced SB-RMSE, skill, bootstrap CIs, paired win probabilities. |
| `fix5b.py` | Paired RF-vs-baseline SB-MSE bootstrap; valid within-study correlation; NSGA cell frequencies. |
| `fix5c.py` | Matched leakage contrast (grouped vs row-random 5-fold). |
| `fix6a.py` | Five-seed ANN ensemble. |
| `fix6b.py` | Unified-RF grouped drop-column relevance. |
| `fix6c.py` | Imputation / specification sensitivity. |
| `dropcol5.py` | Grouped drop-column feature relevance with Holm adjustment. |
| `conformal5.py` | Study-level cluster CV+ residual envelope. |
| `nsga5_run.py` | NSGA-II search with the feasibility / support filter. |
| `nsga5_agg.py` | Pool seeds, deduplicate cells, hypervolume, class frequencies. |

The correspondence to the clean modules is documented in the top-level `README.md`.
