#!/usr/bin/env python3
"""Run the full reproducibility pipeline end to end.

All outputs are written to ``results/`` (git-ignored). Nothing in this repository
contains precomputed results, figures or manuscript text; they are regenerated
here from the shared dataset.

Usage:
    python run_all.py            # full pipeline (NSGA-II uses 10 seeds)
    python run_all.py --quick    # faster smoke run (NSGA-II uses 2 seeds)
"""
import sys
import time
from nanolube import (data_prep, benchmark_loso, residual_envelope, ann_ensemble,
                      paired_bootstrap, leakage_contrast, importance_dropcol, nsga2_screen)


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    step = lambda s: print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)

    step("1/8  Data preparation (modelling frames + manifest)")
    data_prep.build()
    step("2/8  Study-balanced LOSO benchmark")
    benchmark_loso.run()
    step("3/8  Empirical residual (stress) envelope")
    residual_envelope.run()
    step("4/8  Five-seed ANN ensemble")
    ann_ensemble.run()
    step("5/8  Paired study-cluster bootstrap + within-study coupling")
    paired_bootstrap.run()
    step("6/8  Matched leakage contrast (grouped vs row-random)")
    leakage_contrast.run()
    step("7/8  Grouped drop-column feature relevance")
    importance_dropcol.run()
    step("8/8  Exploratory NSGA-II feasibility-filtered screen")
    nsga2_screen.run(seeds=range(2) if quick else range(10))

    print(f"\nDone in {time.time() - t0:.0f}s. All outputs are in results/.")


if __name__ == "__main__":
    main()
