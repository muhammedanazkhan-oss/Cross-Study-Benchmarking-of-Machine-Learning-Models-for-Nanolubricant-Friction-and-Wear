"""Empirical grouped residual (stress) envelope.

One conformity score per study (the within-study 90th percentile of the absolute
LOSO residual, or the max if fewer than three rows); the global half-width q is
the study-level quantile of those scores. Because the same out-of-study residual
collection determines both q and the reported conformity, this is a conservative
stress test rather than a calibrated prediction interval.
"""
import json
import numpy as np
import pandas as pd
from .common import RESULTS

ALPHA = 0.1


def run(targets=("cof", "wsd")):
    out = {}
    for target in targets:
        od = pd.read_csv(RESULTS / f"oof_{target}.csv")
        y = od["g_true"].values
        groups = od["ref"].values
        resid = np.abs(y - od["loso_Random"].values)
        uniq = np.unique(groups)
        S = len(uniq)
        scores = np.array([np.quantile(resid[groups == s], 0.9) if (groups == s).sum() >= 3
                           else resid[groups == s].max() for s in uniq])
        k = min(S, int(np.ceil((S + 1) * (1 - ALPHA))))
        q = float(np.sort(scores)[k - 1])
        row_cov = float(np.mean(resid <= q))
        sb_cov = float(np.mean([np.mean(resid[groups == s] <= q) for s in uniq]))
        out[target] = dict(
            method="study-level cluster CV+: one score per study (within-study 90th-pct |resid|), "
                   "quantile over studies; exchangeable unit = study",
            q_halfwidth_g=q, n_calibration_studies=int(S), alpha=ALPHA,
            row_coverage=row_cov, study_balanced_coverage=sb_cov, rank_index=k)
        print(f"{target}: q={q:.2f} (studies={S}); row cov {row_cov:.2f}, study-balanced cov {sb_cov:.2f}")
    json.dump(out, open(RESULTS / "residual_envelope.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    run()
