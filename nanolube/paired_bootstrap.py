"""Paired study-cluster bootstrap and valid within-study coupling.

(1) Paired RF-vs-baseline difference in study-balanced MSE, resampling studies
    with replacement, conditional on the fixed out-of-fold predictions.
(2) Within-study COF/WSD coupling: a pooled study-centred correlation (rows
    weighted 1/n_s) with a study-cluster bootstrap interval, and a Fisher-z
    summary restricted to studies with at least four paired conditions.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from .common import RESULTS


def _paired(target):
    od = pd.read_csv(RESULTS / f"oof_{target}.csv")
    y = od["g_true"].values
    groups = od["ref"].values
    uniq = np.unique(groups)
    idx = [np.where(groups == s)[0] for s in uniq]
    S = len(uniq)
    rng = np.random.default_rng(1)
    mse_rf = np.array([mean_squared_error(y[ix], od["loso_Random"].values[ix]) for ix in idx])
    mse_du = np.array([mean_squared_error(y[ix], od["loso_Dummy"].values[ix]) for ix in idx])
    SEL = rng.integers(0, S, (5000, S))
    diff = mse_rf[SEL].mean(1) - mse_du[SEL].mean(1)   # RF - baseline (negative favours RF)
    return dict(paired_SBMSE_diff=float(mse_rf.mean() - mse_du.mean()),
                paired_diff_CI=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                p_onesided_RF_better=float(np.mean(diff < 0)))


def _within_study():
    p = pd.read_csv(RESULTS / "model_paired.csv")
    uu = p["ref"].unique()
    im = {s: p.index[p["ref"] == s].to_numpy() for s in uu}
    p["ns"] = p.groupby("ref")["ref"].transform("size")
    p["w"] = 1.0 / p["ns"]

    def wcorr(dd):
        pm = dd.groupby("ref")[["g_cof", "g_wear"]].transform("mean")
        dv = dd[["g_cof", "g_wear"]] - pm
        w = dd["w"].values
        a, b = dv["g_cof"].values, dv["g_wear"].values
        ma, mb = np.average(a, weights=w), np.average(b, weights=w)
        cov = np.average((a - ma) * (b - mb), weights=w)
        va = np.average((a - ma) ** 2, weights=w)
        vb = np.average((b - mb) ** 2, weights=w)
        return cov / np.sqrt(va * vb) if va * vb > 0 else np.nan

    pooled = wcorr(p)
    rng = np.random.default_rng(9)
    bs = []
    for _ in range(3000):
        samp = rng.choice(uu, len(uu), replace=True)
        d2 = p.loc[np.concatenate([im[s] for s in samp])]
        rv = wcorr(d2)
        if not np.isnan(rv):
            bs.append(rv)
    zs, n4 = [], 0
    for s, g in p.groupby("ref"):
        if len(g) >= 4 and g["g_cof"].std() > 0 and g["g_wear"].std() > 0:
            rr = min(max(pearsonr(g["g_cof"], g["g_wear"])[0], -0.999), 0.999)
            zs.append((np.arctanh(rr), len(g) - 3))
            n4 += 1
    fisher = float(np.tanh(np.average([z for z, _ in zs], weights=[wt for _, wt in zs]))) if zs else None
    return dict(pooled_studycentred=float(pooled),
                pooled_CI=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                fisher_z_meta_ns4=fisher, n_studies_ns4=n4)


def run(targets=("cof", "wsd")):
    out = {t: _paired(t) for t in targets}
    out["within_study"] = _within_study()
    json.dump(out, open(RESULTS / "paired_bootstrap.json", "w"), indent=1, default=float)
    for t in targets:
        x = out[t]
        print(f"{t}: paired SB-MSE diff {x['paired_SBMSE_diff']:+.4f} "
              f"CI[{x['paired_diff_CI'][0]:+.4f},{x['paired_diff_CI'][1]:+.4f}] "
              f"p(RF better)={x['p_onesided_RF_better']:.2f}")
    ws = out["within_study"]
    print(f"within-study corr: pooled {ws['pooled_studycentred']:.2f} "
          f"CI[{ws['pooled_CI'][0]:.2f},{ws['pooled_CI'][1]:.2f}] | Fisher-z(ns>=4)={ws['fisher_z_meta_ns4']}")
    return out


if __name__ == "__main__":
    run()
