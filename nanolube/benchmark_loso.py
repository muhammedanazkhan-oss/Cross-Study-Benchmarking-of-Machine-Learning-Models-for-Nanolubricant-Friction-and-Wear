"""Study-balanced leave-one-study-out (LOSO) benchmark.

For each target and estimator this computes weighted LOSO out-of-fold (OOF)
predictions with fold-internal preprocessing, plus weighted row-random 5-fold
OOF predictions (five seeds) used later for the leakage contrast. It then reports
the study-balanced RMSE (SB-RMSE), the skill over the study-balanced mean
baseline, study-cluster bootstrap intervals and paired win probabilities.
"""
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import mean_squared_error
from .common import load, pipe_of, models, study_weights, RESULTS

SHORT = {m: m.split()[0] for m in
         ["Dummy (mean)", "Ridge", "ElasticNet", "Random Forest", "XGBoost", "ANN (32-16-8)"]}


def _oof(target):
    """Weighted LOSO OOF (all models) + weighted random 5-fold OOF (5 seeds)."""
    df, X, y, r, groups = load(target)
    w = study_weights(groups)
    logo = LeaveOneGroupOut()
    od = df[["datapoint_id", "ref", "npclass", "base_oil_category"]].copy()
    od["g_true"] = y
    mods = models()
    for name, est in mods.items():
        pred = np.full(len(y), np.nan)
        for tr, te in logo.split(X, y, groups):
            p = pipe_of(est)
            p.fit(X.iloc[tr], y[tr], m__sample_weight=w[tr])
            pred[te] = p.predict(X.iloc[te])
        od["loso_" + SHORT[name]] = pred
    for name, est in mods.items():
        for seed in range(1, 6):
            pred = np.full(len(y), np.nan)
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
                p = pipe_of(est)
                p.fit(X.iloc[tr], y[tr], m__sample_weight=w[tr])
                pred[te] = p.predict(X.iloc[te])
            od[f"rnd{seed}_" + SHORT[name]] = pred
    od.to_csv(RESULTS / f"oof_{target}.csv", index=False)
    return od


def _metrics(target, od):
    y = od["g_true"].values
    groups = od["ref"].values
    uniq = np.unique(groups)
    idx = [np.where(groups == s)[0] for s in uniq]
    S = len(uniq)
    rng = np.random.default_rng(1)

    def permse(pred):
        return np.array([mean_squared_error(y[ix], pred[ix]) for ix in idx])

    def sb(mse):
        return float(np.sqrt(mse.mean()))

    names = list(SHORT)
    pm = {m: permse(od["loso_" + SHORT[m]].values) for m in names}
    dref = sb(pm["Dummy (mean)"])
    res = {}
    for m in names:
        rnd = np.mean([sb(permse(od[f"rnd{seed}_" + SHORT[m]].values)) for seed in range(1, 6)])
        res[m] = dict(sb_rmse=sb(pm[m]), skill=float(1 - sb(pm[m]) / dref), rnd_sb_rmse=float(rnd))
    rnd_dref = np.mean([sb(permse(od[f"rnd{seed}_" + SHORT['Dummy (mean)']].values)) for seed in range(1, 6)])
    for m in names:
        res[m]["rnd_skill"] = float(1 - res[m]["rnd_sb_rmse"] / rnd_dref)
    B = 3000
    SEL = rng.integers(0, S, (B, S))
    for m in names:
        v = np.sqrt(pm[m][SEL].mean(1))
        res[m]["sb_rmse_CI"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def pairP(a, b):
        return float(np.mean(np.sqrt(pm[a][SEL].mean(1)) < np.sqrt(pm[b][SEL].mean(1))))

    paired = {"P(RF<Dummy)": pairP("Random Forest", "Dummy (mean)"),
              "P(RF<XGB)": pairP("Random Forest", "XGBoost"),
              "P(RF<ANN)": pairP("Random Forest", "ANN (32-16-8)")}
    out = dict(per_model=res, paired=paired, n=int(len(y)), studies=int(S), dummy_sb_rmse=dref)
    json.dump(out, open(RESULTS / f"benchmark_{target}.json", "w"), indent=1, default=float)
    return out


def run(targets=("cof", "wsd")):
    for target in targets:
        od = _oof(target)
        out = _metrics(target, od)
        print(f"=== {target.upper()}  n={out['n']} studies={out['studies']} (study-balanced LOSO) ===")
        print(f"{'model':14}{'SB-RMSE':>9}{'skill':>7}{'95% CI':>16}{'rand skill':>11}")
        for m, x in out["per_model"].items():
            print(f"{m:14}{x['sb_rmse']:9.3f}{x['skill']:7.2f}  "
                  f"[{x['sb_rmse_CI'][0]:.2f},{x['sb_rmse_CI'][1]:.2f}]{x['rnd_skill']:10.2f}")


if __name__ == "__main__":
    run()
