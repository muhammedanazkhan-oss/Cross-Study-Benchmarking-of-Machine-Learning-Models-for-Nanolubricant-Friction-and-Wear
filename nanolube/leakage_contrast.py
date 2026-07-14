"""Matched leakage contrast.

For each of 15 predeclared partitions the Random Forest is scored under grouped
(studies kept intact) versus row-random 5-fold splits, holding the number of
folds, the training fraction and the study-balanced scoring rule fixed. The
difference isolates the effect of study leakage; it is a split-sensitivity
analysis, not an independent-sample confidence interval.
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from .common import load, preprocessor, study_weights, sb_rmse, RANDOM_STATE, RESULTS


def _rf():
    return RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                 max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1)


def run(targets=("cof", "wsd"), n_partitions=15):
    out = {}
    for target in targets:
        df, X, y, r, groups = load(target)
        w = study_weights(groups)
        uniq = np.unique(groups)
        idx = [np.where(groups == s)[0] for s in uniq]

        def rf_oof(folds):
            pred = np.full(len(y), np.nan)
            for tr, te in folds:
                p = Pipeline([("pre", preprocessor()), ("m", _rf())])
                p.fit(X.iloc[tr], y[tr], m__sample_weight=w[tr])
                pred[te] = p.predict(X.iloc[te])
            return pred

        def dummy_oof(folds):
            pred = np.full(len(y), np.nan)
            for tr, te in folds:
                pred[te] = np.average(y[tr], weights=w[tr])
            return pred

        def skill(folds):
            return 1 - sb_rmse(y, rf_oof(folds), groups) / sb_rmse(y, dummy_oof(folds), groups)

        grp, row, deltas = [], [], []
        for seed in range(n_partitions):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(uniq)
            assign = {s: i % 5 for i, s in enumerate(perm)}
            folds_g = []
            for f in range(5):
                te = np.concatenate([idx[i] for i, s in enumerate(uniq) if assign[s] == f])
                folds_g.append((np.setdiff1d(np.arange(len(y)), te), te))
            order = rng.permutation(len(y))
            folds_r = [(np.setdiff1d(np.arange(len(y)), order[f::5]), order[f::5]) for f in range(5)]
            grp.append(skill(folds_g))
            row.append(skill(folds_r))
            deltas.append(row[-1] - grp[-1])
        out[target] = dict(grouped_skill_mean=float(np.mean(grp)), grouped_skill_sd=float(np.std(grp)),
                           rowrandom_skill_mean=float(np.mean(row)), rowrandom_skill_sd=float(np.std(row)),
                           delta_mean=float(np.mean(deltas)), delta_sd=float(np.std(deltas)),
                           delta_CI=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                           n_partitions=n_partitions)
        print(f"{target}: grouped {np.mean(grp):+.2f}+/-{np.std(grp):.2f} | "
              f"row-random {np.mean(row):+.2f}+/-{np.std(row):.2f} | leakage delta {np.mean(deltas):+.2f}")
    json.dump(out, open(RESULTS / "leakage_contrast.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    run()
