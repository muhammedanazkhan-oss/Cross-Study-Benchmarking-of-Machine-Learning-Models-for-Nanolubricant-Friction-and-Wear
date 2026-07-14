"""Five-seed ANN ensemble.

The reported neural-network metric is the mean of the out-of-fold predictions
across five fixed seeds (a single coherent estimand), evaluated with the same
study-balanced LOSO protocol as the other models.
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from .common import load, preprocessor, study_weights, sb_rmse, RESULTS


def _ann(seed):
    return MLPRegressor(hidden_layer_sizes=(32, 16, 8), activation="relu", solver="adam",
                        alpha=1e-3, learning_rate_init=5e-3, max_iter=300, early_stopping=False,
                        tol=0.0, n_iter_no_change=300, random_state=seed)


def run(targets=("cof", "wsd")):
    out = {}
    for target in targets:
        df, X, y, r, groups = load(target)
        w = study_weights(groups)
        logo = LeaveOneGroupOut()
        uniq = np.unique(groups)
        idx = [np.where(groups == s)[0] for s in uniq]
        seedpreds = []
        for seed in range(1, 6):
            pred = np.full(len(y), np.nan)
            for tr, te in logo.split(X, y, groups):
                p = Pipeline([("pre", preprocessor()), ("m", _ann(seed))])
                p.fit(X.iloc[tr], y[tr], m__sample_weight=w[tr])
                pred[te] = p.predict(X.iloc[te])
            seedpreds.append(pred)
        ens = np.mean(seedpreds, axis=0)
        dref = json.load(open(RESULTS / f"benchmark_{target}.json"))["per_model"]["Dummy (mean)"]["sb_rmse"]
        sbr = sb_rmse(y, ens, groups)
        pm = np.array([mean_squared_error(y[ix], ens[ix]) for ix in idx])
        S = len(uniq)
        SEL = np.random.default_rng(1).integers(0, S, (3000, S))
        ci = [float(np.percentile(np.sqrt(pm[SEL].mean(1)), 2.5)),
              float(np.percentile(np.sqrt(pm[SEL].mean(1)), 97.5))]
        out[target] = dict(ann_ensemble_sb_rmse=sbr, ann_ensemble_skill=float(1 - sbr / dref),
                           ann_ensemble_CI=ci, method="mean of out-of-fold predictions across 5 fixed seeds")
        print(f"{target}: ANN ensemble SB-RMSE {sbr:.3f} skill {out[target]['ann_ensemble_skill']:+.2f} "
              f"CI[{ci[0]:.2f},{ci[1]:.2f}]")
    json.dump(out, open(RESULTS / "ann_ensemble.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    run()
