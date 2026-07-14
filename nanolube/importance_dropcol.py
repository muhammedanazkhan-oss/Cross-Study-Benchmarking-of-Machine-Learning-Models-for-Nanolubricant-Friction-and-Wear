"""Grouped drop-column feature relevance (descriptive).

Each of seven predictor groups is removed in turn and the full LOSO procedure is
repeated; the change in SB-RMSE is reported with study-bootstrap intervals and a
Holm-adjusted one-sided tail fraction. These quantities are descriptive:
correlated predictors can redistribute importance, so no multiplicity-controlled
significance claim is made.
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from .common import load, RANDOM_STATE, RESULTS

GROUPS = {"NP class": ["npclass"], "Base oil": ["base_oil_category"], "Morphology": ["morphology"],
          "Tribometer": ["trib"], "Concentration": ["concentration_wt_pct_num"],
          "Applied load": ["load_N_num"], "Particle size": ["np_size_nm_num", "size_missing"]}
ALLC = ["npclass", "base_oil_category", "morphology", "trib"]
ALLN = ["concentration_wt_pct_num", "load_N_num", "np_size_nm_num", "size_missing"]


def _pre(cat, cont):
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("n", Pipeline([("i", SimpleImputer(strategy="median")),
                                              ("s", StandardScaler())]), cont)])


def _rf():
    return RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                 max_features=0.6, random_state=RANDOM_STATE, n_jobs=1)


def run(targets=("cof", "wsd")):
    out = {}
    for target in targets:
        df, X, y, r, groups = load(target)
        logo = LeaveOneGroupOut()
        uniq = np.unique(groups)
        idx = [np.where(groups == s)[0] for s in uniq]
        rng = np.random.default_rng(4)
        df["size_missing"] = df["np_size_nm_num"].isna().astype(int)

        def oof(cat, cont):
            return cross_val_predict(Pipeline([("p", _pre(cat, cont)), ("m", _rf())]),
                                     df[cat + cont], y, cv=logo, groups=groups, n_jobs=-1)

        def permse(pred):
            return np.array([mean_squared_error(y[ix], pred[ix]) for ix in idx])

        pmf = permse(oof(ALLC, ALLN))
        base = float(np.sqrt(pmf.mean()))
        S = len(uniq)
        SEL = rng.integers(0, S, (1000, S))
        imp = {}
        for name, cols in GROUPS.items():
            cat = [c for c in ALLC if c not in cols]
            cont = [c for c in ALLN if c not in cols]
            pmd = permse(oof(cat, cont))
            bs = np.sqrt(pmd[SEL].mean(1)) - np.sqrt(pmf[SEL].mean(1))
            imp[name] = dict(delta_sb=float(np.sqrt(pmd.mean()) - base),
                             ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             p_onesided=float(np.mean(bs <= 0)))
        items = sorted(imp.items(), key=lambda x: x[1]["p_onesided"])
        k = len(items)
        for rank, (nm, v) in enumerate(items):
            v["p_holm"] = min(1.0, v["p_onesided"] * (k - rank))
            v["holm_sig"] = bool(v["p_holm"] < 0.05)
        out[target] = {"base_sb_rmse": base, "importance": imp}
        print(f"=== {target.upper()} drop-column delta-SB-RMSE (base={base:.3f}) ===")
        for nm, v in sorted(imp.items(), key=lambda x: -x[1]["delta_sb"]):
            print(f"  {nm:14} delta={v['delta_sb']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p_holm={v['p_holm']:.2f}")
    json.dump(out, open(RESULTS / "importance_dropcol.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    run()
