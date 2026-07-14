"""Shared configuration, data loading, preprocessing and model definitions.

All analysis modules import from here so that the feature set, the fold-internal
preprocessing and the estimator hyper-parameters are defined in exactly one place.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DATASET = DATA / "nanolubricant_dataset.csv"

# eight raw predictor groups (one scalar size axis, interpreted as morphology-dependent)
CAT = ["npclass", "base_oil_category", "morphology", "trib"]
CONT = ["concentration_wt_pct_num", "load_N_num", "np_size_nm_num", "size_missing"]
FEATS = CAT + CONT
RANDOM_STATE = 7


def load(target):
    """Load a target-specific modelling frame produced by ``data_prep``.

    Returns ``df, X, y, r, groups`` where ``y`` is the log response ratio,
    ``r`` the percentage reduction and ``groups`` the study identifier.
    """
    if target not in ("cof", "wsd"):
        raise ValueError("target must be 'cof' or 'wsd'")
    df = pd.read_csv(RESULTS / f"model_{target}.csv")
    df["size_missing"] = df["np_size_nm_num"].isna().astype(int)
    g = "g_cof" if target == "cof" else "g_wear"
    r = "cof_reduction_pct_num" if target == "cof" else "wear_reduction_pct_num"
    return df, df[FEATS].copy(), df[g].values, df[r].values, df["ref"].values


def preprocessor():
    """Fold-internal preprocessing: one-hot categoricals + median-imputed, scaled continuous."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), CONT),
    ])


def models():
    """The six benchmarked estimators.

    A single ANN seed is used here; the reported ANN metric is the five-seed
    ensemble computed in ``ann_ensemble`` (mean of out-of-fold predictions).
    """
    return {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=3,
            random_state=RANDOM_STATE, verbosity=0),
        "ANN (32-16-8)": MLPRegressor(
            hidden_layer_sizes=(32, 16, 8), activation="relu", solver="adam",
            alpha=1e-3, learning_rate_init=5e-3, max_iter=300,
            early_stopping=False, random_state=RANDOM_STATE),
    }


def pipe_of(est):
    return Pipeline([("pre", preprocessor()), ("m", est)])


def study_weights(groups):
    """Study-balanced weights w = 1/n_s so every study carries equal total fitting weight."""
    s = pd.Series(groups)
    return (1.0 / s.groupby(s).transform("size")).to_numpy()


def sb_rmse(y, pred, groups):
    """Study-balanced RMSE: sqrt(mean over studies of within-study MSE)."""
    from sklearn.metrics import mean_squared_error
    uniq = np.unique(groups)
    per = [mean_squared_error(y[groups == s], pred[groups == s]) for s in uniq]
    return float(np.sqrt(np.mean(per)))
