"""Build target-specific modelling frames from the raw dataset.

Derives the log response ratio g = ln(1 - R/100) for each target, maps each
record to a nanoparticle class / morphology / tribometer family, and applies the
target-specific filtering cascade. Writes the modelling frames and a data
manifest into ``results/``. No model is fitted here.
"""
import json
import numpy as np
import pandas as pd
from .common import DATASET, RESULTS

NUMERIC = ["np_size_nm_num", "concentration_wt_pct_num", "load_N_num",
           "cof_reduction_pct_num", "wear_reduction_pct_num"]


def _trib(x):
    x = str(x).lower()
    if "four" in x or "4-ball" in x or "4 ball" in x:
        return "four-ball"
    if "pin" in x and ("disk" in x or "disc" in x):
        return "pin-on-disk"
    if any(k in x for k in ["recipro", "srv", "hfrr", "ball-on-flat", "ball-on-plate"]):
        return "reciprocating"
    if "block" in x and "ring" in x:
        return "block-on-ring"
    if "ball-on-disk" in x or "ball on disk" in x or "ball-on-disc" in x:
        return "ball-on-disk"
    if "ring" in x and "block" not in x:
        return "ring-on-ring"
    return "other"


def _npclass(row):
    ss = str(row["np_composition"]).lower()
    base = str(row.get("np_class", "")).lower()
    is_hybrid = any(k in ss for k in ["/", "+", "@", "hybrid", "composite", "mxene", "sandwich"]) \
        and not ss.startswith("tio2 (p25")
    if is_hybrid:
        return "hybrid"
    if any(k in ss for k in ["halloysite", "serpentine", "hydrosilicate", "silicate", "clay"]):
        return "clay-silicate"
    if any(k in ss for k in ["cellulose", "cnc"]):
        return "organic-bio"
    if "boron nitride" in ss or "h-bn" in ss or "hbn" in ss or ss.strip() == "bn":
        return "layered-ceramic"
    if any(k in ss for k in ["mos2", "ws2", "fes", "sulfide"]):
        return "metal-sulfide"
    if any(k in ss for k in ["graphene", "graphit", "cnt", "nanotube", "nanodiamond", "diamond",
                             "fullerene", "carbon", "rgo", "go ", "go(", "gnp"]) and "ceo2" not in ss:
        return "carbon"
    if any(k in ss for k in ["cuo", "al2o3", "tio2", "zno", "zro2", "sio2", "fe2o3", "fe3o4",
                             "mn3o4", "ceo2", "mgo", "y2o3", "oxide"]):
        return "metal-oxide"
    if any(k in ss for k in ["cu", "ni", "ag", "bi", "fe", "co", "sn", "zn"]):
        return "metal"
    return base or "other"


def _morph(row):
    ss = str(row["np_composition"]).lower()
    if any(k in ss for k in ["graphene", "boron nitride", "h-bn", "hbn", "platelet", "nanosheet",
                             "nanoplate", "mos2", "ws2", "go", "rgo", "mxene"]):
        return "2D-layered"
    if any(k in ss for k in ["nanotube", "cnt", "nanorod", "nanowire"]):
        return "1D"
    if "halloysite" in ss:
        return "tubular"
    return "near-spherical"


def _wm(x):
    x = str(x).lower()
    if "wsd" in x or "scar diam" in x:
        return "WSD"
    if "scar-width" in x or "scar width" in x:
        return "wear-scar-width"
    if "volume" in x:
        return "wear-volume"
    if "depth" in x:
        return "wear-depth"
    if "rate" in x:
        return "wear-rate"
    if "mass" in x:
        return "mass-loss"
    if "area" in x:
        return "wear-area"
    return ""


def _lrr(r):
    """g = ln(1 - R/100); valid only when the treated/baseline ratio is positive (R < 100)."""
    if pd.isna(r):
        return np.nan
    x = 1 - r / 100.0
    return float(np.log(x)) if x > 0 else np.nan


def build():
    d = pd.read_csv(DATASET)
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["source_type"] != "manuscript-seed"].copy()  # verified records only

    d["trib"] = d["tribometer"].map(_trib)
    d["npclass"] = d.apply(_npclass, axis=1)
    d["morphology"] = d.apply(_morph, axis=1)
    d["wm"] = d["wear_metric"].map(_wm)
    d["g_cof"] = d["cof_reduction_pct_num"].map(_lrr)
    d["g_wear"] = d["wear_reduction_pct_num"].map(_lrr)
    d["size_missing"] = d["np_size_nm_num"].isna().astype(int)

    def branch(target):
        g = "g_cof" if target == "cof" else "g_wear"
        x = d.copy()
        steps = []
        if target == "wsd":
            x = x[x["wm"] == "WSD"]
            steps.append(("verified WSD-metric rows", len(x), int(x["ref"].nunique())))
        else:
            x = x[x["cof_reduction_pct_num"].notna()]
            steps.append(("verified rows with COF outcome", len(x), int(x["ref"].nunique())))
        x = x[x["conc_basis"] == "wt%"]
        steps.append(("wt%-consistent concentration", len(x), int(x["ref"].nunique())))
        x = x[x["confidence"].isin(["High", "Med"])]
        steps.append(("High/Medium extraction confidence", len(x), int(x["ref"].nunique())))
        x = x[x[g].notna()]
        steps.append(("valid log response ratio (Y>0)", len(x), int(x["ref"].nunique())))
        x = x[x["concentration_wt_pct_num"].notna()]
        steps.append(("complete required predictors", len(x), int(x["ref"].nunique())))
        return steps, x

    cof_steps, cof = branch("cof")
    wsd_steps, wsd = branch("wsd")
    paired = d[(d["conc_basis"] == "wt%") & (d["confidence"].isin(["High", "Med"]))
               & (d["g_cof"].notna()) & (d["wm"] == "WSD") & (d["g_wear"].notna())]

    cof.to_csv(RESULTS / "model_cof.csv", index=False)
    wsd.to_csv(RESULTS / "model_wsd.csv", index=False)
    paired.to_csv(RESULTS / "model_paired.csv", index=False)

    manifest = dict(
        verified_rows=len(d), verified_studies=int(d["ref"].nunique()),
        cof_branch=cof_steps, wsd_branch=wsd_steps,
        cof_n=len(cof), cof_studies=int(cof["ref"].nunique()),
        wsd_n=len(wsd), wsd_studies=int(wsd["ref"].nunique()),
        paired_n=len(paired), paired_studies=int(paired["ref"].nunique()),
        class_counts=d["npclass"].value_counts().to_dict(),
        tribometer_counts=d["trib"].value_counts().to_dict(),
        morphology_counts=d["morphology"].value_counts().to_dict())
    json.dump(manifest, open(RESULTS / "data_manifest.json", "w"), indent=1, default=str)
    print(f"verified {len(d)} rows / {d['ref'].nunique()} studies -> "
          f"COF {len(cof)}/{cof['ref'].nunique()}, WSD {len(wsd)}/{wsd['ref'].nunique()}, "
          f"paired {len(paired)}/{paired['ref'].nunique()}")
    return manifest


if __name__ == "__main__":
    build()
