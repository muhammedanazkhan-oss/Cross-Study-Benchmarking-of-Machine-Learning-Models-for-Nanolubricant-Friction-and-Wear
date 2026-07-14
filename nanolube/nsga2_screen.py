"""Exploratory NSGA-II screen with a feasibility support filter.

Two Random Forest surrogates (COF and WSD) are trained on the full weighted data
and searched jointly with NSGA-II. Candidates are restricted to class-oil pairs
supported by at least two studies for both targets, and a hierarchical
support/applicability filter (categorical + normalised continuous nearest-
neighbour distance) removes out-of-support regions. Results are pooled across
seeds; the empirical residual envelope is applied post hoc as a lower bound.

The retained region is a bounded experimental design hypothesis for confirmatory
testing, not an optimum, a recommendation, or a validated Pareto front.
"""
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from .common import load, preprocessor, FEATS, RANDOM_STATE, RESULTS

POP, GENS = 50, 22
LOAD = 392.0   # fixed applied load for candidate evaluation (N)


def _train(target):
    df, X, y, r, g = load(target)
    ns = df.groupby("ref")["ref"].transform("size").values
    w = 1.0 / ns
    rf = Pipeline([("pre", preprocessor()),
                   ("rf", RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                                max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1))])
    rf.fit(X, y, rf__sample_weight=w)
    return df, rf


def _gset(dfx):
    cc = pd.to_numeric(dfx["concentration_wt_pct_num"], errors="coerce")
    sz = pd.to_numeric(dfx["np_size_nm_num"], errors="coerce")
    ld = pd.to_numeric(dfx["load_N_num"], errors="coerce")
    R = np.array([(cc.max() - cc.min()) or 1, (sz.max() - sz.min()) or 1, (ld.max() - ld.min()) or 1])
    T = np.array(dfx[["npclass", "base_oil_category", "morphology"]].values.tolist(), dtype=object)
    C = np.column_stack([cc.fillna(cc.median()), sz.fillna(sz.median()), ld.fillna(ld.median())]).astype(float)
    return R, T, C


def _thr(R, T, C):
    d = []
    for i in range(len(T)):
        cd = (T != T[i]).sum(1) / 3.0
        nd = (np.abs(C - C[i]) / R).sum(1) / 3.0
        dd = (cd + nd) / 2
        dd[i] = 9
        d.append(dd.min())
    return float(np.percentile(d, 90))


def _fronts(F):
    n = len(F)
    le = (F[:, None, :] <= F[None, :, :]).all(2)
    lt = (F[:, None, :] < F[None, :, :]).any(2)
    dom = le & lt
    db = dom.sum(0).astype(int)
    a = np.zeros(n, bool)
    fr = []
    cur = [j for j in range(n) if db[j] == 0]
    while cur:
        fr.append(cur)
        a[cur] = True
        for p in cur:
            db[np.where(dom[p])[0]] -= 1
        cur = [j for j in range(n) if not a[j] and db[j] == 0]
    return fr


def _crowd(F):
    n = len(F)
    d = np.zeros(n)
    for m in range(2):
        o = np.argsort(F[:, m])
        d[o[0]] = d[o[-1]] = 1e9
        rg = (F[o[-1], m] - F[o[0], m]) or 1
        d[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / rg
    return d


def _search(dcof, RFc, dwsd, RFw):
    def sup(df):
        return {k: (len(v), v["ref"].nunique()) for k, v in df.groupby(["npclass", "base_oil_category"])}
    sc, sw = sup(dcof), sup(dwsd)
    allowed = [list(k) for k in sc if k in sw and sc[k][1] >= 2 and sw[k][1] >= 2]
    cmorph = {c: (dcof[dcof["npclass"] == c]["morphology"].unique().tolist() or ["near-spherical"])
              for c, _ in allowed}

    def rng_(dfx, c, col, d0, d1):
        s = pd.to_numeric(dfx[dfx["npclass"] == c][col], errors="coerce").dropna()
        return (float(np.percentile(s, 5)), float(np.percentile(s, 95))) if len(s) > 2 else (d0, d1)
    crange = {c: rng_(dcof, c, "concentration_wt_pct_num", 0.05, 1.5) for c, _ in allowed}
    srange = {c: rng_(dcof, c, "np_size_nm_num", 10, 80) for c, _ in allowed}
    Rc, Tc, Cc = _gset(dcof)
    Rw, Tw, Cw = _gset(dwsd)
    THRc, THRw = _thr(Rc, Tc, Cc), _thr(Rw, Tw, Cw)

    def dfp(pop):
        return pd.DataFrame([{"npclass": c, "base_oil_category": o, "morphology": mo, "trib": "four-ball",
                              "concentration_wt_pct_num": cc, "load_N_num": LOAD,
                              "np_size_nm_num": sz, "size_missing": 0}
                             for (c, o, mo, cc, sz) in pop])[FEATS]

    def objs(pop):
        Xd = dfp(pop)
        gc, gw = RFc.predict(Xd), RFw.predict(Xd)
        cat = np.array([[c, o, mo] for (c, o, mo, cc, sz) in pop], dtype=object)
        cont = np.array([[cc, sz, LOAD] for (c, o, mo, cc, sz) in pop], dtype=float)

        def nn(cat, cont, R, T, C):
            out = np.empty(len(cat))
            for i in range(len(cat)):
                out[i] = np.min(((T != cat[i]).sum(1) / 3.0 + (np.abs(C - cont[i]) / R).sum(1) / 3.0) / 2)
            return out
        dom = (nn(cat, cont, Rc, Tc, Cc) <= THRc) & (nn(cat, cont, Rw, Tw, Cw) <= THRw)
        return np.column_stack([gc, gw]), dom

    def rep(ind, rng):
        if [ind[0], ind[1]] not in allowed:
            ind[0], ind[1] = allowed[rng.integers(len(allowed))]
        if ind[2] not in cmorph[ind[0]]:
            ind[2] = cmorph[ind[0]][rng.integers(len(cmorph[ind[0]]))]
        ind[3] = float(min(crange[ind[0]][1], max(crange[ind[0]][0], ind[3])))
        ind[4] = float(min(srange[ind[0]][1], max(srange[ind[0]][0], ind[4])))
        return ind

    def rind(rng):
        c, o = allowed[rng.integers(len(allowed))]
        return rep([c, o, cmorph[c][rng.integers(len(cmorph[c]))], rng.uniform(*crange[c]),
                    rng.uniform(*srange[c])], rng)

    def mut(ind, rng):
        ind = list(ind)
        if rng.random() < 0.3:
            ind[0], ind[1] = allowed[rng.integers(len(allowed))]
        if rng.random() < 0.2:
            ind[2] = cmorph[ind[0]][rng.integers(len(cmorph[ind[0]]))]
        if rng.random() < 0.6:
            ind[3] += rng.normal(0, 0.12)
        if rng.random() < 0.6:
            ind[4] += rng.normal(0, 9)
        return rep(ind, rng)

    def cx(a, b, rng):
        return rep([a[0], a[1], a[2], (a[3] + b[3]) / 2, (a[4] + b[4]) / 2], rng) if rng.random() < 0.9 else list(a)

    def run_seed(seed):
        rng = np.random.default_rng(seed)
        P = [rind(rng) for _ in range(POP)]
        for _ in range(GENS):
            F, dom = objs(P)
            Fp = F + (~dom)[:, None] * 10
            idx = []
            for fr in _fronts(Fp):
                if len(idx) + len(fr) <= POP:
                    idx += fr
                else:
                    idx += [fr[i] for i in np.argsort(-_crowd(Fp[fr]))[:POP - len(idx)]]
                    break
            newP = [P[i] for i in idx]
            P = newP + [mut(cx(newP[rng.integers(len(newP))], newP[rng.integers(len(newP))], rng), rng)
                        for _ in range(POP)]
        F, dom = objs(P[:POP])
        Fp = F + (~dom)[:, None] * 10
        keep = [i for i in _fronts(Fp)[0] if dom[i]]
        return [{"ind": P[i], "gc": float(F[i, 0]), "gw": float(F[i, 1])} for i in keep]

    meta = dict(allowed=allowed, n_allowed=len(allowed), pop=POP, gens=GENS, THRc=THRc, THRw=THRw,
                support={f"{c}/{o}": {"cof": sc[(c, o)], "wsd": sw[(c, o)]} for c, o in allowed})
    return meta, run_seed


def _cell(ind):
    return (ind[0], ind[1], ind[2], round(float(ind[3]), 1), int(round(float(ind[4]) / 10) * 10))


def run(seeds=range(10)):
    dcof, RFc = _train("cof")
    dwsd, RFw = _train("wsd")
    meta, run_seed = _search(dcof, RFc, dwsd, RFw)
    seed_fronts = [{"seed": s, "front": run_seed(s)} for s in seeds]

    env = json.load(open(RESULTS / "residual_envelope.json"))
    qc, qw = env["cof"]["q_halfwidth_g"], env["wsd"]["q_halfwidth_g"]
    rows = [e for sd in seed_fronts for e in sd["front"]]
    N = len(rows)
    F = np.array([[e["gc"], e["gw"]] for e in rows])
    le = (F[:, None, :] <= F[None, :, :]).all(2)
    lt = (F[:, None, :] < F[None, :, :]).any(2)
    db = (le & lt).sum(0)
    gi = [j for j in range(len(F)) if db[j] == 0]
    seen, front = set(), []
    for i in sorted(gi, key=lambda i: F[i, 0]):
        k = _cell(rows[i]["ind"])
        if k in seen:
            continue
        seen.add(k)
        gc, gw = F[i, 0], F[i, 1]
        front.append({"np_class": k[0], "base_oil": k[1], "morphology": k[2], "conc_wt": k[3], "size_nm": k[4],
                      "impr_cof_pct": round(float(100 * (1 - np.exp(gc)))),
                      "impr_wsd_pct": round(float(100 * (1 - np.exp(gw)))),
                      "cof_lb_pct": round(float(100 * (1 - np.exp(gc + qc)))),
                      "wsd_lb_pct": round(float(100 * (1 - np.exp(gw + qw))))})
    sets = [set(_cell(e["ind"]) for e in sd["front"]) for sd in seed_fronts]
    jac = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
           for a in range(len(sets)) for b in range(a + 1, len(sets)) if sets[a] | sets[b]]
    cf_class = {}
    for e in rows:
        cf_class[e["ind"][0]] = cf_class.get(e["ind"][0], 0) + 1
    out = dict(front=front, n_front=len(front), pooled_members=N, seeds=len(seed_fronts),
               pop=meta["pop"], gens=meta["gens"],
               jaccard=float(np.mean(jac)) if jac else None,
               class_freq={k: f"{v}/{N} ({100 * v / N:.0f}%)" for k, v in sorted(cf_class.items(), key=lambda x: -x[1])},
               n_allowed_pairs=meta["n_allowed"], allowed_support=meta["support"],
               residual_band=f"post-optimisation envelope q on g: COF {qc:.2f}, WSD {qw:.2f}",
               q_cof=qc, q_wsd=qw)
    json.dump(out, open(RESULTS / "nsga2_screen.json", "w"), indent=1, default=str)
    print(f"front(deduped)={len(front)} pooled={N} | allowed pairs={meta['n_allowed']} | "
          f"Jaccard={out['jaccard']} | class freq={out['class_freq']}")
    for f in front[:8]:
        print(f"  {f['np_class']:12} {f['base_oil']:12} {f['morphology']:11} {f['conc_wt']}wt% {f['size_nm']}nm "
              f"| COF~{f['impr_cof_pct']}% (lb {f['cof_lb_pct']}%)  WSD~{f['impr_wsd_pct']}% (lb {f['wsd_lb_pct']}%)")
    return out


if __name__ == "__main__":
    run()
