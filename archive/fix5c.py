import sys, json, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from common2 import load, pre, FEATS
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
def rf(): return RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=3,max_features=0.6,random_state=7,n_jobs=-1)
target=sys.argv[1]; df,X,y,r,groups=load(target); ns=df.groupby('ref')['ref'].transform('size').values; w=1.0/ns
uniq=np.unique(groups); idx=[np.where(groups==s)[0] for s in uniq]
def sbskill(pred):
    mse=np.mean([mean_squared_error(y[ix],pred[ix]) for ix in idx]); mse_d=np.mean([mean_squared_error(y[ix],np.repeat(np.average(y[np.setdiff1d(np.arange(len(y)),ix)]),len(ix))[:len(ix)] if False else np.full(len(ix),y.mean())) for ix in idx])
    return 1-np.sqrt(mse)/np.sqrt(mse_d)
def dummy_pred_cv(folds):  # mean predictor OOF
    pred=np.full(len(y),np.nan)
    for tr,te in folds: pred[te]=np.average(y[tr],weights=w[tr])
    return pred
def rf_cv(folds):
    pred=np.full(len(y),np.nan)
    for tr,te in folds:
        p=Pipeline([('pre',pre()),('m',rf())]); p.fit(X.iloc[tr],y[tr],m__sample_weight=w[tr]); pred[te]=p.predict(X.iloc[te])
    return pred
def sb(pred): return np.sqrt(np.mean([mean_squared_error(y[ix],pred[ix]) for ix in idx]))
deltas=[]; grp_sk=[]; row_sk=[]
for seed in range(15):
    rng=np.random.default_rng(seed)
    # grouped 5-fold: assign studies to 5 folds
    perm=rng.permutation(uniq); folds_g=[]
    fold_assign={s:i%5 for i,s in enumerate(perm)}
    for f in range(5):
        te=np.concatenate([idx[i] for i,s in enumerate(uniq) if fold_assign[s]==f]); tr=np.setdiff1d(np.arange(len(y)),te); folds_g.append((tr,te))
    # row-random 5-fold
    order=rng.permutation(len(y)); folds_r=[(np.setdiff1d(np.arange(len(y)),order[f::5]),order[f::5]) for f in range(5)]
    for folds,store in [(folds_g,grp_sk),(folds_r,row_sk)]:
        sk=1-sb(rf_cv(folds))/sb(dummy_pred_cv(folds)); store.append(sk)
    deltas.append(row_sk[-1]-grp_sk[-1])
res=dict(target=target,grouped_skill_mean=float(np.mean(grp_sk)),grouped_skill_sd=float(np.std(grp_sk)),
    rowrandom_skill_mean=float(np.mean(row_sk)),rowrandom_skill_sd=float(np.std(row_sk)),
    delta_mean=float(np.mean(deltas)),delta_sd=float(np.std(deltas)),delta_CI=[float(np.percentile(deltas,2.5)),float(np.percentile(deltas,97.5))],n_partitions=15)
try: prev=json.load(open("matched_leakage.json"))
except: prev={}
prev[target]=res; json.dump(prev,open("matched_leakage.json","w"),indent=1)
print(f"{target}: grouped 5-fold skill {np.mean(grp_sk):+.2f}±{np.std(grp_sk):.2f} | row-random {np.mean(row_sk):+.2f}±{np.std(row_sk):.2f} | Δ(leakage) {np.mean(deltas):+.2f} CI[{np.percentile(deltas,2.5):+.2f},{np.percentile(deltas,97.5):+.2f}]")
