import sys, json, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from common2 import load, pre, FEATS
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
target=sys.argv[1]; df,X,y,r,groups=load(target); ns=df.groupby('ref')['ref'].transform('size').values; w=1.0/ns
uniq=np.unique(groups); idx=[np.where(groups==s)[0] for s in uniq]; logo=LeaveOneGroupOut()
# ANN 5-seed ENSEMBLE: mean of seed-specific OOF predictions -> one coherent metric (C1)
seedpreds=[]
for seed in range(1,6):
    pred=np.full(len(y),np.nan)
    for tr,te in logo.split(X,y,groups):
        m=MLPRegressor(hidden_layer_sizes=(32,16,8),activation='relu',solver='adam',alpha=1e-3,learning_rate_init=5e-3,max_iter=300,early_stopping=False,tol=0.0,n_iter_no_change=300,random_state=seed)
        p=Pipeline([('pre',pre()),('m',m)]); p.fit(X.iloc[tr],y[tr],m__sample_weight=w[tr]); pred[te]=p.predict(X.iloc[te])
    seedpreds.append(pred)
ens=np.mean(seedpreds,axis=0)
def sb(pred): return np.sqrt(np.mean([mean_squared_error(y[ix],pred[ix]) for ix in idx]))
dref=json.load(open(f"results_{target}5.json"))['per_model']['Dummy (mean)']['sb_rmse']
sbr=float(sb(ens)); skill=float(1-sbr/dref)
# bootstrap CI on ensemble SB-RMSE
pm=np.array([mean_squared_error(y[ix],ens[ix]) for ix in idx]); S=len(uniq); rng=np.random.default_rng(1); SEL=rng.integers(0,S,(3000,S))
ci=[float(np.percentile(np.sqrt(pm[SEL].mean(1)),2.5)),float(np.percentile(np.sqrt(pm[SEL].mean(1)),97.5))]
try: prev=json.load(open("ann_ensemble.json"))
except: prev={}
prev[target]=dict(ann_ensemble_sb_rmse=sbr,ann_ensemble_skill=skill,ann_ensemble_CI=ci,method="mean of out-of-fold predictions across 5 fixed seeds")
json.dump(prev,open("ann_ensemble.json","w"),indent=1)
np.save(f"ann_ens_{target}.npy",ens)
print(f"{target}: ANN ENSEMBLE SB-RMSE {sbr:.3f} skill {skill:+.2f} CI[{ci[0]:.2f},{ci[1]:.2f}] (mean of 5 seeds)")
