import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from common2 import load
G={'NP class':['npclass'],'Base oil':['base_oil_category'],'Morphology':['morphology'],'Tribometer':['trib'],'Concentration':['concentration_wt_pct_num'],'Applied load':['load_N_num'],'Particle size':['np_size_nm_num','size_missing']}
ALLC=['npclass','base_oil_category','morphology','trib']; ALLN=['concentration_wt_pct_num','load_N_num','np_size_nm_num','size_missing']
def pref(cat,cont): return ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),cat),('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),cont)])
def rf(): return RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=3,max_features=0.6,random_state=7,n_jobs=1)
target=sys.argv[1]; df,X,y,r,groups=load(target); logo=LeaveOneGroupOut(); uniq=np.unique(groups); idx=[np.where(groups==s)[0] for s in uniq]; rng=np.random.default_rng(4)
df['size_missing']=df['np_size_nm_num'].isna().astype(int)
def oof(cat,cont): return cross_val_predict(Pipeline([('p',pref(cat,cont)),('m',rf())]),df[cat+cont],y,cv=logo,groups=groups,n_jobs=-1)
def sbperstudy(pred): return np.array([mean_squared_error(y[ix],pred[ix]) for ix in idx])
full=oof(ALLC,ALLN); pmf=sbperstudy(full); base=np.sqrt(pmf.mean())
imp={}; S=len(uniq); SEL=rng.integers(0,S,(1000,S))
for name,cols in G.items():
    cat=[c for c in ALLC if c not in cols]; cont=[c for c in ALLN if c not in cols]
    pmd=sbperstudy(oof(cat,cont)); delta=float(np.sqrt(pmd.mean())-base)
    bs=np.sqrt(pmd[SEL].mean(1))-np.sqrt(pmf[SEL].mean(1))
    # one-sided p: fraction of bootstrap where delta<=0 (feature not helpful)
    pval=float(np.mean(bs<=0))
    imp[name]=dict(delta_sb=delta,ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],p_onesided=pval)
# Holm adjustment
items=sorted(imp.items(),key=lambda x:x[1]['p_onesided']); k=len(items)
for rank,(nm,v) in enumerate(items):
    v['p_holm']=min(1.0,v['p_onesided']*(k-rank)); v['holm_sig']=bool(v['p_holm']<0.05)
try: prev=json.load(open("dropcol5.json"))
except: prev={}
prev[target]={"base_sb_rmse":float(base),"importance":imp}
json.dump(prev,open("dropcol5.json","w"),indent=1)
print(f"=== {target.upper()} drop-column ΔSB-RMSE (Holm-adjusted) base={base:.3f} ===")
for nm,v in sorted(imp.items(),key=lambda x:-x[1]['delta_sb']):
    print(f"  {nm:14} Δ={v['delta_sb']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p_holm={v['p_holm']:.2f} {'*' if v['holm_sig'] else ''}")
