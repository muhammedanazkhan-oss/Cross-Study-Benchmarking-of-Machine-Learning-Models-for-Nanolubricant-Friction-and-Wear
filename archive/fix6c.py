import sys, json, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from common2 import FEATS, CAT, CONT
from sklearn.metrics import mean_squared_error
target=sys.argv[1]; g='g_cof' if target=='cof' else 'g_wear'
def prep(): return ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),CAT),('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),CONT)])
def rf(): return RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=3,max_features=0.6,random_state=7,n_jobs=-1)
def loso_skill(dfm, studybal_impute=False):
    dfm=dfm.copy(); dfm['size_missing']=dfm['np_size_nm_num'].isna().astype(int)
    X=dfm[FEATS].copy(); y=dfm[g].values; groups=dfm['ref'].values; uniq=np.unique(groups)
    if len(uniq)<6: return None
    idx=[np.where(groups==s)[0] for s in uniq]; ns=dfm.groupby('ref')['ref'].transform('size').values; w=1.0/ns; logo=LeaveOneGroupOut()
    predrf=np.full(len(y),np.nan); predd=np.full(len(y),np.nan)
    for tr,te in logo.split(X,y,groups):
        Xtr=X.iloc[tr].copy(); Xte=X.iloc[te].copy()
        if studybal_impute:  # median of per-study medians on training fold
            sm=dfm.iloc[tr].groupby('ref')['np_size_nm_num'].median(); sbmed=np.nanmedian(sm.values)
            Xtr['np_size_nm_num']=Xtr['np_size_nm_num'].fillna(sbmed); Xte['np_size_nm_num']=Xte['np_size_nm_num'].fillna(sbmed)
        p=Pipeline([('pre',prep()),('m',rf())]); p.fit(Xtr,y[tr],m__sample_weight=w[tr]); predrf[te]=p.predict(Xte); predd[te]=np.average(y[tr],weights=w[tr])
    def sb(pr): return np.sqrt(np.mean([mean_squared_error(y[ix],pr[ix]) for ix in idx]))
    return dict(n=int(len(y)),studies=int(len(uniq)),rf_sb=float(sb(predrf)),skill=float(1-sb(predrf)/sb(predd)))
dm=pd.read_csv(f"model_{target}2.csv")
res=dict(full=loso_skill(dm),high_conf=loso_skill(dm[dm['confidence']=='High']),
         studybal_impute=loso_skill(dm,studybal_impute=True))
bl='baseline_cof' if target=='cof' else 'baseline_wear'; na='nano_cof' if target=='cof' else 'nano_wear'
raw=dm.copy()
for cc in [bl,na]: raw[cc]=pd.to_numeric(raw[cc],errors='coerce')
res['direct_raw']=loso_skill(raw[raw[bl].notna()&raw[na].notna()])
try: prev=json.load(open("sensitivity6.json"))
except: prev={}
prev[target]=res; json.dump(prev,open("sensitivity6.json","w"),indent=1)
print(f"=== {target.upper()} (RF 300 trees) ===")
for k,v in res.items():
    if v: print(f"  {k:18} n={v['n']}/{v['studies']}st RF SB-RMSE {v['rf_sb']:.3f} skill {v['skill']:+.2f}")
