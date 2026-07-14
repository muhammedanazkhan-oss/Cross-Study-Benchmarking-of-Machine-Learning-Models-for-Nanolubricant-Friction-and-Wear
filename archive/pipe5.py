import sys, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from common2 import load, pre, FEATS
from sklearn.pipeline import Pipeline
# ONE RF config (300 trees) everywhere; ANN weighted, early_stopping OFF, fixed epochs (pre-specified)
def models():
    return {'Dummy (mean)':DummyRegressor(strategy='mean'),'Ridge':Ridge(alpha=1.0),'ElasticNet':ElasticNet(alpha=0.05,l1_ratio=0.5,max_iter=5000),
     'Random Forest':RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=3,max_features=0.6,random_state=7,n_jobs=-1),
     'XGBoost':XGBRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.85,colsample_bytree=0.8,reg_lambda=2.0,min_child_weight=3,random_state=7,verbosity=0),
     'ANN (32-16-8)':MLPRegressor(hidden_layer_sizes=(32,16,8),activation='relu',solver='adam',alpha=1e-3,learning_rate_init=5e-3,max_iter=300,early_stopping=False,random_state=7)}
for target in [sys.argv[1]] if len(sys.argv)>1 else ['cof','wsd']:
    df,X,y,r,groups=load(target); logo=LeaveOneGroupOut()
    ns=df.groupby('ref')['ref'].transform('size').values; w=1.0/ns
    od=df[['datapoint_id','ref','npclass','base_oil_category']].copy(); od['g_true']=y
    # weighted LOSO OOF (all models incl ANN)
    for m,est in models().items():
        pred=np.full(len(y),np.nan)
        for tr,te in logo.split(X,y,groups):
            p=Pipeline([('pre',pre()),('m',est)]); p.fit(X.iloc[tr],y[tr],m__sample_weight=w[tr]); pred[te]=p.predict(X.iloc[te])
        od['w_'+m.split()[0]]=pred
    # weighted random 5-fold OOF (matched leakage demo) over seeds -> store per-seed SB skill later; keep one representative pred per seed set
    for m,est in models().items():
        for seed in range(1,6):
            pred=np.full(len(y),np.nan)
            kf=KFold(5,shuffle=True,random_state=seed)
            for tr,te in kf.split(X):
                p=Pipeline([('pre',pre()),('m',est)]); p.fit(X.iloc[tr],y[tr],m__sample_weight=w[tr]); pred[te]=p.predict(X.iloc[te])
            od[f'rnd{seed}_'+m.split()[0]]=pred
    od.to_csv(f"oof_{target}5.csv",index=False); print(target,"weighted LOSO+random OOF done, n=",len(y))
