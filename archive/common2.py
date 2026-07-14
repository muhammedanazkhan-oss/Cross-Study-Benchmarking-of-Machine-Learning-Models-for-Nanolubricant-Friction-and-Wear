import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
CAT=['npclass','base_oil_category','morphology','trib']; CONT=['concentration_wt_pct_num','load_N_num','np_size_nm_num','size_missing']
FEATS=CAT+CONT
def load(target):
    df=pd.read_csv(f"model_{target}2.csv"); df['size_missing']=df['np_size_nm_num'].isna().astype(int)
    g='g_cof' if target=='cof' else 'g_wear'; r='cof_reduction_pct_num' if target=='cof' else 'wear_reduction_pct_num'
    return df, df[FEATS].copy(), df[g].values, df[r].values, df['ref'].values
def pre(): return ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),CAT),('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),CONT)])
def models():
    return {'Dummy (mean)':DummyRegressor(strategy='mean'),'Ridge':Ridge(alpha=1.0),'ElasticNet':ElasticNet(alpha=0.05,l1_ratio=0.5,max_iter=5000),
     'Random Forest':RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=3,max_features=0.6,random_state=7,n_jobs=1),
     'XGBoost':XGBRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.85,colsample_bytree=0.8,reg_lambda=2.0,min_child_weight=3,random_state=7,verbosity=0),
     'ANN (32-16-8)':MLPRegressor(hidden_layer_sizes=(32,16,8),activation='relu',alpha=1e-3,learning_rate_init=5e-3,max_iter=500,early_stopping=True,n_iter_no_change=15,validation_fraction=0.15,random_state=7)}
def pipe_of(est): return Pipeline([('pre',pre()),('m',est)])
