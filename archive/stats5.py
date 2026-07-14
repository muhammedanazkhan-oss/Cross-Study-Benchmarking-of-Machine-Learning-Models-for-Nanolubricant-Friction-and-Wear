import json, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
mn=['Dummy (mean)','Ridge','ElasticNet','Random Forest','XGBoost','ANN (32-16-8)']; sh={m:m.split()[0] for m in mn}
for target in ['cof','wsd']:
    od=pd.read_csv(f"oof_{target}5.csv"); y=od['g_true'].values; groups=od['ref'].values
    uniq=np.unique(groups); idx=[np.where(groups==s)[0] for s in uniq]; S=len(uniq); rng=np.random.default_rng(1)
    def permse(pred): return np.array([mean_squared_error(y[ix],pred[ix]) for ix in idx])  # per-study MSE
    pm={m:permse(od['w_'+sh[m]].values) for m in mn}
    def sb(mse): return float(np.sqrt(mse.mean()))
    dref=sb(pm['Dummy (mean)'])
    res={}
    for m in mn:
        rnd=np.mean([sb(permse(od[f'rnd{seed}_'+sh[m]].values)) for seed in range(1,6)])
        res[m]=dict(sb_rmse=sb(pm[m]),skill=float(1-sb(pm[m])/dref),rnd_sb_rmse=float(rnd))
    rnd_dref=np.mean([sb(permse(od[f'rnd{seed}_'+sh['Dummy (mean)']].values)) for seed in range(1,6)])
    for m in mn: res[m]['rnd_skill']=float(1-res[m]['rnd_sb_rmse']/rnd_dref)
    B=3000; SEL=rng.integers(0,S,(B,S))
    for m in mn:
        v=np.sqrt(pm[m][SEL].mean(1)); res[m]['sb_rmse_CI']=[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    def pairP(a,b): return float(np.mean(np.sqrt(pm[a][SEL].mean(1))<np.sqrt(pm[b][SEL].mean(1))))
    paired={"P(RF<Dummy)":pairP('Random Forest','Dummy (mean)'),"P(RF<XGB)":pairP('Random Forest','XGBoost'),"P(RF<ANN)":pairP('Random Forest','ANN (32-16-8)')}
    out=dict(per_model=res,paired=paired,n=int(len(y)),studies=int(S),dummy_sb_rmse=dref)
    json.dump(out,open(f"results_{target}5.json","w"),indent=1,default=float)
    print(f"=== {target.upper()} n={len(y)} studies={S} (aligned SB-RMSE; ANN weighted, fixed epochs) ===")
    print(f"{'model':14}{'SB-RMSE':>9}{'skill':>7}{'95% CI':>16}{'rand skill':>11}")
    for m in mn:
        x=res[m]; print(f"{m:14}{x['sb_rmse']:9.3f}{x['skill']:7.2f}  [{x['sb_rmse_CI'][0]:.2f},{x['sb_rmse_CI'][1]:.2f}]{x['rnd_skill']:10.2f}")
    print("paired:",{k:round(v,2) for k,v in paired.items()})
