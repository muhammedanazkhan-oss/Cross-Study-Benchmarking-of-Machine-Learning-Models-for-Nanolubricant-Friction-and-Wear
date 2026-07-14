import json, numpy as np, pandas as pd
alpha=0.1; out={}
for target in ['cof','wsd']:
    od=pd.read_csv(f"oof_{target}5.csv"); y=od['g_true'].values; groups=od['ref'].values
    resid=np.abs(y-od['w_Random'].values); uniq=np.unique(groups); S=len(uniq)
    # study-level conformity score = within-study (1-alpha) quantile of |resid| (max if <3 rows) -> exchangeable unit = study
    scores=[]
    for s in uniq:
        rs=resid[groups==s]; scores.append(np.quantile(rs,0.9) if len(rs)>=3 else rs.max())
    scores=np.array(scores)
    k=min(S,int(np.ceil((S+1)*(1-alpha)))); q=float(np.sort(scores)[k-1])  # cluster CV+ study-level quantile
    # coverages
    row_cov=float(np.mean(resid<=q))
    sb_cov=float(np.mean([np.mean(resid[groups==s]<=q) for s in uniq]))
    out[target]=dict(method="study-level cluster CV+: one conformity score per study (within-study 90th-pct |resid|), quantile over all studies; exchangeable unit = study",
        q_halfwidth_g=q,n_calibration_studies=int(S),alpha=alpha,row_coverage=row_cov,study_balanced_coverage=sb_cov,
        rank_index=k)
    print(f"{target}: study-level q={q:.2f} (n_studies={S}, rank {k}); row coverage {row_cov:.2f}, study-balanced coverage {sb_cov:.2f}")
json.dump(out,open("conformal5.json","w"),indent=1)
