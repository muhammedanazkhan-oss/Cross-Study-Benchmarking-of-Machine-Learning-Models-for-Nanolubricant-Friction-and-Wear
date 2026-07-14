import json, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
out={}
# 1) paired RF-dummy SB-MSE CI + one-sided p  (C4)
for target in ['cof','wsd']:
    od=pd.read_csv(f"oof_{target}5.csv"); y=od['g_true'].values; groups=od['ref'].values; uniq=np.unique(groups); idx=[np.where(groups==s)[0] for s in uniq]; S=len(uniq); rng=np.random.default_rng(1)
    mse_rf=np.array([mean_squared_error(y[ix],od['w_Random'].values[ix]) for ix in idx])
    mse_du=np.array([mean_squared_error(y[ix],od['w_Dummy'].values[ix]) for ix in idx])
    SEL=rng.integers(0,S,(5000,S)); diff=mse_rf[SEL].mean(1)-mse_du[SEL].mean(1)  # RF - dummy (neg = RF better)
    out.setdefault(target,{})['paired_SBMSE_diff']=float(mse_rf.mean()-mse_du.mean())
    out[target]['paired_diff_CI']=[float(np.percentile(diff,2.5)),float(np.percentile(diff,97.5))]
    out[target]['p_onesided_RF_better']=float(np.mean(diff<0))
# 2) valid within-study correlation (C10): pooled study-centred (weight 1/n_s) + cluster bootstrap; Fisher-z over ns>=4
p=pd.read_csv("model_paired2.csv"); uu=p['ref'].unique(); im={s:p.index[p['ref']==s].to_numpy() for s in uu}
p['ns']=p.groupby('ref')['ref'].transform('size'); p['w']=1.0/p['ns']
pm=p.groupby('ref')[['g_cof','g_wear']].transform('mean'); dev=p[['g_cof','g_wear']]-pm
def wcorr(dd,dv):
    w=dd['w'].values; a=dv['g_cof'].values; b=dv['g_wear'].values
    ma=np.average(a,weights=w); mb=np.average(b,weights=w)
    cov=np.average((a-ma)*(b-mb),weights=w); va=np.average((a-ma)**2,weights=w); vb=np.average((b-mb)**2,weights=w)
    return cov/np.sqrt(va*vb) if va*vb>0 else np.nan
pooled=wcorr(p,dev)
rng=np.random.default_rng(9); bs=[]
for _ in range(3000):
    samp=rng.choice(uu,len(uu),replace=True); ii=np.concatenate([im[s] for s in samp]); d2=p.loc[ii]; pm2=d2.groupby('ref')[['g_cof','g_wear']].transform('mean'); dv2=d2[['g_cof','g_wear']]-pm2
    r=wcorr(d2,dv2); 
    if not np.isnan(r): bs.append(r)
# Fisher-z meta over studies with >=4 pairs
zs=[]; nstud4=0
for s,g in p.groupby('ref'):
    if len(g)>=4 and g['g_cof'].std()>0 and g['g_wear'].std()>0:
        r=pearsonr(g['g_cof'],g['g_wear'])[0]; r=min(max(r,-0.999),0.999); zs.append((np.arctanh(r),len(g)-3)); nstud4+=1
if zs:
    zw=np.average([z for z,w in zs],weights=[w for z,w in zs]); fisher=float(np.tanh(zw))
else: fisher=None
out['within_study']=dict(pooled_studycentred=float(pooled),pooled_CI=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
    fisher_z_meta_ns4=fisher,n_studies_ns4=nstud4,note="pooled study-centred (rows weighted 1/n_s) with study-cluster bootstrap; Fisher-z meta restricted to studies with >=4 paired points")
# 3) class freq over nondominated cells + HV per-seed range (M6, M7)
seeds=[]; 
for f in ["nsga5_0_5.json","nsga5_5_10.json"]:
    seeds+=json.load(open(f))['seeds']
def cell(ind): return (ind[0],ind[1],ind[2],round(float(ind[3]),1),int(round(float(ind[4])/10)*10))
rows=[e for sd in seeds for e in sd['front']]; F=np.array([[e['gc'],e['gw']] for e in rows])
n=len(F); le=(F[:,None,:]<=F[None,:,:]).all(2); lt=(F[:,None,:]<F[None,:,:]).any(2); db=(le&lt).sum(0); gi=[j for j in range(n) if db[j]==0]
nd_cells={}; seen=set()
for i in gi:
    k=cell(rows[i]['ind'])
    if k in seen: continue
    seen.add(k); nd_cells[k[0]]=nd_cells.get(k[0],0)+1
ndc=sum(nd_cells.values())
parents={}; 
for e in rows: parents[e['ind'][0]]=parents.get(e['ind'][0],0)+1
nd_all={}
for i in gi: nd_all[rows[i]['ind'][0]]=nd_all.get(rows[i]['ind'][0],0)+1
# HV per seed (fixed anchors)
def hv(sd):
    pts=np.array([[100*(1-np.exp(e['gc'])),100*(1-np.exp(e['gw']))] for e in sd['front']]); cost=1-np.clip(pts/100,0,1)
    m=len(cost); le=(cost[:,None,:]<=cost[None,:,:]).all(2); lt=(cost[:,None,:]<cost[None,:,:]).any(2); nd=[j for j in range(m) if (le&lt).sum(0)[j]==0]
    P=cost[nd]; P=P[np.argsort(P[:,0])]; ref=1.1; h=0; py=ref
    for x,yv in P: h+=(ref-x)*(py-yv); py=yv
    return h
hvs=[hv(sd) for sd in seeds]
out['nsga_frequencies']=dict(unique_cells=ndc,cells_by_class={k:f"{v}/{ndc} ({100*v/ndc:.0f}%)" for k,v in sorted(nd_cells.items(),key=lambda x:-x[1])},
    nondominated_by_class={k:v for k,v in sorted(nd_all.items(),key=lambda x:-x[1])},parents_by_class={k:v for k,v in sorted(parents.items(),key=lambda x:-x[1])},
    hv_per_seed=[round(float(h),4) for h in hvs],hv_min=float(np.min(hvs)),hv_max=float(np.max(hvs)),hv_mean=float(np.mean(hvs)),hv_sd=float(np.std(hvs)))
json.dump(out,open("fix5b.json","w"),indent=1,default=float)
print("PAIRED RF-dummy SB-MSE:")
for t in ['cof','wsd']: print(f"  {t}: diff={out[t]['paired_SBMSE_diff']:+.4f} CI[{out[t]['paired_diff_CI'][0]:+.4f},{out[t]['paired_diff_CI'][1]:+.4f}] p(RF better)={out[t]['p_onesided_RF_better']:.2f}")
print("WITHIN-STUDY corr: pooled study-centred=%.2f CI[%.2f,%.2f] | Fisher-z(ns>=4, %d studies)=%.2f"%(out['within_study']['pooled_studycentred'],out['within_study']['pooled_CI'][0],out['within_study']['pooled_CI'][1],out['within_study']['n_studies_ns4'],out['within_study']['fisher_z_meta_ns4'] or -9))
print("NSGA cells_by_class:",out['nsga_frequencies']['cells_by_class']," | HV %.4f [%.4f,%.4f]"%(out['nsga_frequencies']['hv_mean'],out['nsga_frequencies']['hv_min'],out['nsga_frequencies']['hv_max']))
