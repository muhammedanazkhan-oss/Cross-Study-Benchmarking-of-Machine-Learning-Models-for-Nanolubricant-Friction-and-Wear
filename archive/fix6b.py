import json, numpy as np, pandas as pd
from scipy.stats import pearsonr
out={}
# 1) finite-sample p for drop-column (M7): p=(b+1)/(B+1), B=1000; re-Holm
dc=json.load(open("dropcol5.json")); B=1000
for tgt in ['cof','wsd']:
    imp=dc[tgt]['importance']; items=[]
    for k,v in imp.items():
        b=round(v['p_onesided']*B); pf=(b+1)/(B+1); v['p_finite']=float(pf); items.append((k,pf))
    items.sort(key=lambda x:x[1]); kf=len(items)
    for rank,(k,pf) in enumerate(items):
        imp[k]['p_holm_finite']=float(min(1.0,pf*(kf-rank))); imp[k]['holm_sig']=bool(imp[k]['p_holm_finite']<0.05)
json.dump(dc,open("dropcol6.json","w"),indent=1)
out['holm_B']=B
print("COF drop-column finite p (B=1000):")
for k,v in sorted(dc['cof']['importance'].items(),key=lambda x:x[1]['p_finite']):
    print(f"  {k:14} p={v['p_finite']:.3f} p_holm={v['p_holm_finite']:.3f} {'*' if v['holm_sig'] else ''}")
# 2) WSD feature dictionary (M4)
from common2 import pre, FEATS, CAT, CONT
mw=pd.read_csv("model_wsd2.csv"); mw['size_missing']=mw['np_size_nm_num'].isna().astype(int)
ct=pre().fit(mw[FEATS]); ohe=ct.named_transformers_['cat']
wsd_dict=[]
for i,c in enumerate(CAT): wsd_dict.append([c,len(ohe.categories_[i]),", ".join(map(str,ohe.categories_[i]))])
out['wsd_encoded_cols']=int(sum(len(x) for x in ohe.categories_)+len(CONT)); out['wsd_dict']=wsd_dict
print("\nWSD encoded columns:",out['wsd_encoded_cols'],"| classes:",list(ohe.categories_[0]),"oils:",list(ohe.categories_[1]))
# 3) conditional (stratum) applicability domain for 2 front regions (C8)
A=pd.read_csv("analysis_all2.csv"); A=A[A['conc_basis']=='wt%']
def trib(x): return 'four-ball' if 'four' in str(x).lower() else 'other'
A['trib2']=A['tribometer'].map(trib)
def stratum_ad(cls,oil,morph,conc,size,load=392.0):
    st=A[(A['npclass']==cls)&(A['base_oil_category']==oil)&(A['morphology']==morph)&(A['trib2']=='four-ball')]
    if len(st)<2: return dict(stratum_rows=len(st),stratum_studies=int(st['ref'].nunique()),note="insufficient stratum")
    cc=pd.to_numeric(st['concentration_wt_pct_num'],errors='coerce'); sz=pd.to_numeric(st['np_size_nm_num'],errors='coerce').fillna(pd.to_numeric(st['np_size_nm_num'],errors='coerce').median()); ld=pd.to_numeric(st['load_N_num'],errors='coerce').fillna(392)
    Rc=(cc.max()-cc.min()) or 1; Rs=(sz.max()-sz.min()) or 1; Rl=(ld.max()-ld.min()) or 1
    C=np.column_stack([cc,sz,ld]); cand=np.array([conc,size,load])
    d=(np.abs(C-cand)/np.array([Rc,Rs,Rl])).mean(1); 
    # LOO 1-NN threshold within stratum
    thr=[]
    for i in range(len(C)):
        dd=(np.abs(C-C[i])/np.array([Rc,Rs,Rl])).mean(1); dd[i]=9; thr.append(dd.min())
    return dict(stratum_rows=int(len(st)),stratum_studies=int(st['ref'].nunique()),nn_distance=float(d.min()),threshold=float(np.percentile(thr,90)),in_domain=bool(d.min()<=np.percentile(thr,90)))
regions={"metal-sulfide/Mineral/2D-layered":stratum_ad('metal-sulfide','Mineral','2D-layered',1.05,180),
         "hybrid/Mineral/2D-layered":stratum_ad('hybrid','Mineral','2D-layered',1.2,60)}
out['conditional_AD']=regions
print("\nConditional (stratum) AD:")
for k,v in regions.items(): print(f"  {k}: {v}")
# 4) Fisher-z details (M6)
p=pd.read_csv("model_paired2.csv"); zs=[]; ws=[]
for s,g in p.groupby('ref'):
    if len(g)>=4 and g['g_cof'].std()>0 and g['g_wear'].std()>0:
        r=min(max(pearsonr(g['g_cof'],g['g_wear'])[0],-0.999),0.999); zs.append(np.arctanh(r)); ws.append(len(g)-3)
zs=np.array(zs); ws=np.array(ws); zbar=np.average(zs,weights=ws); se=np.sqrt(1/ws.sum())
out['fisher']=dict(n_studies=len(zs),weights="n_s-3",r=float(np.tanh(zbar)),CI=[float(np.tanh(zbar-1.96*se)),float(np.tanh(zbar+1.96*se))])
out['within_corr_sample']="50 paired rows from 19 studies"
print("\nFisher-z: r=%.2f CI[%.2f,%.2f] over %d studies (weights n_s-3)"%(out['fisher']['r'],out['fisher']['CI'][0],out['fisher']['CI'][1],out['fisher']['n_studies']))
json.dump(out,open("fix6b.json","w"),indent=1,default=float)
