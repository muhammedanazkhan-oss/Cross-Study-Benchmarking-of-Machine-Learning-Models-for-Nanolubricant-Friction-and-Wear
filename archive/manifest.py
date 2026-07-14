import pandas as pd, numpy as np, json
from common2 import pre, FEATS, CAT, CONT
d=pd.read_csv("analysis_all2.csv")
for c in ['cof_reduction_pct_num','wear_reduction_pct_num','baseline_cof_num','nano_cof_num','baseline_wear_num','nano_wear_num']:
    if c in d.columns: d[c]=pd.to_numeric(d[c],errors='coerce')
d['g_cof']=pd.to_numeric(d['g_cof'],errors='coerce'); d['g_wear']=pd.to_numeric(d['g_wear'],errors='coerce')
def path(target):
    g='g_cof' if target=='cof' else 'g_wear'; r='cof_reduction_pct_num' if target=='cof' else 'wear_reduction_pct_num'
    x=d.copy(); steps=[]
    if target=='wsd': x=x[x['wm']=='WSD']; steps.append(("verified WSD-metric rows",len(x),x['ref'].nunique(),""))
    else: x=x[x[r].notna()|x[g].notna()]; steps.append(("verified rows citing a COF outcome",len(x),x['ref'].nunique(),""))
    x2=x[x[g].notna()]; steps.append(("reconstructable log response ratio (Y>0; from raw or reported reduction)",len(x2),x2['ref'].nunique(),f"-{len(x)-len(x2)} no reported/derivable reduction"))
    x3=x2[x2['conc_basis']=='wt%']; steps.append(("wt%-consistent concentration",len(x3),x3['ref'].nunique(),f"-{len(x2)-len(x3)} non-wt% basis"))
    x4=x3[x3['confidence'].isin(['High','Med'])]; steps.append(("High/Medium extraction confidence",len(x4),x4['ref'].nunique(),f"-{len(x3)-len(x4)} low confidence"))
    x5=x4[x4['concentration_wt_pct_num'].notna()]; steps.append(("complete required predictors -> modelling set",len(x5),x5['ref'].nunique(),f"-{len(x4)-len(x5)} missing concentration"))
    return steps
cofp=path('cof'); wsdp=path('wsd')
# LRR route provenance
draw=((d['baseline_cof_num'].notna()&d['nano_cof_num'].notna())).sum() if 'baseline_cof_num' in d.columns else 0
draw_w=((d['baseline_wear_num'].notna()&d['nano_wear_num'].notna())).sum() if 'baseline_wear_num' in d.columns else 0
# feature dictionary: encoded column count
import pandas as pd
mc=pd.read_csv("model_cof2.csv"); mc['size_missing']=mc['np_size_nm_num'].isna().astype(int)
ct=pre().fit(mc[FEATS]); 
ohe=ct.named_transformers_['cat']; enc_cols=sum(len(c) for c in ohe.categories_)
p_encoded=enc_cols+len(CONT)
featdict=[]
for i,c in enumerate(CAT): featdict.append((c,"categorical (one-hot)",", ".join(map(str,ohe.categories_[i])),len(ohe.categories_[i])))
for c in CONT: featdict.append((c,"continuous (imputed+scaled)","—",1))
manifest=dict(verified_rows=len(d),verified_studies=int(d['ref'].nunique()),
  cof_path=cofp,wsd_path=wsdp,
  wsd_metric_rows=int((d['wm']=='WSD').sum()),wsd_with_outcome=int(((d['wm']=='WSD')&d['g_wear'].notna()).sum()),
  p_raw=len(FEATS),p_encoded=int(p_encoded),
  feature_dictionary=featdict,
  trib_categories=sorted(d['trib'].unique().tolist()), oil_categories=sorted(d['base_oil_category'].unique().tolist()),
  morph_categories=sorted(d['morphology'].unique().tolist()), morph_counts=d['morphology'].value_counts().to_dict(),
  lrr_route="All g reconstructed as g=ln(1-R/100) from the reported or computed percentage reduction; raw baseline+treated present for %d COF and %d wear rows (direct-raw sensitivity)."%(int(draw),int(draw_w)),
  wear_metric_counts=d[d['g_wear'].notna()]['wm'].value_counts().to_dict())
json.dump(manifest,open("manifest.json","w"),indent=1,default=str)
print("p_raw=8 groups, p_encoded=",p_encoded,"columns")
print("tribometer categories:",manifest['trib_categories'])
print("oil categories:",manifest['oil_categories'])
print("\nWSD path:")
for s,n,st,ex in wsdp: print(f"  {s}: {n}/{st}  {ex}")
print("\nCOF path:")
for s,n,st,ex in cofp: print(f"  {s}: {n}/{st}  {ex}")
print("\nwsd_metric_rows=%d wsd_with_outcome=%d (reconciles 77 vs 72)"%(manifest['wsd_metric_rows'],manifest['wsd_with_outcome']))
