import pandas as pd, numpy as np, json
d=pd.read_csv("master_final.csv")
for c in ['np_size_nm_num','concentration_wt_pct_num','load_N_num','cof_reduction_pct_num','wear_reduction_pct_num']:
    d[c]=pd.to_numeric(d[c],errors='coerce')
d=d[d['source_type']!='manuscript-seed'].copy()   # verified
def trib(x):
    x=str(x).lower()
    if 'four' in x or '4-ball' in x or '4 ball' in x: return 'four-ball'
    if ('pin' in x and ('disk' in x or 'disc' in x)): return 'pin-on-disk'
    if 'recipro' in x or 'srv' in x or 'hfrr' in x or 'ball-on-flat' in x or 'ball-on-plate' in x: return 'reciprocating'
    if 'block' in x and 'ring' in x: return 'block-on-ring'
    if 'ball-on-disk' in x or 'ball on disk' in x or 'ball-on-disc' in x: return 'ball-on-disk'
    if 'ring' in x and 'block' not in x: return 'ring-on-ring'
    return 'other'
d['trib']=d['tribometer'].map(trib)

def npclass(row):
    ss=str(row['np_composition']).lower(); base=str(row.get('np_class','')).lower()
    is_hybrid=any(k in ss for k in ['/','+','@','hybrid','composite','mxene','sandwich']) and not ss.startswith('tio2 (p25')
    if is_hybrid: return 'hybrid'
    if any(k in ss for k in ['halloysite','serpentine','hydrosilicate','silicate','clay']): return 'clay-silicate'
    if any(k in ss for k in ['cellulose','cnc']): return 'organic-bio'
    if 'boron nitride' in ss or 'h-bn' in ss or 'hbn' in ss or ss.strip()=='bn': return 'layered-ceramic'
    if any(k in ss for k in ['mos2','ws2','fes','sulfide']): return 'metal-sulfide'
    if any(k in ss for k in ['graphene','graphit','cnt','nanotube','nanodiamond','diamond','fullerene','carbon','rgo','go ','go(','gnp']) and 'ceo2' not in ss: return 'carbon'
    if any(k in ss for k in ['cuo','al2o3','tio2','zno','zro2','sio2','fe2o3','fe3o4','mn3o4','ceo2','mgo','y2o3','oxide']): return 'metal-oxide'
    if any(k in ss for k in ['cu','ni','ag','bi','fe','co','sn','zn']): return 'metal'
    return base or 'other'
def morph(row):
    ss=str(row['np_composition']).lower()
    if any(k in ss for k in ['graphene','boron nitride','h-bn','hbn','platelet','nanosheet','nanoplate','mos2','ws2','go','rgo','mxene']): return '2D-layered'
    if any(k in ss for k in ['nanotube','cnt','nanorod','nanowire']): return '1D'
    if 'halloysite' in ss: return 'tubular'
    return 'near-spherical'
d['npclass']=d.apply(npclass,axis=1); d['morphology']=d.apply(morph,axis=1)

def wm(x):
    x=str(x).lower()
    if 'wsd' in x or 'scar diam' in x: return 'WSD'
    if 'scar-width' in x or 'scar width' in x: return 'wear-scar-width'
    if 'volume' in x: return 'wear-volume'
    if 'depth' in x: return 'wear-depth'
    if 'rate' in x: return 'wear-rate'
    if 'mass' in x: return 'mass-loss'
    if 'area' in x: return 'wear-area'
    return ''
d['wm']=d['wear_metric'].map(wm)
# LRR validity: Y_nano/Y_base = 1 - r/100 > 0  <=>  r < 100. Check.
def lrr_ok(r): 
    return pd.isna(r) or (r<100)
bad_cof=int((~d['cof_reduction_pct_num'].apply(lrr_ok)).sum()); bad_w=int((~d['wear_reduction_pct_num'].apply(lrr_ok)).sum())
def lrr(r):
    if pd.isna(r): return np.nan
    x=1-r/100.0
    return float(np.log(x)) if x>0 else np.nan
d['g_cof']=d['cof_reduction_pct_num'].map(lrr); d['g_wear']=d['wear_reduction_pct_num'].map(lrr)
d['size_missing']=d['np_size_nm_num'].isna().astype(int)
d['load_missing']=d['load_N_num'].isna().astype(int)
# ---- branching flow, per target (overlapping paths) ----
def branch(target):
    g='g_cof' if target=='cof' else 'g_wear'
    x=d.copy()
    steps=[]
    if target=='wsd':
        x=x[x['wm']=='WSD']; steps.append(("verified WSD-metric rows",len(x),x['ref'].nunique()))
    else:
        x=x[x['cof_reduction_pct_num'].notna()]; steps.append(("verified rows with COF outcome",len(x),x['ref'].nunique()))
    x=x[x['conc_basis']=='wt%']; steps.append(("wt%-consistent concentration",len(x),x['ref'].nunique()))
    x=x[x['confidence'].isin(['High','Med'])]; steps.append(("High/Medium extraction confidence",len(x),x['ref'].nunique()))
    x=x[x[g].notna()]; steps.append(("valid log response ratio (Y>0)",len(x),x['ref'].nunique()))
    x=x[x['concentration_wt_pct_num'].notna()]; steps.append(("complete required predictors",len(x),x['ref'].nunique()))
    return steps,x
cof_steps,cof=branch('cof'); wsd_steps,wsd=branch('wsd')
paired=d[(d['conc_basis']=='wt%')&(d['confidence'].isin(['High','Med']))&(d['g_cof'].notna())&(d['wm']=='WSD')&(d['g_wear'].notna())]
d.to_csv("analysis_all2.csv",index=False); cof.to_csv("model_cof2.csv",index=False); wsd.to_csv("model_wsd2.csv",index=False); paired.to_csv("model_paired2.csv",index=False)
prov=dict(verified_rows=len(d),verified_studies=int(d['ref'].nunique()),
    cof_branch=cof_steps,wsd_branch=wsd_steps,cof_n=len(cof),cof_studies=int(cof['ref'].nunique()),
    wsd_n=len(wsd),wsd_studies=int(wsd['ref'].nunique()),paired_n=len(paired),paired_studies=int(paired['ref'].nunique()),
    lrr_excluded_cof=bad_cof,lrr_excluded_wsd=bad_w,
    wear_metric_counts=d[d['g_wear'].notna()]['wm'].value_counts().to_dict(),
    trib_counts=d['trib'].value_counts().to_dict(), morph_counts=d['morphology'].value_counts().to_dict())
json.dump(prov,open("provenance2.json","w"),indent=1,default=str)
print("verified",len(d),"studies",d['ref'].nunique(),"| LRR-excluded (r>=100): COF",bad_cof,"WSD",bad_w)
print("\nCOF branching:"); 
for s,n,st in cof_steps: print(f"  {s}: {n} rows / {st} studies")
print("WSD branching:")
for s,n,st in wsd_steps: print(f"  {s}: {n} rows / {st} studies")
print(f"\nfinal: COF {len(cof)}/{cof['ref'].nunique()}  WSD {len(wsd)}/{wsd['ref'].nunique()}  paired {len(paired)}/{paired['ref'].nunique()}")
print("trib in COF:",cof['trib'].value_counts().to_dict())
