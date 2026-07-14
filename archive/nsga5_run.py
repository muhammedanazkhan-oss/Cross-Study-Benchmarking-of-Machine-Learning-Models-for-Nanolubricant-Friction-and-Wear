import sys, json, warnings, numpy as np, pandas as pd, time; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from common2 import pre, FEATS, load
from sklearn.pipeline import Pipeline
s0,s1=int(sys.argv[1]),int(sys.argv[2]); t0=time.time()
def train(target):
    df,X,y,r,g=load(target); ns=df.groupby('ref')['ref'].transform('size').values; w=1.0/ns
    rf=Pipeline([('pre',pre()),('rf',RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=3,max_features=0.6,random_state=7,n_jobs=-1))])
    rf.fit(X,y,rf__sample_weight=w); return df,rf
dcof,RFc=train('cof'); dwsd,RFw=train('wsd')
# allowed class-oil pairs: intersection with >=2 studies BOTH targets (M8)
def sup(df): return {k:(len(v),v['ref'].nunique()) for k,v in df.groupby(['npclass','base_oil_category'])}
sc=sup(dcof); sw=sup(dwsd)
allowed=[list(k) for k in sc if k in sw and sc[k][1]>=2 and sw[k][1]>=2]
cmorph={c:(dcof[dcof['npclass']==c]['morphology'].unique().tolist() or ['near-spherical']) for c,_ in allowed}
def rng_(dfx,c,col,d0,d1):
    s=pd.to_numeric(dfx[dfx['npclass']==c][col],errors='coerce').dropna(); return (float(np.percentile(s,5)),float(np.percentile(s,95))) if len(s)>2 else (d0,d1)
crange={c:rng_(dcof,c,'concentration_wt_pct_num',0.05,1.5) for c,_ in allowed}; srange={c:rng_(dcof,c,'np_size_nm_num',10,80) for c,_ in allowed}
LOAD=392.0
def gset(dfx):
    cc=pd.to_numeric(dfx['concentration_wt_pct_num'],errors='coerce'); sz=pd.to_numeric(dfx['np_size_nm_num'],errors='coerce'); ld=pd.to_numeric(dfx['load_N_num'],errors='coerce')
    R=np.array([(cc.max()-cc.min()) or 1,(sz.max()-sz.min()) or 1,(ld.max()-ld.min()) or 1]); T=np.array(dfx[['npclass','base_oil_category','morphology']].values.tolist(),dtype=object)
    C=np.column_stack([cc.fillna(cc.median()),sz.fillna(sz.median()),ld.fillna(ld.median())]).astype(float); return R,T,C
Rc,Tc,Cc=gset(dcof); Rw,Tw,Cw=gset(dwsd)
def thr(R,T,C):
    d=[]
    for i in range(len(T)):
        cd=(T!=T[i]).sum(1)/3.0; nd=(np.abs(C-C[i])/R).sum(1)/3.0; dd=(cd+nd)/2; dd[i]=9; d.append(dd.min())
    return float(np.percentile(d,90))
THRc=thr(Rc,Tc,Cc); THRw=thr(Rw,Tw,Cw)
def dfp(pop): return pd.DataFrame([{'npclass':c,'base_oil_category':o,'morphology':mo,'trib':'four-ball','concentration_wt_pct_num':cc,'load_N_num':LOAD,'np_size_nm_num':sz,'size_missing':0} for (c,o,mo,cc,sz) in pop])[FEATS]
def objs(pop):
    Xd=dfp(pop); gc=RFc.predict(Xd); gw=RFw.predict(Xd)
    cat=np.array([[c,o,mo] for (c,o,mo,cc,sz) in pop],dtype=object); cont=np.array([[cc,sz,LOAD] for (c,o,mo,cc,sz) in pop],dtype=float)
    def nn(cat,cont,R,T,C):
        o=np.empty(len(cat))
        for i in range(len(cat)): o[i]=np.min(((T!=cat[i]).sum(1)/3.0+(np.abs(C-cont[i])/R).sum(1)/3.0)/2)
        return o
    dom=(nn(cat,cont,Rc,Tc,Cc)<=THRc)&(nn(cat,cont,Rw,Tw,Cw)<=THRw)
    return np.column_stack([gc,gw]),dom
def fronts(F):
    n=len(F); le=(F[:,None,:]<=F[None,:,:]).all(2); lt=(F[:,None,:]<F[None,:,:]).any(2); dom=le&lt; db=dom.sum(0).astype(int); a=np.zeros(n,bool); fr=[]; cur=[j for j in range(n) if db[j]==0]
    while cur:
        fr.append(cur); a[cur]=True
        for p in cur: db[np.where(dom[p])[0]]-=1
        cur=[j for j in range(n) if not a[j] and db[j]==0]
    return fr
def crowd(F):
    n=len(F); d=np.zeros(n)
    for m in range(2):
        o=np.argsort(F[:,m]); d[o[0]]=d[o[-1]]=1e9; rg=(F[o[-1],m]-F[o[0],m]) or 1; d[o[1:-1]]+=(F[o[2:],m]-F[o[:-2],m])/rg
    return d
def rep(ind,rng):
    if [ind[0],ind[1]] not in allowed: ind[0],ind[1]=allowed[rng.integers(len(allowed))]
    if ind[2] not in cmorph[ind[0]]: ind[2]=cmorph[ind[0]][rng.integers(len(cmorph[ind[0]]))]
    ind[3]=float(min(crange[ind[0]][1],max(crange[ind[0]][0],ind[3]))); ind[4]=float(min(srange[ind[0]][1],max(srange[ind[0]][0],ind[4]))); return ind
def rind(rng): c,o=allowed[rng.integers(len(allowed))]; return rep([c,o,cmorph[c][rng.integers(len(cmorph[c]))],rng.uniform(*crange[c]),rng.uniform(*srange[c])],rng)
def mut(ind,rng):
    ind=list(ind)
    if rng.random()<0.3: c,o=allowed[rng.integers(len(allowed))]; ind[0],ind[1]=c,o
    if rng.random()<0.2: ind[2]=cmorph[ind[0]][rng.integers(len(cmorph[ind[0]]))]
    if rng.random()<0.6: ind[3]+=rng.normal(0,0.12)
    if rng.random()<0.6: ind[4]+=rng.normal(0,9)
    return rep(ind,rng)
def cx(a,b,rng): return rep([a[0],a[1],a[2],(a[3]+b[3])/2,(a[4]+b[4])/2],rng) if rng.random()<0.9 else list(a)
POP,GENS=50,22
def run(seed):
    rng=np.random.default_rng(seed); P=[rind(rng) for _ in range(POP)]
    for _ in range(GENS):
        F,dom=objs(P); Fp=F+(~dom)[:,None]*10; frs=fronts(Fp); idx=[]
        for fr in frs:
            if len(idx)+len(fr)<=POP: idx+=fr
            else: idx+=[fr[i] for i in np.argsort(-crowd(Fp[fr]))[:POP-len(idx)]]; break
        newP=[P[i] for i in idx]; P=newP+[mut(cx(newP[rng.integers(len(newP))],newP[rng.integers(len(newP))],rng),rng) for _ in range(POP)]
    # final = PARENT population only (<=50) -> nondominated subset (C9: N<=500)
    F,dom=objs(P[:POP]); Fp=F+(~dom)[:,None]*10; keep=[i for i in fronts(Fp)[0] if dom[i]]
    return [{"ind":P[i],"gc":float(F[i,0]),"gw":float(F[i,1])} for i in keep]
meta=dict(THRc=THRc,THRw=THRw,allowed=allowed,n_allowed=len(allowed),cmorph=cmorph,
    support={f"{c}/{o}":{"cof":sc[(c,o)],"wsd":sw[(c,o)]} for c,o in allowed},pop=POP,gens=GENS)
rows=[{"seed":s,"front":run(s)} for s in range(s0,s1)]
open(f"nsga5_{s0}_{s1}.json","w").write(json.dumps({"meta":meta,"seeds":rows},default=str))
print(f"seeds {s0}-{s1} {time.time()-t0:.0f}s | allowed(>=2 studies both)={len(allowed)} | parent-front sizes:",[len(r['front']) for r in rows])
