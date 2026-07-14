import json, numpy as np
data=[json.load(open("nsga5_0_5.json")),json.load(open("nsga5_5_10.json"))]
meta=data[0]['meta']; seeds=[]
for d in data: seeds+=d['seeds']
cf=json.load(open("conformal5.json")); qc=cf['cof']['q_halfwidth_g']; qw=cf['wsd']['q_halfwidth_g']
def cell(ind): return (ind[0],ind[1],ind[2],round(float(ind[3]),1),int(round(float(ind[4])/10)*10))
rows=[e for sd in seeds for e in sd['front']]
N=len(rows)  # pooled parent-front members (<=500)
F=np.array([[e['gc'],e['gw']] for e in rows])
n=len(F); le=(F[:,None,:]<=F[None,:,:]).all(2); lt=(F[:,None,:]<F[None,:,:]).any(2); db=(le&lt).sum(0)
gi=[j for j in range(n) if db[j]==0]; nd_before=len(gi)
seen=set(); front=[]
for i in sorted(gi,key=lambda i:F[i,0]):
    k=cell(rows[i]['ind'])
    if k in seen: continue
    seen.add(k); gc,gw=F[i,0],F[i,1]
    front.append({"np_class":k[0],"base_oil":k[1],"morphology":k[2],"conc_wt":k[3],"size_nm":k[4],
        "impr_cof_pct":round(float(100*(1-np.exp(gc)))),"impr_wsd_pct":round(float(100*(1-np.exp(gw)))),
        "cof_lb_pct":round(float(100*(1-np.exp(gc+qc)))),"wsd_lb_pct":round(float(100*(1-np.exp(gw+qw))))})
# HV frozen anchors: improvement in [0,100]% -> cost=1-impr/100, ref (1.1,1.1)
def hv_seed(sd):
    pts=np.array([[100*(1-np.exp(e['gc'])),100*(1-np.exp(e['gw']))] for e in sd['front']])
    cost=1-np.clip(pts/100,0,1); m=len(cost); le=(cost[:,None,:]<=cost[None,:,:]).all(2); lt=(cost[:,None,:]<cost[None,:,:]).any(2); nd=[j for j in range(m) if (le&lt).sum(0)[j]==0]
    P=cost[nd]; P=P[np.argsort(P[:,0])]; ref=1.1; hv=0; py=ref
    for x,yv in P: hv+=(ref-x)*(py-yv); py=yv
    return hv
hvs=[hv_seed(sd) for sd in seeds]
sets=[set(cell(e['ind']) for e in sd['front']) for sd in seeds]
jac=[len(sets[a]&sets[b])/len(sets[a]|sets[b]) for a in range(len(sets)) for b in range(a+1,len(sets)) if sets[a]|sets[b]]
cf_class={}; seeds_with={}
for i,sd in enumerate(seeds):
    cls_here=set(e['ind'][0] for e in sd['front'])
    for c in cls_here: seeds_with[c]=seeds_with.get(c,0)+1
    for e in sd['front']: cf_class[e['ind'][0]]=cf_class.get(e['ind'][0],0)+1
out=dict(front=front,n_front=len(front),pooled_members=N,nondominated_before_dedup=nd_before,unique_cells=len(front),
    seeds=len(seeds),pop=meta['pop'],gens=meta['gens'],evals_per_seed=meta['pop']*(meta['gens']+1),total_evals=meta['pop']*(meta['gens']+1)*len(seeds),
    hv_mean=float(np.mean(hvs)),hv_sd=float(np.std(hvs)),hv_anchors="improvement in [0,100]% fixed; cost=1-impr/100; ref (1.1,1.1)",
    jaccard=float(np.mean(jac)),class_freq={k:f"{v}/{N} ({100*v/N:.0f}%)" for k,v in sorted(cf_class.items(),key=lambda x:-x[1])},
    class_seeds={k:f"{v}/{len(seeds)} seeds" for k,v in sorted(seeds_with.items(),key=lambda x:-x[1])},
    n_allowed_pairs=meta['n_allowed'],allowed_support=meta['support'],
    global_band="global study-level cluster CV+ tolerance q applied post-optimisation (constant on g): COF %.2f, WSD %.2f"%(qc,qw),
    q_cof=qc,q_wsd=qw)
json.dump(out,open("nsga5_results.json","w"),indent=1,default=str)
print(f"front(deduped)={len(front)} | pooled N={N} nondom_before_dedup={nd_before} unique_cells={len(front)}")
print(f"HV {np.mean(hvs):.2f}±{np.std(hvs):.2f} | Jaccard {np.mean(jac):.2f} | evals total {out['total_evals']}")
print("class freq:",out['class_freq']," | class in seeds:",out['class_seeds'])
for f in front[:8]: print(f"  {f['np_class']:12} {f['base_oil']:12} {f['morphology']:13} {f['conc_wt']}wt% {f['size_nm']}nm | COF~{f['impr_cof_pct']}%(lb {f['cof_lb_pct']}%) WSD~{f['impr_wsd_pct']}%(lb {f['wsd_lb_pct']}%)")
