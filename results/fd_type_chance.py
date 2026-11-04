import sys, numpy as np, pandas as pd, json
import os; R=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,R)
import fd_config as C
from fd_metrics import Evaluator
w=pd.read_csv(R+"/results/external_FoG-STAR_windows.csv")
ev=Evaluator(w.drop(columns=["y4","p","cue_on"]), w["y4"].to_numpy())
on=w["cue_on"].to_numpy().astype(bool)
print("check overall", {k:round(v,2) for k,v in ev.evaluate(on).items() if k in("coverage_%","timely_%")})
det_s,_=ev.surrogate(on,samples=True)
# per-severity: replicate surrogate by computing per-episode coverage under the same shifts
rng=np.random.default_rng(C.SEED); n_shifts=C.N_SHIFTS
sev=w["severity"].to_numpy()
eps=[]
for q,s,e in ev.episodes:
    idx=ev.seqs[q]; v=sev[idx[s:e]]; v=v[v>0]; k=int(np.bincount(v).argmax()) if len(v) else 0
    eps.append((q,s,e,k))
hits={}; obs={}
for q,idx in enumerate(ev.seqs):
    n=len(idx)
    my=[x for x in eps if x[0]==q]
    if n<2: continue
    k=rng.integers(1,n,size=n_shifts)
    if not my: continue
    o=on[idx].astype(np.int64); c=np.concatenate([[0],np.cumsum(np.concatenate([o,o]))])
    for (_,s,e,sv) in my:
        a=(s-k)%n; cov=(c[a+(e-s)]-c[a])>0
        hits.setdefault(sv,[]).append(cov)
        obs.setdefault(sv,[]).append(on[idx[s:e]].any())
out=[]
for sv in sorted(hits):
    H=np.array(hits[sv]); ob=np.mean(obs[sv])*100; ch=H.mean(0)*100
    p=(1+np.sum(ch>=ob))/(1+n_shifts)
    dur=np.median([e-s for (q,s,e,k) in eps if k==sv])*C.WIN/C.FS
    out.append(dict(severity={1:"shuffling",2:"trembling",3:"akinesia"}.get(sv,sv),episodes=len(H.T[0]) if False else H.shape[0],coverage=round(ob,1),chance=round(ch.mean(),1),gain=round(ob-ch.mean(),1),p=round(p,3),median_dur_s=dur))
df=pd.DataFrame(out); print(df.to_string(index=False)); df.to_csv(os.path.join(R,"results","external_FoG-STAR_by_type_chance.csv"),index=False)
