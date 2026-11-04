import numpy as np, pandas as pd, glob, json
import os; R=os.path.dirname(os.path.abspath(__file__))
th={"tdcsfog":(0.85,0.10),"defog":(0.90,0.10),"Daphnet":(0.85,0.10),"Multimodal":(0.85,0.10),"FoG-STAR":(0.85,0.10)}
def run(name,p,on,off):
    # state-aware: compare with the threshold that can flip the current state
    st=False; d=np.empty(len(p))
    for i,x in enumerate(p):
        d[i]=abs(x-off) if st else abs(x-on)
        if not st and x>on: st=True
        elif st and x<off: st=False
    anyd=np.minimum(abs(p-on),abs(p-off))
    r={"set":name,"n":len(p)}
    for e in (1e-3,1e-4,1e-5,1.6e-6):
        r[f"<{e:g}"]=int((d<e).sum())
    r["min_margin"]=float(d.min()); r["min_any"]=float(anyd.min())
    return r
rows=[]
for k,(a,b) in th.items():
    rows.append(run(k,np.load(f"{R}/results/stream_p_{k}.npy").astype(float),a,b))
lp=[]
for f in sorted(glob.glob(f"{R}/phone_results/live/live_20261029_1[457]*/decisions.csv")):
    lp.append(pd.read_csv(f).p.to_numpy(float))
rows.append(run("live 6 runs",np.concatenate(lp),0.90,0.10))
df=pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv(os.path.join(R,"results","decision_margins.csv"),index=False)
