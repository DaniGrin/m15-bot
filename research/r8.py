"""Lead-lag: after BTC jumps, do alts keep moving the same way over the next 15m..4h? Net of costs, train/test."""
import numpy as np, pandas as pd
from eng import *
AV=[c for c in COINS if c!="BTC"]
B=load("BTC"); C={c:load(c) for c in AV}
rb=np.log(B.close).diff()
sig=rb.rolling(96).std().shift(1)
z=(np.log(B.close).diff(4)/ (sig*2)) # 1h BTC move in units of 1h sigma
rows=[]
for zt in (2.0,3.0):
    ev=z[(z.abs()>=zt)]
    last=None; ev_idx=[]
    for t in ev.index:                       # de-cluster: one event per 4h
        if last is None or (t-last)>=pd.Timedelta("4h"): ev_idx.append(t); last=t
    for c in AV:
        d=C[c]; cl=d.close; op=d.open
        for t in ev_idx:
            if t not in d.index: continue
            i=d.index.get_loc(t)
            if i+17>=len(d): continue
            # alt's move over the same 1h window as BTC jump: the part it has NOT yet matched
            btc_mv=np.log(B.close).diff(4).loc[t]; alt_mv=np.log(cl).diff(4).iloc[i]
            gap=btc_mv*0.9-alt_mv                              # crude beta~0.9 expectation minus what happened
            for H in (1,4,16):
                g=np.sign(btc_mv)*(np.log(cl.iloc[i+H]/op.iloc[i+1]))
                rows.append((zt,c,t,H,g,np.sign(btc_mv)*gap))
E=pd.DataFrame(rows,columns=["z","coin","t","H","g","lag"]); E["tr"]=E.t<SPLIT
COST=0.0016
print("event: BTC 1h move >= z sigma; trade the ALT in BTC's direction at next open; net of 0.16% RT")
for zt in (2.0,3.0):
    for H in (1,4,16):
        for nm,sel in (("all",lambda x:x),("alt lagging (gap>0.3%)",lambda x:x[x.lag>0.003])):
            x=E[(E.z==zt)&(E.H==H)]; x=sel(x)
            a=x[x.tr].g; b=x[~x.tr].g
            if len(a)<30 or len(b)<30: continue
            print("z>=%.1f H=%2d bars %-24s train n=%4d gross %+.3f%% net %+.3f%% | test n=%4d gross %+.3f%% net %+.3f%% t=%+.2f"%(
                zt,H,nm,len(a),a.mean()*100,(a.mean()-COST)*100,len(b),b.mean()*100,(b.mean()-COST)*100,(b.mean()-COST)/(b.std()/np.sqrt(len(b)))))
