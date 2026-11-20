import os, itertools, numpy as np, pandas as pd
from eng import *
import r3
from r3 import DF15, AV, rmax_incl, rmin_incl

BTCEMA={}
b=DF15["BTC"]
BTCEMA["c"]=b.close; BTCEMA["e"]={s:pd.Series(ema(b.close.values,s),index=b.index) for s in (1600,3200)}

def build(c,dc,es,k,btcf=0,btcs=3200,gate=0,donch_len=None):
    d=DF15[c]; o,h,l,cl=(d[x].values for x in ("open","high","low","close"))
    a=pd.Series(rmax_incl(h,16)-rmin_incl(l,16)).ewm(span=224,adjust=False).mean().values
    hi=rmax(h,dc); lo=rmin(l,dc); e=ema(cl,es)
    L=(cl>hi)&(cl>e); S=(cl<lo)&(cl<e)
    if btcf:
        bt=(BTCEMA["c"]>BTCEMA["e"][btcs]).reindex(d.index).ffill().fillna(False).astype(bool).values
        L&=bt; S&=~bt
    if gate:
        x=er(cl,480); ok=x>pd.Series(x).rolling(8000,min_periods=2000).median().values
        L&=ok; S&=ok
    return d,L,S,k*a

def run(dc=640,es=1600,k=2.0,arm=2.0,give=1.0,btcf=0,gate=0,cost=0.0006,coins=None,maxhold=0,longonly=False):
    ts=[]
    for c in (coins or AV):
        d,L,S,R=build(c,dc,es,k,btcf,3200,gate)
        if longonly: S=np.zeros_like(S)
        t=trades(d,L,S,R,arm=arm,give=give,cost=cost,maxhold=maxhold); t["coin"]=c; ts.append(t)
    return pd.concat(ts,ignore_index=True)

def cap_positions(t,cap):
    if cap is None: return t
    t=t.sort_values("entry"); keep=[]; open_exits=[]
    for i,r in t.iterrows():
        open_exits=[x for x in open_exits if x>r.entry]
        if len(open_exits)<cap: keep.append(i); open_exits.append(r.exit)
    return t.loc[keep]

def report(name,t,risk=0.0075,cap=None):
    t=cap_positions(t,cap)
    tr=summarize(t[t.entry<SPLIT]); te=summarize(t[t.entry>=SPLIT]); al=summarize(t)
    pf=portfolio(t,risk)
    yr=t.assign(y=t.exit.dt.year).groupby("y").net.agg(["sum","count"])
    ys=" ".join("%d:%+.0f"%(y,v*risk*100) for y,v in zip(yr.index,yr["sum"]))   # %return/yr at flat risk
    print("%-34s n=%4d | trainE %+.3f testE %+.3f (t=%+.2f PF %.2f WR %2.0f%%) | CAGR %+5.1f%% DD %6.1f%% | %s"%(
        name,al["n"],tr["expR"],te["expR"],te["t"],te["pf"],te["wr"],pf["cagr"],pf["dd"],ys))
    return t

if __name__=="__main__":
    print("coins:",AV,"\nrisk/trade 0.75%   year columns = sum of net R x 0.75% (non-compounded)\n")
    print("--- exits (dc=40x4h, ema=100x4h, k=2) ---")
    for arm,give in ((1,0.5),(1.5,1),(2,1),(2,1.5),(3,1.5),(3,2),(2,2)):
        report("arm %.1f give %.1f"%(arm,give),run(arm=arm,give=give))
    print("\n--- stop width, exit 2/1 ---")
    for k in (1.5,2.0,2.5,3.0,4.0):
        report("k=%.1f"%k,run(k=k))
    print("\n--- BTC regime filter / ER gate, exit 2/1, k=2 ---")
    report("baseline",run())
    report("BTC>EMA3200 filter",run(btcf=1))
    report("ER gate",run(gate=1))
    report("BTC filter + ER gate",run(btcf=1,gate=1))
    report("LONG ONLY",run(longonly=True))
    report("LONG ONLY + BTC filter",run(longonly=True,btcf=1))
    print("\n--- portfolio caps (baseline) ---")
    for cap in (None,6,4,3,2):
        report("max %s concurrent"%cap,run(),cap=cap)
    print("\n--- cost sensitivity (baseline) ---")
    for cs in (0.0003,0.0006,0.0010,0.0015):
        report("cost %.2f%%/side"%(cs*100),run(cost=cs))
