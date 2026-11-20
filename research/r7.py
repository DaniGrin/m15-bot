import numpy as np, pandas as pd
from eng import *
import r4, r5, r6
from r4 import DF15, AV, BTCEMA
from r3 import rmax_incl, rmin_incl
from r6 import mtm_w

CL=pd.DataFrame({c:DF15[c].close for c in AV}).sort_index().ffill()
def rs_table(n):
    r=np.log(CL/CL.shift(n)); return r.sub(r.median(axis=1),axis=0)       # cross-sectionally demeaned (vs median)
RS={n:rs_table(n) for n in (1920,2880,5760)}

def build_rs(c,dc,es,k,btcs,rsn,rsmode):
    d=DF15[c]; o,h,l,cl=(d[x].values for x in ("open","high","low","close"))
    a=pd.Series(rmax_incl(h,16)-rmin_incl(l,16)).ewm(span=224,adjust=False).mean().values
    hi=rmax(h,dc); lo=rmin(l,dc); e=ema(cl,es)
    L=(cl>hi)&(cl>e); S=(cl<lo)&(cl<e)
    bt=(BTCEMA["c"]>BTCEMA["e"][btcs]).reindex(d.index).ffill().fillna(False).astype(bool).values
    L&=bt; S&=~bt
    if rsn:
        rs=RS[rsn][c].reindex(d.index).ffill().values
        if rsmode=="sign": L&=(rs>0); S&=(rs<0)
        elif rsmode=="long_only_strong": L&=(rs>0); S&=False
    return d,L,S,k*a

def run_rs(k,dc=640,es=1600,btcs=3200,rsn=0,rsmode="sign",arm=2.0,give=1.0,cost=0.0006):
    ts=[]
    for c in AV:
        d,L,S,R=build_rs(c,dc,es,k,btcs,rsn,rsmode)
        t=trades(d,L,S,R,arm=arm,give=give,cost=cost); t["coin"]=c
        t["epx"]=d.open.values[np.searchsorted(d.index.values,t.entry.values)] if len(t) else []
        ts.append(t)
    return pd.concat(ts,ignore_index=True)

def legs(ks,**kw):
    parts=[]
    for k in ks:
        t=run_rs(k,**kw); t["w"]=1.0/len(ks); parts.append(t)
    return pd.concat(parts,ignore_index=True)

def line(name,t,risk=0.005):
    m=mtm_w(t,risk,None); wt=lambda x:(x.net*x.w).sum()/x.w.sum() if len(x) else np.nan
    te=t[t.entry>=SPLIT]; tr=t[t.entry<SPLIT]
    print("%-38s n=%4d | trE %+.3f teE %+.3f | CAGR %+5.1f%% DD %6.1f%% Sharpe %.2f ret/DD %.2f | %s"%(
        name,len(t)//len(set(t.w)) if len(t) else 0,wt(tr),wt(te),m["cagr"],m["dd"],m["sharpe"],abs(m["cagr"]/m["dd"]),
        " ".join("%d:%+.0f%%"%(y,v) for y,v in m["yr"].items())))
    return m

if __name__=="__main__":
    ks=(2.0,2.5,3.0)
    print("3-leg ensemble (k=2/2.5/3), BTC regime filter on in all rows, risk 0.5%/signal\n")
    line("baseline (no RS filter)",legs(ks))
    for n in (1920,2880,5760):
        line("RS sign filter, %dd"%(n//96),legs(ks,rsn=n))
    line("RS long-only-strong 30d",legs(ks,rsn=2880,rsmode="long_only_strong"))
    print("\nneighbour check on the entry channel with the best RS lookback:")
    for dc in (480,640,960):
        line("RS30d, Donchian %d x4h"%(dc//16),legs(ks,rsn=2880,dc=dc))
