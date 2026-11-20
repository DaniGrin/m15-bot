import numpy as np, pandas as pd
from eng import *
import r4
from r4 import DF15, AV, build, BTCEMA

def run2(dc=640,es=1600,k=2.0,arm=2.0,give=1.0,btcs=3200,btcf=1,gate=0,cost=0.0006,coins=None,longonly=False):
    ts=[]
    for c in (coins or AV):
        d,L,S,R=build(c,dc,es,k,btcf,btcs,gate)
        if longonly: S=np.zeros_like(S)
        o=d.open.values
        t=trades(d,L,S,R,arm=arm,give=give,cost=cost); t["coin"]=c
        t["epx"]=o[np.searchsorted(d.index.values,t.entry.values)] if len(t) else []
        ts.append(t)
    return pd.concat(ts,ignore_index=True)

DAILY={c:DF15[c].close.resample("1D").last().ffill() for c in AV}
def mtm(t,risk=0.0075,start=10000.0):
    """Daily mark-to-market portfolio: each open trade contributes its R-multiple change that day."""
    idx=DAILY["BTC"].index; dR=pd.Series(0.0,index=idx)
    for r in t.itertuples():
        ser=DAILY[r.coin]; R=r.epx/r.ratio
        d0=r.entry.normalize(); d1=r.exit.normalize()
        seg=ser.loc[d0:d1]
        if len(seg)==0: continue
        path=r.side*(seg.values-r.epx)/R            # R-multiple at each daily close
        path[-1]=r.net                              # settle on the real net result
        prev=0.0
        for dt,v in zip(seg.index,path):
            dR[dt]+=v-prev; prev=v
    eq=start*np.cumprod(1+risk*dR.values); s=pd.Series(eq,index=idx)
    dd=(s/s.cummax()-1).min()*100
    yrs=(idx[-1]-idx[0]).days/365.25
    ret=s.pct_change().dropna(); sharpe=ret.mean()/ret.std()*np.sqrt(365)
    wk=s.resample("W").last().pct_change().dropna()
    return dict(cagr=((eq[-1]/start)**(1/yrs)-1)*100,dd=dd,sharpe=sharpe,final=(eq[-1]/start-1)*100,
                posw=(wk>0).mean()*100,worstw=wk.min()*100,curve=s)

def line(name,t,risk=0.0075):
    tr=summarize(t[t.entry<SPLIT]); te=summarize(t[t.entry>=SPLIT]); m=mtm(t,risk)
    print("%-36s n=%4d trE %+.3f teE %+.3f (t=%+.2f PF %.2f) | risk %.2f%%: CAGR %+5.1f%% DD %6.1f%% Sharpe %.2f | +weeks %2.0f%% worst wk %5.1f%%"%(
        name,len(t),tr["expR"],te["expR"],te["t"],te["pf"],risk*100,m["cagr"],m["dd"],m["sharpe"],m["posw"],m["worstw"]))
    return m

if __name__=="__main__":
    base=run2()
    print("=== candidate: 4h-style Donchian40/EMA100 at 15m resolution, k=2 ATR4h, trail 2R/1R, BTC regime filter ===\n")
    for rk in (0.0025,0.004,0.005,0.0075):
        line("candidate",base,rk)
    print("\n=== neighbours (does it survive small parameter moves? risk 0.5%) ===")
    for lab,kw in (("dc 20 (320)",dict(dc=320)),("dc 30 (480)",dict(dc=480)),("dc 60 (960)",dict(dc=960)),
                   ("ema 50 (800)",dict(es=800)),("ema 200 (3200)",dict(es=3200)),
                   ("k 1.5",dict(k=1.5)),("k 2.5",dict(k=2.5)),("k 3.0",dict(k=3.0)),
                   ("btc ema 1600",dict(btcs=1600)),("arm 3 / give 1.5",dict(arm=3,give=1.5)),
                   ("arm 1.5 / give 1.0",dict(arm=1.5,give=1.0)),("arm 2 / give 1.5",dict(arm=2,give=1.5))):
        line(lab,run2(**kw),0.005)
    print("\n=== stress ===")
    line("cost 0.10%/side",run2(cost=0.0010),0.005)
    line("cost 0.15%/side",run2(cost=0.0015),0.005)
    print("\n=== drop one year at a time (candidate, risk 0.5%) ===")
    for y in (2024,2025,2026):
        line("without %d"%y,base[base.exit.dt.year!=y],0.005)
    print("\n=== per coin (net expR, n) ===")
    g=base.groupby("coin").net.agg(["count","mean"]); g["t"]=base.groupby("coin").net.apply(lambda x:x.mean()/(x.std()/np.sqrt(len(x))))
    print(g.round(3).T.to_string())
    print("\n=== long vs short (net expR) ===")
    print(base.groupby("side").net.agg(["count","mean"]).round(3).T.to_string())
