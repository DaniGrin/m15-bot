"""4h-style trend, run at 15m resolution vs at native 4h resolution. Same params, same costs."""
import os, itertools, time, numpy as np, pandas as pd
from eng import *
AV=[c for c in COINS if os.path.exists(os.path.join(D,c+".csv"))]
DF15={c:load(c) for c in AV}
def to4h(d):
    return d.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
DF4={c:to4h(d) for c,d in DF15.items()}

def sigs(d,dc,es,k,atrmode,gate,res):
    o,h,l,c=(d[x].values for x in ("open","high","low","close"))
    if res=="15":
        a=pd.Series(rmax_incl(h,16)-rmin_incl(l,16)).ewm(span=224,adjust=False).mean().values   # rolling-4h-range ATR
        erw,medw=480,8000
    else:
        a=atr(h,l,c,14); erw,medw=30,500
    hi=rmax(h,dc); lo=rmin(l,dc); e=ema(c,es)
    L=(c>hi)&(c>e); S=(c<lo)&(c<e)
    if gate:
        x=er(c,erw); ok=x>pd.Series(x).rolling(medw,min_periods=medw//4).median().values
        L&=ok; S&=ok
    return L,S,k*a
def rmax_incl(x,n): return pd.Series(x).rolling(n).max().values
def rmin_incl(x,n): return pd.Series(x).rolling(n).min().values

def run(res,dc,es,k,gate,arm,give,cost=0.0006):
    ts=[]
    for c in AV:
        d=DF15[c] if res=="15" else DF4[c]
        L,S,R=sigs(d,dc,es,k,None,gate,res)
        t=trades(d,L,S,R,arm=arm,give=give,cost=cost); t["coin"]=c; ts.append(t)
    return pd.concat(ts,ignore_index=True)

if __name__=="__main__":
    rows=[]; t0=time.time()
    for res,mult in (("15",16),("4h",1)):
        for dcb,esb,k,gate,(arm,give) in itertools.product((10,20,40),(100,200),(1.0,1.5,2.0,3.0),(0,1),((1.0,0.5),(2.0,1.0))):
            t=run(res,dcb*mult,esb*mult,k,gate,arm,give)
            tr=summarize(t[t.entry<SPLIT]); te=summarize(t[t.entry>=SPLIT])
            if tr is None or te is None: continue
            pf=portfolio(t)
            rows.append(dict(res=res,dc=dcb,ema=esb,k=k,gate=gate,arm=arm,give=give,
                             trn=tr["n"],trE=tr["expR"],ten=te["n"],teE=te["expR"],teT=te["t"],tePF=te["pf"],teWR=te["wr"],
                             ratio=te["ratio"],gross=te["gross"],days=te["bars"]*(0.25 if res=="15" else 4)/24,
                             cagr=pf["cagr"],dd=pf["dd"]))
    R=pd.DataFrame(rows); pd.set_option("display.width",230); pd.set_option("display.max_columns",30)
    print("done %.0fs  %d variants\n"%(time.time()-t0,len(R)))
    for res in ("4h","15"):
        x=R[R.res==res]
        print("=== %s resolution: %d variants | train>0 %d | test>0 %d | both>0 %d | median teE %+.3f | median gross %+.3f | median entry/R %.0f"%(
            res,len(x),(x.trE>0).sum(),(x.teE>0).sum(),((x.trE>0)&(x.teE>0)).sum(),x.teE.median(),x.gross.median(),x.ratio.median()))
    print("\n=== head to head, matched params (test expR): 15m-resolution minus 4h-resolution ===")
    m=R[R.res=="15"].merge(R[R.res=="4h"],on=["dc","ema","k","gate","arm","give"],suffixes=("_15","_4h"))
    print("mean diff expR %+.3f | 15m better in %d of %d | mean trades 15m %.0f vs 4h %.0f | days held %.1f vs %.1f"%(
        (m.teE_15-m.teE_4h).mean(),((m.teE_15>m.teE_4h)).sum(),len(m),m.ten_15.mean(),m.ten_4h.mean(),m.days_15.mean(),m.days_4h.mean()))
    print("\n=== top 10 by TRAIN expR (res=15, n>=100) with TEST ===")
    print(R[(R.res=="15")&(R.trn>=100)].sort_values("trE",ascending=False).head(10)[["dc","ema","k","gate","arm","give","trn","trE","ten","teE","teT","tePF","teWR","days","cagr","dd"]].round(3).to_string(index=False))
    print("\n=== by k (res=15): train/test expR ===")
    print(R[R.res=="15"].groupby("k")[["trE","teE","ratio","days"]].mean().round(3))
    print("\n=== by arm/give (res=15) ===")
    print(R[R.res=="15"].groupby(["arm","give"])[["trE","teE"]].mean().round(3))
    print("\n=== by gate (res=15) ===")
    print(R[R.res=="15"].groupby("gate")[["trE","teE"]].mean().round(3))
    R.to_csv("r3_results.csv",index=False)
