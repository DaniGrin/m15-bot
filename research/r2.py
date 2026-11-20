"""Event study: what happens AFTER a violent 15m move? Reversal or continuation? Net of costs, train vs test."""
import os, numpy as np, pandas as pd
from eng import *
AV=[c for c in COINS if os.path.exists(os.path.join(D,c+".csv"))]
COST_RT=0.0016          # taker both ways + slippage, mid-cap perps
COST_LO=0.0008          # optimistic: mostly-maker execution
HS=(1,4,16,48,96)
def events(c,z,voln=None):
    d=load(c); o,cl=d.open.values,d.close.values; v=d.vol.values
    r=np.r_[0,np.diff(cl)/cl[:-1]]
    sig=pd.Series(r).rolling(96).std().shift(1).values
    zz=r/sig
    ok=np.isfinite(zz)
    vs=pd.Series(v).rolling(96).mean().shift(1).values
    volx=v/vs
    return d,o,cl,zz,volx
rows=[]
for c in AV:
    d,o,cl,zz,volx=events(c,0)
    n=len(cl)
    for zt in (2.5,3,4,5):
        for vcond in (0,2):
            for dirn,name in ((-1,"drop->LONG"),(1,"pump->SHORT")):
                idx=np.where(((zz*dirn*-1)>=zt)&(volx>=vcond) if False else ((-dirn*zz)>=zt)&(volx>=vcond))[0]
                # dirn=-1: z<=-zt (drop) -> go long; dirn=+1: z>=zt (pump) -> go short
                for H in HS:
                    last=-10**9
                    for i in idx:
                        if i+H+1>=n or i<last+H: continue
                        last=i
                        side=-dirn*-1 if False else (1 if dirn==-1 else -1)
                        ent=o[i+1]; ex=cl[i+H]
                        g=side*(ex/ent-1)
                        rows.append((c,d.index[i],zt,vcond,name,H,g))
E=pd.DataFrame(rows,columns=["coin","ts","z","vol","dir","H","g"])
E["tr"]=E.ts<SPLIT
def agg(x):
    r=x.g.values
    return pd.Series(dict(n=len(r),gross=r.mean()*100,net=(r.mean()-COST_RT)*100,netLO=(r.mean()-COST_LO)*100,
                          t=(r.mean()-COST_RT)/(r.std()/np.sqrt(len(r))) if len(r)>2 and r.std()>0 else np.nan,
                          wr=(r-COST_RT>0).mean()*100))
pd.set_option("display.width",220); pd.set_option("display.max_rows",400)
g=E.groupby(["dir","z","vol","H","tr"]).apply(agg).reset_index()
tr=g[g.tr].drop(columns="tr").set_index(["dir","z","vol","H"]); te=g[~g.tr].drop(columns="tr").set_index(["dir","z","vol","H"])
J=tr.join(te,lsuffix="_tr",rsuffix="_te")
J=J[(J.n_tr>=40)&(J.n_te>=25)]
print("variants with enough events:",len(J))
print("\n=== best by TRAIN net (taker costs 0.16% RT), with TEST alongside ===")
print(J.sort_values("net_tr",ascending=False).head(14)[["n_tr","gross_tr","net_tr","t_tr","n_te","gross_te","net_te","t_te","netLO_te"]].round(3).to_string())
print("\npositive net on train: %d/%d   test: %d/%d   both: %d"%((J.net_tr>0).sum(),len(J),(J.net_te>0).sum(),len(J),((J.net_tr>0)&(J.net_te>0)).sum()))
print("\nmean gross move by direction (test), all variants:")
print(J.groupby(level="dir")[["gross_tr","gross_te","net_te"]].mean().round(3))
print("\nmean by horizon:")
print(J.groupby(level="H")[["gross_tr","gross_te","net_te"]].mean().round(3))
