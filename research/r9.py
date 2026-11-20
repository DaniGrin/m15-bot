"""Final backtest on the exact instruments + exchange the bot trades (Coinbase spot USD), using core.py."""
import os, sys, numpy as np, pandas as pd
import core
from eng import sim, SPLIT, summarize
HERE=os.path.dirname(os.path.abspath(__file__)); CB=os.path.join(HERE,"cb")
COINS="BTC ETH SOL XRP DOGE ADA AVAX LINK LTC DOT UNI AAVE BCH ETC NEAR XLM ATOM FIL SHIB".split()
COST=0.0006; SLIP=0.0002

def load_all(coins):
    raw={c:pd.read_csv(os.path.join(CB,c+".csv"),index_col=0,parse_dates=True) for c in coins if os.path.exists(os.path.join(CB,c+".csv"))}
    end=min(d.index[-1] for d in raw.values())
    return {c:core.clean(d,end) for c,d in raw.items()}

def run_arm(dfs,use_rs,ks,cfg=None,cost=COST,coins=None):
    sg=core.signals(dfs,cfg,use_rs); c0={**core.DEFAULT,**(cfg or {})}
    ts=[]
    for c in (coins or dfs):
        d=dfs[c]; o,h,l,cl=(d[x].values for x in ("open","high","low","close"))
        for k in ks:
            ei,xi,sd,rg,ra=sim(o,h,l,cl,sg[c]["L"],sg[c]["S"],k*sg[c]["A"],c0["arm"],c0["give"],0,SLIP)
            t=pd.DataFrame({"entry":d.index[ei],"exit":d.index[xi],"side":sd,"gross":rg,"ratio":ra,"net":rg-2*cost*ra,
                            "epx":o[ei],"coin":c,"k":k,"w":1.0/len(ks)})
            ts.append(t)
    return pd.concat(ts,ignore_index=True)

def daily_closes(dfs): return {c:d.close.resample("1D").last().ffill() for c,d in dfs.items()}

def mtm(t,dfs,risk,start=10000.0):
    DC=daily_closes(dfs); idx=DC["BTC"].index; dR=pd.Series(0.0,index=idx)
    for r in t.itertuples():
        seg=DC[r.coin].loc[r.entry.normalize():r.exit.normalize()]
        if len(seg)==0: continue
        R=r.epx/r.ratio; path=r.side*(seg.values-r.epx)/R; path[-1]=r.net
        dR.loc[seg.index]+=np.diff(np.r_[0.0,path])*r.w
    eq=start*np.cumprod(1+risk*dR.values); s=pd.Series(eq,index=idx)
    dd=(s/s.cummax()-1).min()*100; yrs=(idx[-1]-idx[0]).days/365.25
    ret=s.pct_change().dropna(); sh=ret.mean()/ret.std()*np.sqrt(365)
    wk=s.resample("W").last().pct_change().dropna()
    yr={y:(g.iloc[-1]/g.iloc[0]-1)*100 for y,g in s.groupby(s.index.year)}
    return dict(cagr=((eq[-1]/start)**(1/yrs)-1)*100,dd=dd,sharpe=sh,posw=(wk>0).mean()*100,yr=yr,curve=s,final=(eq[-1]/start-1)*100)

def wexp(x): return (x.net*x.w).sum()/x.w.sum() if len(x) else float("nan")
def line(name,t,dfs,risk):
    m=mtm(t,dfs,risk); te=t[t.entry>=SPLIT]; tr=t[t.entry<SPLIT]
    tt=te.net.mean()/(te.net.std()/np.sqrt(len(te)))
    print("%-30s risk %.2f%% | trE %+.3f teE %+.3f (t=%+.2f) | CAGR %+6.1f%% DD %6.1f%% Sharpe %.2f ret/DD %.2f | %s"%(
        name,risk*100,wexp(tr),wexp(te),tt,m["cagr"],m["dd"],m["sharpe"],abs(m["cagr"]/m["dd"]),
        " ".join("%d:%+.0f%%"%(y,v) for y,v in m["yr"].items())))
    return m
