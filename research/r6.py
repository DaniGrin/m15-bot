import numpy as np, pandas as pd
from eng import *
import r4, r5
from r5 import run2, DAILY
from r4 import AV

def mtm_w(t,risk,budget=None,start=10000.0):
    """Daily MTM with per-trade weight column w and an optional cap on TOTAL open risk (in units of equity)."""
    idx=DAILY["BTC"].index
    perday={}  # day -> list of (trade_id, dR)
    open_cnt=pd.Series(0.0,index=idx)     # sum of weights open that day
    paths=[]
    for n,r in enumerate(t.itertuples()):
        ser=DAILY[r.coin]; R=r.epx/r.ratio
        seg=ser.loc[r.entry.normalize():r.exit.normalize()]
        if len(seg)==0: continue
        path=r.side*(seg.values-r.epx)/R; path[-1]=r.net
        d=np.diff(np.r_[0.0,path])
        paths.append((seg.index,d,r.w)); open_cnt.loc[seg.index]+=r.w
    scale=pd.Series(1.0,index=idx)
    if budget is not None:
        tot_risk=open_cnt*risk
        scale=(budget/tot_risk).clip(upper=1.0).fillna(1.0)
    dR=pd.Series(0.0,index=idx)
    for ix,d,w in paths: dR.loc[ix]+=d*w*scale.loc[ix].values
    eq=start*np.cumprod(1+risk*dR.values); s=pd.Series(eq,index=idx)
    dd=(s/s.cummax()-1).min()*100; yrs=(idx[-1]-idx[0]).days/365.25
    ret=s.pct_change().dropna(); sh=ret.mean()/ret.std()*np.sqrt(365)
    wk=s.resample("W").last().pct_change().dropna()
    yr={y:(g.iloc[-1]/g.iloc[0]-1)*100 for y,g in s.groupby(s.index.year)}
    return dict(cagr=((eq[-1]/start)**(1/yrs)-1)*100,dd=dd,sharpe=sh,posw=(wk>0).mean()*100,yr=yr,curve=s)

def legs(ks,**kw):
    parts=[]
    for k in ks:
        t=run2(k=k,**kw); t["w"]=1.0/len(ks); parts.append(t)
    return pd.concat(parts,ignore_index=True)

def line(name,t,risk,budget=None):
    m=mtm_w(t,risk,budget)
    te=t[t.entry>=SPLIT]; tr=t[t.entry<SPLIT]
    wt=lambda x:(x.net*x.w).sum()/x.w.sum()
    print("%-44s risk %.2f%% bud %-5s | trE %+.3f teE %+.3f | CAGR %+5.1f%% DD %6.1f%% Sharpe %.2f ret/DD %.2f | %s"%(
        name,risk*100,("%.1f%%"%(budget*100)) if budget else "-",wt(tr),wt(te),m["cagr"],m["dd"],m["sharpe"],abs(m["cagr"]/m["dd"]),
        " ".join("%d:%+.0f%%"%(y,v) for y,v in m["yr"].items())))
    return m

if __name__=="__main__":
    print("coins:",AV,"\n")
    single=run2(k=2.0); single["w"]=1.0
    print("=== single k=2 (the peak) vs leg-ensembles; same entry signal, stops split across widths ===")
    line("single k=2.0 (peak)",single,0.0050)
    E4=legs((1.5,2.0,2.5,3.0));       line("4 legs k=1.5/2/2.5/3",E4,0.0050)
    E3=legs((2.0,2.5,3.0));           line("3 legs k=2/2.5/3",E3,0.0050)
    E5=legs((1.5,2.0,2.5,3.0,4.0));   line("5 legs k=1.5..4",E5,0.0050)
    print("\n=== ensemble + portfolio risk budget (cap on TOTAL open risk) ===")
    for bud in (None,0.04,0.03,0.02):
        line("4 legs",E4,0.0050,bud)
    print("\n=== risk scaling of the 4-leg ensemble ===")
    for rk in (0.0025,0.0035,0.0050,0.0075): line("4 legs",E4,rk,0.03)
    print("\n=== 4-leg ensemble: test period only, and without 2025 ===")
    te=E4[E4.entry>=SPLIT]; line("test period only (2025-03..)",te,0.0050,0.03)
    line("without 2025",E4[E4.exit.dt.year!=2025],0.0050,0.03)
    line("only 2024",E4[E4.exit.dt.year==2024],0.0050,0.03)
    line("only 2026",E4[E4.exit.dt.year==2026],0.0050,0.03)
    print("\n=== stress: costs on the 4-leg ensemble ===")
    for cs in (0.0010,0.0015):
        t=legs((1.5,2.0,2.5,3.0),cost=cs); line("cost %.2f%%/side"%(cs*100),t,0.0050,0.03)
