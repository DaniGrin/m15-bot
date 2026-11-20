import sys, numpy as np, pandas as pd, time
import core, bot, r9
coins=sys.argv[1].split(",") if len(sys.argv)>1 else ["BTC","ETH","SOL","XRP"]
use_rs=(len(sys.argv)>2 and sys.argv[2]=="rs")
dfs=r9.load_all(coins)
import os; ks=tuple(float(x) for x in os.environ.get("KS","2.0,2.5,3.0").split(","))
t0=time.time()
nb=r9.run_arm(dfs,use_rs,ks)                                   # numba simulator used for every backtest number
print("numba: %d trades (%.1fs)"%(len(nb),time.time()-t0))
sg=core.signals(dfs,None,use_rs)
st=bot.fresh(); rec=[]
cfg={**core.DEFAULT,"ks":ks,"name":"verify"}
grid=dfs["BTC"].index
t0=time.time()
bot.run_bars(st,cfg,dfs,sg,grid,record=rec)
print("stepper: %d trades (%.1fs) | still open at end: %d"%(len(rec),time.time()-t0,len(st["pos"])))
sp=pd.DataFrame(rec); sp["entry"]=pd.to_datetime(sp.entry); sp["exit"]=pd.to_datetime(sp.exit)
nb=nb.rename(columns={"net":"net_nb"})
m=nb.merge(sp,on=["coin","k","entry","exit"],how="outer",indicator=True)
print(m["_merge"].value_counts().to_string())
both=m[m._merge=="both"]
print("max |net difference| on matched trades: %.2e"%np.abs(both.net_nb-both.net).max())
print("sides agree:",bool((both.side_x==both.side_y).all()))
bad=m[m._merge!="both"]
if len(bad): print(bad.sort_values("entry").head(8).to_string())
