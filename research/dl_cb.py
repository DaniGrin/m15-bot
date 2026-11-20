import urllib.request,json,os,time,datetime,sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"cb")
COINS="BTC ETH SOL XRP DOGE ADA AVAX LINK LTC DOT UNI AAVE BCH ETC NEAR XLM ATOM FIL SHIB".split()
UA={"User-Agent":"Mozilla/5.0"}
iso=lambda t:datetime.datetime.fromtimestamp(t,datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
START=int(datetime.datetime(2023,11,20,tzinfo=datetime.UTC).timestamp())
def call(c,s,e):
    u="https://api.exchange.coinbase.com/products/%s-USD/candles?granularity=900&start=%s&end=%s"%(c,iso(s),iso(e))
    for a in range(6):
        try: return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=30).read())
        except Exception as ex: time.sleep(1+2*a)
    raise RuntimeError("fail %s %s"%(c,iso(s)))
def get(c):
    f=os.path.join(D,c+".csv")
    if os.path.exists(f): return c,"cached"
    rows=[]; e=int(time.time()); step=300*900
    while e>START:
        s=max(START,e-step)
        r=call(c,s,e); rows+=r; e=s; time.sleep(0.12)
    df=pd.DataFrame(rows,columns=["ts","low","high","open","close","vol"]).drop_duplicates("ts").sort_values("ts")
    df["ts"]=pd.to_datetime(df.ts,unit="s"); df=df.set_index("ts")[["open","high","low","close","vol"]]
    df.to_csv(f); return c,"%d bars %s..%s"%(len(df),df.index[0],df.index[-1])
with ThreadPoolExecutor(4) as ex:
    for c,m in ex.map(get,COINS): print(c,m,flush=True)
