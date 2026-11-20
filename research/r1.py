"""Round 1: does ANY plain 15m trend entry survive realistic costs? Walk-forward: pick on train, report test."""
import itertools, sys, os, time
import numpy as np, pandas as pd
from eng import *

COINS_AVAIL = [c for c in COINS if os.path.exists(os.path.join(D, c + ".csv"))]
DF = {c: load(c) for c in COINS_AVAIL}
IND = {}
for c, d in DF.items():
    h, l, cl = d.high.values, d.low.values, d.close.values
    e = er(cl, 480)
    IND[c] = dict(atr96=atr(h, l, cl, 96), atr14=atr(h, l, cl, 14), er=e,
                  ermed=pd.Series(e).rolling(8000, min_periods=2000).median().values,
                  ema={s: ema(cl, s) for s in (800, 1600, 3200)},
                  hi={n: rmax(h, n) for n in (96, 192, 384, 768)},
                  lo={n: rmin(l, n) for n in (96, 192, 384, 768)})


def f1(c, dc, es, atrk, k, gate):
    d = DF[c]; I = IND[c]; cl = d.close.values
    up = cl > I["hi"][dc]; dn = cl < I["lo"][dc]
    e = I["ema"][es]
    L = up & (cl > e); S = dn & (cl < e)
    if gate:
        ok = I["er"] > I["ermed"]
        L &= ok; S &= ok
    return L, S, k * I[atrk]


def run(params, cost=0.0006, arm=1.0, give=0.5, split=None):
    dc, es, atrk, k, gate = params
    ts = []
    for c in COINS_AVAIL:
        L, S, R = f1(c, dc, es, atrk, k, gate)
        t = trades(DF[c], L, S, R, arm=arm, give=give, cost=cost)
        t["coin"] = c
        ts.append(t)
    return pd.concat(ts, ignore_index=True)


def split_stats(t):
    tr = t[t.entry < SPLIT]; te = t[t.entry >= SPLIT]
    return summarize(tr), summarize(te)


if __name__ == "__main__":
    print("coins:", COINS_AVAIL)
    grid = list(itertools.product((96, 192, 384, 768), (800, 1600, 3200), ("atr96",), (2, 3, 4, 6), (False, True)))
    rows = []
    t0 = time.time()
    for p in grid:
        t = run(p)
        a, b = split_stats(t)
        if a is None or b is None:
            continue
        rows.append(dict(dc=p[0], ema=p[1], k=p[3], gate=int(p[4]),
                         trn=a["n"], trE=a["expR"], trT=a["t"], trPF=a["pf"],
                         ten=b["n"], teE=b["expR"], teT=b["t"], tePF=b["pf"],
                         grossTe=b["gross"], ratioTe=b["ratio"], barsTe=b["bars"]))
    R = pd.DataFrame(rows)
    print("grid done in %.0fs, %d variants\n" % (time.time() - t0, len(R)))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("=== top 12 by TRAIN expectancy (n>=150) -> and what they did on the untouched TEST ===")
    top = R[R.trn >= 150].sort_values("trE", ascending=False).head(12)
    print(top.round(3).to_string(index=False))
    print("\n=== how many of %d variants are positive net of cost? ===" % len(R))
    print("train expR>0: %d   test expR>0: %d   BOTH: %d" %
          ((R.trE > 0).sum(), (R.teE > 0).sum(), ((R.trE > 0) & (R.teE > 0)).sum()))
    print("median train expR %+.3f   median test expR %+.3f" % (R.trE.median(), R.teE.median()))
    print("median gross R/trade (test) %+.3f   median entry/R %.0f   median cost drag %.3f R" %
          (R.grossTe.median(), R.ratioTe.median(), (R.grossTe - R.teE).median()))
    print("\nby stop width k (test, mean expR):")
    print(R.groupby("k")[["trE", "teE", "ratioTe", "barsTe"]].mean().round(3))
    print("\nby gate:")
    print(R.groupby("gate")[["trE", "teE"]].mean().round(3))
