"""15m research engine. numba simulator + indicator helpers. One position per coin, long or short.

Entry fills at the NEXT bar's open (never the signal bar's close). The stop is checked on every bar
BEFORE that bar's high/low is allowed to ratchet the trail (conservative: the adverse extreme is assumed
to come first). Net R = gross R - 2 * cost_per_side * (entry / R)  -- the law that killed the ORB bot.
"""
import os
import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "BNB", "LTC"]
SPLIT = pd.Timestamp("2025-03-01")          # train: 2023-11 .. 2025-02   test: 2025-03 .. now


def load(c):
    return pd.read_csv(os.path.join(D, c + ".csv"), index_col=0, parse_dates=True)


def ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().values


def er(c, n):
    s = pd.Series(c)
    return ((s - s.shift(n)).abs() / s.diff().abs().rolling(n).sum().replace(0, np.nan)).values


def rmax(x, n):
    return pd.Series(x).rolling(n).max().shift(1).values


def rmin(x, n):
    return pd.Series(x).rolling(n).min().shift(1).values


@njit(cache=True)
def sim(o, h, l, c, sigL, sigS, Rarr, arm, give, maxhold, slip):
    n = len(c)
    cap = n // 3 + 10
    ei = np.empty(cap, np.int64); xi = np.empty(cap, np.int64)
    sd = np.empty(cap, np.int8); rg = np.empty(cap); ra = np.empty(cap)
    k = 0
    pos = 0; entry = 0.0; R = 0.0; stop = 0.0; peak = 0.0; e_i = 0
    for i in range(n - 1):
        if pos != 0:
            done = False; px = 0.0
            if pos == 1:
                if l[i] <= stop:
                    px = min(stop, o[i]) * (1.0 - slip); done = True
                else:
                    pk = (h[i] - entry) / R
                    if pk > peak: peak = pk
                    if peak >= arm:
                        ns = entry + (peak - give) * R
                        if ns > stop: stop = ns
            else:
                if h[i] >= stop:
                    px = max(stop, o[i]) * (1.0 + slip); done = True
                else:
                    pk = (entry - l[i]) / R
                    if pk > peak: peak = pk
                    if peak >= arm:
                        ns = entry - (peak - give) * R
                        if ns < stop: stop = ns
            if (not done) and maxhold > 0 and (i - e_i) >= maxhold:
                px = c[i]; done = True
            if done:
                ei[k] = e_i; xi[k] = i; sd[k] = pos
                rg[k] = pos * (px - entry) / R; ra[k] = entry / R
                k += 1; pos = 0
        if pos == 0:
            s = 0
            if sigL[i]: s = 1
            elif sigS[i]: s = -1
            if s != 0 and Rarr[i] > 0 and not np.isnan(Rarr[i]):
                pos = s; entry = o[i + 1]; R = Rarr[i]; peak = 0.0; e_i = i + 1
                stop = entry - s * R
    return ei[:k], xi[:k], sd[:k], rg[:k], ra[:k]


def trades(coin_df, sigL, sigS, Rarr, arm=1.0, give=0.5, maxhold=0, cost=0.0006, slip=0.0002):
    o, h, l, c = (coin_df[x].values for x in ("open", "high", "low", "close"))
    ei, xi, sd, rg, ra = sim(o, h, l, c, sigL, sigS, Rarr, arm, give, maxhold, slip)
    net = rg - 2 * cost * ra
    t = pd.DataFrame({"entry": coin_df.index[ei], "exit": coin_df.index[xi], "side": sd,
                      "gross": rg, "net": net, "ratio": ra,
                      "bars": xi - ei})
    return t


def summarize(t, label="", ret=False):
    if len(t) == 0:
        return None
    r = t.net.values
    se = r.std() / np.sqrt(len(r)) if len(r) > 1 else np.nan
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    out = dict(n=len(r), expR=r.mean(), t=r.mean() / se if se else np.nan,
               wr=(r > 0).mean() * 100, pf=gp / gl if gl > 0 else np.inf,
               gross=t.gross.mean(), ratio=t.ratio.mean(), bars=t.bars.mean())
    return out


def portfolio(all_t, risk=0.0075, start=10000.0):
    """Chronological compounding by exit time, risking `risk` of current equity per trade.
    Approximation: ignores intra-trade drawdown, so reported maxDD is a floor not a ceiling."""
    if not len(all_t):
        return None
    t = all_t.sort_values("exit")
    eq = start * np.cumprod(1.0 + risk * t.net.values)
    s = pd.Series(eq, index=t.exit.values)
    dd = (s / s.cummax() - 1).min() * 100
    yrs = (t.exit.max() - t.entry.min()).days / 365.25
    cagr = ((eq[-1] / start) ** (1 / max(yrs, 1e-9)) - 1) * 100
    return dict(final=(eq[-1] / start - 1) * 100, cagr=cagr, dd=dd, curve=s)
