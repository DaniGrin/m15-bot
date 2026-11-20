"""Shared signal logic. The backtest AND the live bot import this one file, so they cannot drift apart.

Strategy ("M15"): a 4-hour-style trend system run on a 15-minute clock.
  regime   BTC above its 3200-bar EMA (~33 days) -> only longs, below -> only shorts
  entry    close breaks the prior 640-bar (~6.7 day) high/low AND sits on the right side of the 1600-bar EMA
  rs       optional: coin must also be stronger/weaker than the universe median over 2880 bars (30 days)
  R        k x ATR4h, where ATR4h = EMA(span 224) of the rolling 16-bar (4h) high-low range
  exit     stop at 1R; trail arms at +2R and never gives back more than 1R from the peak (continuous)
"""
import numpy as np
import pandas as pd

BAR = pd.Timedelta("15min")
DEFAULT = dict(dc=640, es=1600, btcs=3200, atr_win=16, atr_span=224, rs_n=2880,
               arm=2.0, give=1.0)


def clean(df, end=None):
    """Reindex to a gap-free 15m grid. A missing candle means nothing traded: carry the last close flat."""
    df = df[~df.index.duplicated()].sort_index()
    idx = pd.date_range(df.index[0], end if end is not None else df.index[-1], freq="15min")
    df = df.reindex(idx)
    df["close"] = df["close"].ffill()
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])
    df["vol"] = df["vol"].fillna(0.0)
    return df


def ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def atr4h(h, l, win, span):
    rng = pd.Series(h).rolling(win).max().values - pd.Series(l).rolling(win).min().values
    return pd.Series(rng).ewm(span=span, adjust=False).mean().values


def rs_frame(closes, n):
    """Cross-sectionally demeaned (vs median) n-bar log return. closes: DataFrame coin -> close."""
    r = np.log(closes / closes.shift(n))
    return r.sub(r.median(axis=1), axis=0)


def signals(dfs, cfg=None, use_rs=False):
    """dfs: coin -> cleaned DataFrame. Returns coin -> dict(L, S, A) numpy arrays on each coin's own index."""
    cfg = {**DEFAULT, **(cfg or {})}
    btc = dfs["BTC"]
    btc_bull = pd.Series(btc.close.values > ema(btc.close.values, cfg["btcs"]), index=btc.index)
    rs = None
    if use_rs:
        grid = pd.DataFrame({c: d.close for c, d in dfs.items()}).sort_index().ffill()
        rs = rs_frame(grid, cfg["rs_n"])
    out = {}
    for c, d in dfs.items():
        h, l, cl = d.high.values, d.low.values, d.close.values
        hi = pd.Series(h).rolling(cfg["dc"]).max().shift(1).values
        lo = pd.Series(l).rolling(cfg["dc"]).min().shift(1).values
        e = ema(cl, cfg["es"])
        L = (cl > hi) & (cl > e)
        S = (cl < lo) & (cl < e)
        bull = btc_bull.reindex(d.index).ffill().fillna(False).astype(bool).values
        L &= bull
        S &= ~bull
        if use_rs:
            r = rs[c].reindex(d.index).ffill().values
            L &= (r > 0)
            S &= (r < 0)
        L &= np.isfinite(hi) & np.isfinite(e)
        S &= np.isfinite(lo) & np.isfinite(e)
        out[c] = dict(L=L, S=S, A=atr4h(h, l, cfg["atr_win"], cfg["atr_span"]))
    return out
