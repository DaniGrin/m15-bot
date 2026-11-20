"""M15 BOT - a 4-hour-style trend system run on a 15-minute clock. Paper money only.

WHY IT LOOKS LIKE THIS (the research behind it is in research/ and in the README)
  Every pure 15-minute edge we tested died on costs: breakouts, shock reversals, BTC->alt lead-lag.
  Gross edge was ~0.00-0.07% per trade against 0.16% round-trip cost. The law from the ORB bot holds:
  break-even gross edge = 2 x cost x (entry / R), and a 15-minute stop makes entry/R huge.
  What survives is the slow trend with wide stops, watched at 15-minute resolution: earlier entries
  and a trail that updates every quarter hour instead of every four hours. Hold time is ~2-3 days.

THREE ARMS, $10,000 paper each, same exits, same regime filter:
  big_k2    the 9 large caps, one stop width (k = 2.0 x ATR4h)      backtest: CAGR +10%, DD -25%, Sharpe 0.53
  big_3leg  the 9 large caps, three widths k = 2.0 / 2.5 / 3.0      backtest: CAGR  +6%, DD -22%, Sharpe 0.38
            (risk split 1/3 each - the same signal with the knife-edge in k averaged away)
  all_k2    all 19 coins, k = 2.0                                    CONTROL. Backtest says the 10 smaller coins
            dilute the edge (Sharpe 0.46 and DD -36%, vs 0.53 and -25%). Forward data settles it.

Risk 0.35% of equity per signal. Costs 0.06%/side plus 0.02% slippage on stop fills.
Perp funding is NOT modelled. Coinbase spot prices; shorts are simulated (a real account would use perps).

  python bot.py --once      one tick (what GitHub Actions runs every 15 minutes)
  python bot.py --dry       fetch + compute, trade nothing
  python bot.py --reset     wipe all arms back to $10,000
"""
import os, sys, json, time, argparse, datetime, traceback, urllib.request
import numpy as np, pandas as pd
import core
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0"}
COINS = "BTC ETH SOL XRP DOGE ADA AVAX LINK LTC DOT UNI AAVE BCH ETC NEAR XLM ATOM FIL SHIB".split()
COST, SLIP = 0.0006, 0.0002
RISK = 0.0035                    # total risk per signal, split across the arm's legs
BIG = "BTC ETH SOL XRP DOGE ADA AVAX LINK LTC".split()
CAPITAL = 10000.0
HISTORY_DAYS = 75                # enough that EMA3200 / Donchian640 / RS2880 are fully warmed up
ARMS = {
    "big_k2":   dict(coins=BIG,   ks=(2.0,)),
    "big_3leg": dict(coins=BIG,   ks=(2.0, 2.5, 3.0)),
    "all_k2":   dict(coins=COINS, ks=(2.0,)),
}


# ------------------------------------------------------------------ plumbing
def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(HERE, "m15_bot.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sfile(arm):
    return os.path.join(HERE, "state_%s.json" % arm)


def fresh():
    return {"equity": CAPITAL, "pos": {}, "pending": [], "last_ts": "", "trades": 0, "wins": 0,
            "gross_R": 0.0}


def load_state(arm):
    return json.load(open(sfile(arm))) if os.path.exists(sfile(arm)) else fresh()


def save_state(arm, s):
    json.dump(s, open(sfile(arm), "w"), indent=1)


# ------------------------------------------------------------------ data
def _call(coin, s, e, tries=6):
    iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    u = ("https://api.exchange.coinbase.com/products/%s-USD/candles?granularity=900&start=%s&end=%s"
         % (coin, iso(s), iso(e)))
    last = None
    for a in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA),
                                                     timeout=30).read())
        except Exception as ex:
            last = ex
            time.sleep(1 + 2 * a)
    raise RuntimeError("coinbase %s: %s" % (coin, last))


def fetch(coin, days=HISTORY_DAYS):
    end = int(time.time())
    start = end - days * 86400
    rows, e = [], end
    while e > start:
        s = max(start, e - 300 * 900)
        rows += _call(coin, s, e)
        e = s
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "vol"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df.ts, unit="s")
    return df.set_index("ts")[["open", "high", "low", "close", "vol"]]


def gather(coins=COINS):
    with ThreadPoolExecutor(4) as ex:
        raw = dict(zip(coins, ex.map(fetch, coins)))
    # the newest candle is still forming: keep only fully closed 15m bars
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("15min") - core.BAR
    return {c: core.clean(d[d.index <= end], end) for c, d in raw.items()}, end


# ------------------------------------------------------------------ the stepper (same semantics as eng.sim)
def run_bars(st, cfg, dfs, sg, grid, risk=RISK, log_fn=None, record=None):
    """Advance one arm's state across `grid` (fully closed bars, ascending).
    Order inside a bar, exactly as in the numba simulator the backtest uses:
      1. fill entries decided at the previous bar's close, at THIS bar's open
      2. manage every open position against THIS bar's high/low (stop first, then ratchet the trail)
      3. look for new signals on THIS bar's close -> they become pending for the next bar
    """
    ks, arm_at, give = cfg["ks"], cfg["arm"], cfg["give"]
    leg_risk = risk / len(ks)
    ix = {c: dfs[c].index for c in dfs}
    arr = {c: (dfs[c].open.values, dfs[c].high.values, dfs[c].low.values, dfs[c].close.values) for c in dfs}
    pos_of = {c: ix[c].get_indexer(grid) for c in dfs}
    for n, t in enumerate(grid):
        # 1. fills
        for p in st["pending"]:
            o = arr[p["coin"]][0][pos_of[p["coin"]][n]]
            key = "%s|%.1f" % (p["coin"], p["k"])
            qty = st["equity"] * leg_risk / p["R"]
            st["pos"][key] = dict(coin=p["coin"], k=p["k"], side=p["side"], entry=float(o), R=p["R"],
                                  qty=qty, peak=0.0, stop=float(o) - p["side"] * p["R"], opened=str(t))
            if log_fn:
                log_fn("  [%s] ENTER %-5s k=%.1f %-5s @ %12.6f  R %10.6f  stop %12.6f  (%s)"
                       % (cfg["name"], p["coin"], p["k"], "LONG" if p["side"] == 1 else "SHORT", o, p["R"],
                          float(o) - p["side"] * p["R"], t))
        st["pending"] = []
        # 2. manage
        for key in list(st["pos"].keys()):
            p = st["pos"][key]
            o, h, l, c = (a[pos_of[p["coin"]][n]] for a in arr[p["coin"]])
            done, px = False, 0.0
            if p["side"] == 1:
                if l <= p["stop"]:
                    px, done = min(p["stop"], o) * (1 - SLIP), True
                else:
                    p["peak"] = max(p["peak"], (h - p["entry"]) / p["R"])
                    if p["peak"] >= arm_at:
                        p["stop"] = max(p["stop"], p["entry"] + (p["peak"] - give) * p["R"])
            else:
                if h >= p["stop"]:
                    px, done = max(p["stop"], o) * (1 + SLIP), True
                else:
                    p["peak"] = max(p["peak"], (p["entry"] - l) / p["R"])
                    if p["peak"] >= arm_at:
                        p["stop"] = min(p["stop"], p["entry"] - (p["peak"] - give) * p["R"])
            if done:
                gross = p["side"] * (px - p["entry"]) / p["R"]
                pnl = p["side"] * (px - p["entry"]) * p["qty"] - 2 * COST * p["entry"] * p["qty"]
                net = gross - 2 * COST * p["entry"] / p["R"]
                st["equity"] += pnl
                st["trades"] += 1
                st["gross_R"] += net
                st["wins"] += 1 if pnl > 0 else 0
                if record is not None:
                    record.append(dict(coin=p["coin"], k=p["k"], side=p["side"], entry=p["opened"],
                                       exit=str(t), net=net))
                if log_fn:
                    log_fn("  [%s] EXIT  %-5s k=%.1f %-5s @ %12.6f  %+5.2fR  PnL %+8.2f  equity %9.2f  (WR %d/%d)  (%s)"
                           % (cfg["name"], p["coin"], p["k"], "LONG" if p["side"] == 1 else "SHORT", px,
                              net, pnl, st["equity"], st["wins"], st["trades"], t))
                del st["pos"][key]
        # 3. signals
        for c in dfs:
            i = pos_of[c][n]
            if i < 0:
                continue
            L, S, A = sg[c]["L"][i], sg[c]["S"][i], sg[c]["A"][i]
            if not (L or S) or not np.isfinite(A) or A <= 0:
                continue
            for k in ks:
                key = "%s|%.1f" % (c, k)
                if key in st["pos"] or any(q["coin"] == c and q["k"] == k for q in st["pending"]):
                    continue
                st["pending"].append(dict(coin=c, k=k, side=1 if L else -1, R=float(k * A)))
        st["last_ts"] = str(t)
    return st


def mark(st, dfs):
    u = 0.0
    for p in st["pos"].values():
        u += p["side"] * (dfs[p["coin"]].close.iloc[-1] - p["entry"]) * p["qty"]
    return st["equity"] + u


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset:
        for arm in ARMS:
            save_state(arm, fresh())
            for f in ("curve_%s.csv" % arm,):
                p = os.path.join(HERE, f)
                if os.path.exists(p):
                    os.remove(p)
        log("=== all arms reset to $%.0f ===" % CAPITAL)
        return

    dfs, end = gather()
    log("=== M15 tick | %d coins | last closed bar %s UTC ===" % (len(dfs), end))
    btc = dfs["BTC"]
    bull = btc.close.values[-1] > core.ema(btc.close.values, core.DEFAULT["btcs"])[-1]
    log("  BTC regime: %s (close %.0f)" % ("BULL -> longs only" if bull else "BEAR -> shorts only",
                                          btc.close.values[-1]))
    sig = core.signals(dfs, None, False)
    if a.dry:
        for arm, cfg in ARMS.items():
            live = [c for c in cfg["coins"] if sig[c]["L"][-1] or sig[c]["S"][-1]]
            log("  [%s] coins with a live entry signal on the last bar: %s" % (arm, live or "none"))
        return

    for arm, cfg in ARMS.items():
        st = load_state(arm)
        full = {**core.DEFAULT, **cfg, "name": arm}
        sub = {c: dfs[c] for c in cfg["coins"]}
        subsig = {c: sig[c] for c in cfg["coins"]}
        if not st["last_ts"]:
            st["last_ts"] = str(end)          # fresh start: history is only for indicator warm-up
            save_state(arm, st)
            log("  [%s] initialised fresh at $%.0f - trading only NEW bars from now" % (arm, st["equity"]))
        else:
            grid = dfs["BTC"].index[dfs["BTC"].index > pd.Timestamp(st["last_ts"])]
            if len(grid):
                run_bars(st, full, sub, subsig, grid, log_fn=log)
                save_state(arm, st)
            else:
                log("  [%s] no new closed bar yet" % arm)
        m = mark(st, sub)
        log("  [%s] realized $%.2f (%+.2f%%) | mark-to-market $%.2f (%+.2f%%) | %d open | %d closed, WR %s"
            % (arm, st["equity"], (st["equity"] / CAPITAL - 1) * 100, m, (m / CAPITAL - 1) * 100,
               len(st["pos"]), st["trades"],
               ("%d%%" % round(100 * st["wins"] / st["trades"])) if st["trades"] else "-"))
        cf = os.path.join(HERE, "curve_%s.csv" % arm)
        new = not os.path.exists(cf)
        with open(cf, "a", encoding="utf-8") as fh:
            if new:
                fh.write("ts,realized,mtm,open\n")
            fh.write("%s,%.2f,%.2f,%d\n" % (end, st["equity"], m, len(st["pos"])))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        sys.exit(1)
