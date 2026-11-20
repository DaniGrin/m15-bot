# m15-bot — a 4-hour-style trend system run on a 15-minute clock

Paper money only. No API keys, no exchange account, no credit card. Runs free on GitHub Actions
every 15 minutes, on Coinbase public candles.

## The honest summary

This bot was supposed to be "a 15-minute bot". **No pure 15-minute edge survived costs.** What
survives is a slow trend system with wide stops, *watched* at 15-minute resolution: earlier
entries, and a trail that updates every quarter hour instead of every four hours. Typical hold is
2-3 days.

The backtest result is **modestly positive, statistically thin, cost-sensitive and lumpy.** The
numbers are below, including the ugly ones. The six-month forward run is what decides whether it
is real.

## What did not work (research/r1, r2, r8)

Costs are 0.06%/side + 0.02% slippage on stops. The law that killed the ORB bot applies:
break-even gross edge = `2 x cost x (entry / R)`, and a tight stop makes `entry / R` huge.

| idea | gross edge | round-trip cost | verdict |
|---|---|---|---|
| 15m breakouts / trend, 96 variants, stops 2-6 x 15m ATR | +0.02R median | 0.08R | loses; only wider stops improve |
| fade violent 15m moves (z >= 2.5..5, 5 horizons) | +0.31% best case (t=0.9 on test) | 0.16% | not significant |
| BTC jumps, alts follow with a lag | 0.00% to +0.07% | 0.16% | dead (t = -2.5 to -25 net) |

## What survived

```
regime    BTC above its 3200-bar EMA (~33 days): longs only.  Below: shorts only.
entry     close breaks the prior 640-bar (~6.7 day) high/low AND is on the right side of the 1600-bar EMA
R         k x ATR4h  (EMA of the rolling 4-hour high-low range)
exit      stop at 1R. The trail arms at +2R and never gives back more than 1R from the peak (continuous)
risk      0.35% of equity per signal, split evenly across the arm's legs
costs     0.06%/side, +0.02% slippage on stop fills. Perp funding is NOT modelled.
```

The BTC regime filter is the single biggest contributor. It is the same lesson as the trend-bot:
gate the system by regime and the same entries go from weak to positive out of sample.

## Three arms, so one number can't fool us

| arm | universe | stops | backtest CAGR | max DD | Sharpe |
|---|---|---|---|---|---|
| `big_k2` | 9 large caps | k = 2.0 | **+10.0%** | -25.0% | 0.53 |
| `big_3leg` | 9 large caps | k = 2.0 / 2.5 / 3.0, risk split 1/3 | +5.8% | -21.5% | 0.38 |
| `all_k2` | all 19 coins | k = 2.0 | +11.4% | -36.2% | 0.46 |

- `big_3leg` is the same signal with the knife-edge in `k` averaged away. In the early research
  `k = 2` was a sharp peak (k = 1.5 and 2.5 were each ~60% worse), so a single-width arm may be
  flattered. Running both is how we find out.
- `all_k2` is the **control for universe selection**. The 9 large caps were the original research
  universe, fixed before any result was seen. Extending to 19 was a later test, and the 10 added
  coins did not hold up (core arm: Sharpe 0.15 on them alone, and 2025 was negative). The final
  universe was chosen after seeing that, so it is partly fitted. The control tells us on live data
  whether the dilution was real.

Universe: BTC ETH SOL XRP DOGE ADA AVAX LINK LTC (large) + DOT UNI AAVE BCH ETC NEAR XLM ATOM FIL SHIB.

## Every number, including the bad ones

3 years, 2023-11 to 2026-11, 19 Coinbase USD pairs, train < 2025-03 <= test. Drawdown is
mark-to-market at daily closes, so it is a floor, not a ceiling.

| | big_k2 | big_3leg | all_k2 |
|---|---|---|---|
| expectancy / trade, train | +0.080R | +0.059R | +0.046R |
| expectancy / trade, test | +0.079R (t=1.34) | +0.061R (t=1.62) | +0.078R (t=1.90) |
| **cost 0.10%/side** | CAGR +6.5% | +3.4% | +4.8% |
| **cost 0.15%/side** | CAGR +2.2% | +0.5% | **-3.0%** |
| without 2025 | CAGR +4.5% | +3.9% | +10.4% |
| 2026 only | CAGR +0.5% | +0.6% | +4.6% |
| positive weeks | 41% | 40% | 40% |
| median week | -0.32% | -0.46% | -1.09% |
| best 10% of days / total gains | 785% | 1060% | 890% |
| long expectancy / short | +0.113R / +0.039R | +0.117R / -0.010R | +0.103R / +0.017R |

What that means in plain words:

- **No t-statistic reaches 2.** The edge is plausible, not proven.
- **It is cost-sensitive.** At 0.15% per side it is roughly zero. Real execution on a perp
  exchange with taker fees and slippage must stay at or below ~0.08%/side for this to mean anything.
- **It is lumpy.** A median week loses money. Almost all the gain arrives in a few bursts when the
  whole book trends together, which is why a portfolio risk cap hurt rather than helped.
- **It is a long-side edge.** Shorts add nothing reliable. This is crypto beta in bull regimes,
  timed by the BTC filter - not a market-neutral strategy.
- **Do not expect 1-2% a week.** The realistic range is mid-single to low-double-digit % a year at
  20-25% drawdown, if the backtest holds. It may not.

## The research

`research/` holds everything, in order. `core.py` is shared: **the backtest and the live bot import
the same signal code**, and `research/verify.py` replays the bot's stepper over three years and
checks it against the numba backtest. All three arms reproduce **1316 / 3228 / 2699 trades with
zero mismatches** (max difference 4e-16).

| file | what |
|---|---|
| `r1` | does any plain 15m trend entry survive costs? (96 variants, walk-forward) |
| `r2` | event study: reversal vs continuation after violent moves |
| `r3` | 4h-style trend at 15m resolution vs native 4h, matched params |
| `r4`-`r6` | exits, BTC filter, caps, stop-width ensembles, mark-to-market risk |
| `r7` | cross-sectional relative-strength filter (small gain on 10 coins, none on 19; not used) |
| `r8` | BTC -> alt lead-lag |
| `r9`, `verify.py` | final Coinbase backtest and the bot-vs-backtest equivalence check |

## Files

| file | what |
|---|---|
| `bot.py` | the bot |
| `core.py` | signal logic shared with the backtest |
| `state_<arm>.json` | one account per arm |
| `curve_<arm>.csv` | equity (realized and mark-to-market) every 15 minutes |
| `m15_bot.log` | every entry and exit, with the real bar time |

## Running it

```bash
pip install pandas numpy
python bot.py --dry      # fetch + compute, trade nothing
python bot.py --once     # one tick
python bot.py --reset    # wipe all arms back to $10,000
```

The first run starts fresh and never trades history. Each later run processes every closed 15-minute
bar since the last one, so a delayed or skipped GitHub run loses nothing.
