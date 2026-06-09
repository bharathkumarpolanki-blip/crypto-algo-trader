# Research Findings — Does this bot have a tradeable edge?

Honest summary of the strategy research. All tests use real Coinbase data,
real fees, and no look-ahead bias. The conclusion is consistent and negative
for active retail trading — documented here so it isn't forgotten.

## The question
Can a retail trader profitably trade crypto with this bot on $1,000 capital?

## Tests run and results

| Test | Setup | Result | File |
|---|---|---|---|
| 1h strategy, with fees | 175d, full signal engine | **−92% (SOL)** | `backtest.py` |
| 1h parameter sweep | 5 configs, fees | **No config PF > 1.0** | `tune.py` |
| 1h at ZERO fees (Coinbase One) | BTC/SOL | **PF 1.02–1.04 = coin flip** | `compare_fees.py` |
| 1h "best single pick of many coins" | 8 coins scanned hourly | **0 trades clear the fee bar** | `best_pick_backtest.py` |
| Daily SMA200, single asset | BTC/ETH, 5.5y | **Matches hold, ½ the drawdown** | `daily_backtest.py` |
| Daily momentum rotation (15 coins) | top-5, rebalanced | **−70% (buys tops)** | `daily_portfolio_backtest.py` |
| Buy & hold BTC | 3.5–5.5y | **+273%, best Sharpe** | benchmark |

## Conclusions (honest)

1. **1-hour trading has no edge.** Even at zero fees it's a coin flip (PF ~1.03).
   With fees it loses badly. Short timeframes are dominated by noise.
2. **Momentum / "trade what's performing" loses** — in crypto the hottest coins
   mean-revert violently; chasing them = buying tops (−70% test).
3. **More signals ≠ more edge.** A 16-component score still scored a coin flip.
   Stacking indicators/sentiment/news overfits; it doesn't predict.
4. **Sentiment/news give retail no edge** — by the time we read an RSS feed the
   move is priced in. Real news-trading needs millisecond speed we don't have.
5. **The only non-loser:** dead-simple daily SMA200 trend filter on a quality
   asset (BTC/ETH) — and its merit is DRAWDOWN REDUCTION (≈−40% vs −77%), NOT
   beating buy-and-hold on return.
6. **Buy & hold BTC beat everything** we built, on return and risk-adjusted return.

## The honest recommendation
- Do not trade actively with real money — the edge isn't there.
- If you want crypto exposure: hold BTC/ETH, optionally with a 200-day SMA
  filter to halve drawdowns (a behavioural tool, not a profit engine).
- Keep this bot as a DRY_RUN paper-trading / learning sandbox.
- Coinbase One ($360/yr) is a trap on small capital (36% of $1,000/yr).

## What this project IS good at
Professional-grade engineering: risk management, exchange-side stops, circuit
breaker, partial-fill handling, ML infra, dashboard, AND — most valuable — an
honest, leak-free, fee-accurate testing framework that PREVENTED a real-money
loss by exposing that strategies which looked profitable were not.

## Additional tests (timeframe + portfolio variations)

| Test | Setup | Result | File |
|---|---|---|---|
| Daily full-engine portfolio | top-5 scoring, 3.5y, fees | **−18.9%, −75% DD (worse than hold)** | `tf_portfolio_backtest.py 1d` |
| Weekly full-engine portfolio | top-5 scoring, data-thin | **0 trades (sat in cash)** | `tf_portfolio_backtest.py 1w` |

KEY FINDING: the sophisticated 16-component signal engine on DAILY bars LOST
money and had a WORSE drawdown than buy & hold — while the dumb one-line
"price > 200d SMA" rule was the only non-loser. More analysis made it WORSE.
Complexity is the enemy. Tested across 1h / 1d / 1w, single-pick and top-N
portfolio, momentum and full-engine — the conclusion is identical every time:
no tradeable edge exists for retail here. Hold BTC/ETH (optional 200d filter).

## Breakout / Retest / Floor strategies (daily, long-only, fees)

Tested at user request — the popular price-action strategies.

| Strategy | BTC return | ETH return | vs Buy&Hold |
|---|---|---|---|
| Buy & Hold | +188% (Calmar 2.45) | +152% (Calmar 1.92) | — |
| Donchian breakout | −15% | +107% | lost on both |
| Breakout + retest | −31% (10% win) | −55% (**0% win**) | lost badly |
| Support floor bounce | −57% | −20% | lost on both |

KEY FINDING: every breakout/retest/floor strategy lost to buy & hold; most lost
money outright. Breakout+retest had a 0–10% win rate — the "fakeout" problem made
visible. Support/resistance levels are the most-watched lines on the chart, so
they are exactly where stop-hunts and fakeouts concentrate; retail breakout
buyers become exit liquidity. File: `breakout_backtest.py`.

## FINAL VERDICT (all approaches tested)
1h scalping · daily/weekly full-engine · momentum rotation · best-of-many-coins ·
breakout/retest/floor — EVERY approach loses to simply holding BTC. The pattern
never broke. No retail TA strategy has a tradeable edge here. The only endorsed
active tool is the SMA200 daily filter (sma_bot.py) — drawdown reduction, not alpha.

## Information-Coefficient deep dive (the decisive evidence)

Moved beyond trade P&L to measure whether the signal PREDICTS forward returns
(Spearman IC vs N-bar-ahead return, cost-independent). Tools: `edge_validation.py`,
`signal_discovery.py`, `horizon_discovery.py`.

- **Composite 1h signal IC = -0.037 (p=0.03)** — statistically significant but
  NEGATIVE. Higher score predicts LOWER returns. The engine is on the wrong side.
- **Per-component IC:** trend signals are the poison — `ema_trend` -0.057,
  `ichimoku` -0.055, `regime` -0.053 (all p<0.003). Only `volume` (+0.039) and
  `stochastic` (+0.031) are positive & significant. RSI/Bollinger ~random.
- **Inverting trend signals flips IC positive** (InvIchimoku +0.069 — strongest
  predictor found). Confirms 1h crypto MEAN-REVERTS; trend-following is backwards.
- **Confidence score is invalid** — forward return is non-monotonic across score
  buckets; the highest-confidence bucket was the WORST.
- **Holding-period analysis:** edge builds from 1H to a peak at 48-72H, then dies
  by 1 week. Gross edge per trade first exceeds the 1.30% round-trip cost at ~72H
  — but even there a real (non-tail) system isn't profitable (`mr_validation.py`,
  `horizon_discovery.py` Phase 4).
- **Mean-reversion rewrite** (`core/mean_reversion.py`): genuinely positive IC and
  profitable at ZERO fees (+11.7% avg), but -76% after retail fees. Edge is real
  but ~11x too small to beat costs. SUB-FEE, not deployable.

## Timeframe inversion (where the edge actually lives)

`htf_systems.py` — native DAILY/WEEKLY systems over 6.8y, fees on:
- Weekly Trend Following BEATS Buy&Hold on return, Sharpe AND drawdown
  (BTC +531%/Sh0.85/-60%DD; ETH +1009%/Sh0.89/-57%DD).
- Daily Mean-Reversion LOSES (-27%/-55%).
- **The trend relationship INVERTS with timeframe:** fade trend at 1h, follow
  trend at daily/weekly. The bot was trend-following on the one timeframe where
  trend-following is wrong.

## Weekly-Trend deployability audit (15-phase, `wtf_robustness.py`)

Even the best candidate FAILS deployment: parameter- and cost-robust, but
out-of-sample collapses (IS PF 5.34 -> OOS PF 0.43, Sharpe +0.80 -> -0.41),
profitable only 3/8 years, edge is 100% bull-regime beta. Verdict: NOT deployable
as alpha — it is drawdown-controlled long beta. The integrated rotation+regime
redesign (`core/rotation_strategy.py`) also FAILED `validate.py` (OOS -6%, Sharpe 0.91).

## Permanent guardrails added this round
- `validate.py` — unified deployability GATE (OOS PF>1.2, Sharpe>1.0, positive
  OOS CAGR, cost-robust, walk-forward stable). Nothing goes to paper without it.
- `ENGINE_1H_LIVE_ENABLED=False` — the anti-predictive 1h engine is hard-blocked
  from real orders regardless of DRY_RUN.
- ML signal predictor rebuilt to 3-class (long/short/sideways) — short is now a
  LEARNED class, not 1-P(long). Correctness fix; AUC still ~0.55 (weak).

## Config tuned to the evidence
WATCHLIST -> BTC/ETH/SOL only (alts bled worst). SENTIMENT_WEIGHT 1->0 (no edge).
ML_SCALE_POSITIONS True->False (confidence invalid). ENTRY_ORDER_TYPE taker->maker
(cut fee drag ~33%). MIN_WIN_FEE_MULTIPLE 3->4. ML_RETRAIN_HOURS 12->6.

## FINAL STANDING CONCLUSION (unchanged, now over-proven)
No retail TA strategy — simple or complex, 1h or daily, long or short, trend or
mean-reversion, single-coin or rotation — passes an honest out-of-sample + cost
gate as ALPHA. The only thing that clears the bar is slow trend-following
(`sma_bot.py`) used for DRAWDOWN CONTROL on quality assets (BTC/ETH). Trade rarely,
pay maker fees, hold quality, sidestep bear markets. That is the whole game.

# Lessons Learned

This section captures the methodological lessons from the full strategy-search and
forensic-validation effort. These matter more than any single backtest number.

## 1. Lookahead bias discovery
- **How it occurred.** The regime-adaptive engine decided each day's exposure using
  that same day's *close* (the 200d SMA, slope, and trend gate all included
  `close[T]`), then applied that exposure to the *same day's* return. That is
  same-bar lookahead: trading on information not available until the bar had
  already closed.
- **How it was detected.** A final forensic pass re-ran the strategy with the
  decision lagged one bar (decide on `close[T]`, earn `bret[T+1]` — the only
  realistic alignment). The audit explicitly compared lag-0 vs lag-1.
- **Why it invalidated previous results.** Under realistic T+1 execution the
  regime system collapsed: **Sharpe 1.46 → 0.50, CAGR +65.6% → +13.1%, MaxDD
  −33% → −76%.** The entire apparent edge — and the earlier "robust / paper-trading
  candidate" verdict and its walk-forward/OOS numbers — were artifacts of the
  one-bar lookahead. They were retracted. Slow signals (Weekly SMA) survived the
  lag; the regime system's edge was almost entirely same-bar crash-day timing.

## 2. Why OOS testing matters
- Several strategies looked attractive **in-sample**: the 1h engine, daily
  breakout, rotation, weekly trend, and the regime system all produced
  good-looking full-sample curves at some point.
- **Most failed out-of-sample.** Under strict T+1 on the newest 20%: WeeklySMA
  −14%, Breakout −23%, Regime −22%, Rotation −48% — every candidate lost money on
  unseen data. Full-sample performance was repeatedly a poor predictor of OOS.

## 3. Why significance testing matters
- **Sharpe improvements alone were insufficient.** WeeklySMA's Sharpe (0.84) beat
  buy-and-hold (0.72), which looked like a win.
- **p-values changed the interpretation.** A permutation test (random timing, same
  exposure) and a randomized-entry test (same % time invested, random days) both
  returned **p ≈ 0.12–0.13** — not significant. The bootstrap Sharpe 95% CI was
  [+0.12, +1.59]. The "edge" could not be statistically distinguished from simply
  being invested ~55% of the time at random. A higher Sharpe was not, by itself,
  evidence of skill.

## 4. Difference between alpha and risk management
- Weekly SMA **improved some risk metrics**: MaxDD −77% → −60%, Calmar 0.37 → 0.51,
  Ulcer 37.8 → 35.3, Return/MaxDD 5.81 → 8.70.
- It **did not create excess return.** Attribution showed it *lags* buy-and-hold
  inside bull markets (captured ~47%), does *worse* in sideways, and adds value
  only by sitting out bears. Its returns come from *participation* in up-trends,
  not timing or shorting alpha. It also had a *lower* Sortino than holding.

## 5. Research conclusion
- **No statistically validated alpha was found** — across 1h/daily/weekly,
  trend/mean-reversion/breakout/rotation/regime, single-asset and portfolio,
  long and (attempted) short.
- **The best result was modest drawdown reduction** (Weekly SMA), and even that is
  statistically marginal (p ≈ 0.12) and protects only in slow bears (2020, 2022),
  failing in fast crashes (2021) and choppy bears (2025).

## If this were my project
I would consider the strategy-search phase **complete**. The most valuable artifact
is not a trading bot — it is the **validation framework**, which now enforces:
- T+1 execution enforcement
- Lookahead detection
- Walk-forward testing
- Out-of-sample testing
- Slippage modeling
- Monte Carlo analysis
- Permutation testing
- Randomized-entry comparisons

That framework is worth more than any single strategy tested, because it prevents
future research from repeating the same mistakes.

Based on the evidence, the conclusion is:
**No statistically significant, deployable trading edge was demonstrated.** The
strongest surviving approach is a drawdown-management overlay that modestly alters
risk characteristics but does not provide validated alpha.
