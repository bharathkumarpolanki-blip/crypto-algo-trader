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
