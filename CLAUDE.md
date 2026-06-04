# Project: crypto-algo-trader

## ⭐ CORE PRINCIPLE — PROFESSIONAL GRADE ALWAYS

**This bot is held to the same standard as professional/institutional trading bots.
Every change must be production-grade. No shortcuts, no toy implementations.**

When building anything here, default to the professional choice:

- **Risk first.** Protect capital before chasing profit. Exchange-side stop orders,
  hard risk caps, circuit breakers, and graceful degradation are the baseline.
- **Robustness.** Handle API failures, partial fills, restarts, network timeouts,
  and degenerate data without crashing. Never leave half-initialised state.
- **No blocking the trade loop.** Slow I/O (Telegram, news, training) runs off the
  hot path. The trading loop must never stall on a network call.
- **Correctness over cleverness.** If a simpler design is more reliable, choose it.
- **Persistence.** Positions, models, and state survive restarts.
- **Honesty.** Be transparent about what works, what's weak (e.g. ML AUC ~0.56),
  and what's risky. Never oversell. Surface real numbers.
- **Observability.** Clear logs, dashboard visibility, Telegram alerts, health checks.
- **Test before claiming done.** Verify with real data, not just imports.

If a feature can be done at a basic level or a professional level, always do the
professional level — even if the user didn't explicitly ask for it. Mention the
trade-offs, but build it right.

---

## Architecture (packages)

```
bot.py / backtest.py / config.py   — entry points + config (project root)
core/        — indicators, strategies, candle_patterns (signal engine)
exchange/    — market_data (Coinbase CCXT), universe (dynamic discovery)
risk/        — risk_manager, position_store (persistence)
data/        — sentiment (Fear&Greed + CoinGecko + RSS, no keys)
notifications/ — notifier (Telegram, non-blocking)
ml/          — feature_builder, preprocessor, signal_predictor,
               regime_classifier, extrema_predictor, auto_tuner (FreqAI-inspired)
ui/          — dashboard (Flask + web UI), state (thread-safe shared state)
```

## Key facts

- **Exchange:** Coinbase Advanced Trade. Public data uses a KEYLESS client
  (`get_public_exchange`); trading uses the authenticated client. This avoids the
  401 on `transaction_summary` that breaks public candle fetches.
- **Timeframes:** 1h signals, 6h trend gate (Coinbase has no 4h).
- **ML:** sklearn HistGradientBoosting (no system deps — avoids LightGBM/libomp).
  `early_stopping=False` everywhere to avoid degenerate-split crashes on skewed data.
  AUC guard ≥0.53 ignores models that don't beat random. Models persist to `models/`.
- **Branch model:** work on `dev`, PR into `main`. `main` is protected (PR + review).
  Commits author as the user only — do NOT add Co-Authored-By: Claude.
- **Secrets:** `.env` is gitignored. Never commit keys. `models/`, `positions.json`,
  `trading_bot.log` are gitignored.
- **Licensing:** FreqAI (GPL-3.0) was analysed for concepts only — no code copied.

## Known professional-grade gaps to keep improving

- [x] Exchange-side stop-loss/take-profit orders (stop-limit + limit, manual OCO)
- [x] Decoupled fast position-monitoring loop (POSITION_CHECK_SECONDS, own thread)
- [x] Circuit breaker (risk/circuit_breaker.py — 4 trip conditions, auto-cooldown)
- [x] Partial-fill handling on live orders (place_market_order_filled)
- [x] Stop-limit gap-through safety net (force market exit if price gaps past stop)

All originally-identified professional-grade gaps are now closed. Keep raising
the bar — add new gaps here as they're discovered.

- [x] Fee accounting + profit-floor gate + daily profit lock
- [x] Limit/maker-order entries to cut fee drag (post-only + taker fallback)

## Backtest correctness (IMPORTANT)

The backtest had 3 bugs that made it misleading:
1. Candle cap: `min(days*24+300, 1000)` capped ALL backtests to ~41 days.
   Removed — now fetches full requested history (Coinbase has ~180-190 days of
   1h candles; reports the ACTUAL tested span honestly when shorter than asked).
2. Look-ahead bias: analyse() ran ML models trained on RECENT data against
   HISTORICAL candles. Added `include_ml` param to analyse(); backtest passes
   include_ml=False so ML never contaminates backtest results.
3. No fees: backtest P&L was gross. Now subtracts round-trip taker fees
   (_round_trip_fees) so backtest P&L is TRUE NET, matching live.
Backtest is rule-based + candlesticks only (the honest, leak-free signal set).
Live trading adds ML on top — its benefit can only be proven by forward/paper
testing, never by backtest (would be look-ahead).

## ML performance (training speed)

Training was ~45s/model and ran twice (no concurrency guard). Fixed:
- `_ml_training_lock` in bot.py — only ONE training pass at a time (no 2× waste).
- feature_builder LAG_CANDLES [1,2,3,5,10]→[1,3], PERIODS dropped 28: 328→154
  features. Faster AND less overfitting (test AUC now stable ~0.55 vs wild swings).
- permutation_importance: n_repeats 3→1, subsample 120, gated by
  ML_COMPUTE_IMPORTANCE (importance is dashboard-only, not needed for trading).
Result: signal model train 45s → ~6s (7×). AUC unchanged/healthier.

## Maker entries (fee reduction)

ENTRY_ORDER_TYPE = "taker" (default) | "maker". Maker places a post-only limit at
the best bid (buy) / best ask (sell) — guaranteed maker fee (MAKER_FEE_RATE_PCT
~0.4% vs taker ~0.6%). May not fill → waits ENTRY_FILL_TIMEOUT_SEC, then either
falls back to a market/taker order (ENTRY_FALLBACK_TO_TAKER=True) or skips.
place_maker_entry() in market_data; _place_entry_order() dispatches in bot.py.
The profit-floor gate still uses the conservative taker rate, so maker fills only
make trades MORE profitable than the gate assumes.

## Fees & profit discipline (IMPORTANT — honest economics)

Coinbase taker fee ≈ 0.6%/side (~1.2% round trip). On a $200 position that's
~$2.40 — price must move ~1.7% just to net $1. A trade can go the right way and
still lose to fees. So:
- close_position subtracts round_trip_fees → all P&L shown is TRUE NET.
- Profit-floor gate (MIN_NET_PROFIT_USD): the bot refuses any trade whose target
  can't net ≥ $1 after fees. Fewer but genuinely-worth-it trades.
- Daily profit lock (DAILY_PROFIT_TARGET_USD, 0=off): once up X for the UTC day,
  stop opening new trades.
- NEVER promise profit on every run — impossible. The bot improves the odds of
  green runs; it cannot guarantee them. Biggest lever to hit the user's profit
  goal is reducing fees (limit/maker entries, bigger account, volume tiers).

## Gap-through safety net (professional-grade)

In a violent move price can blow past BOTH the stop trigger and the stop-limit's
limit price without filling (no buyers at the limit). The fast monitor detects
this: if price has moved STOP_GAP_BUFFER_PCT (0.5%) beyond the stop but the
stop-limit order is still open, it cancels both protective orders and fires a
MARKET exit (always fills; accepts slippage to guarantee the position closes).
Telegram alert on trigger. Only active in live+protected mode.

## Order fill handling (professional-grade)

- `place_market_order_filled()` places a market order and confirms the ACTUAL
  fill: polls fetch_order until closed, returns real filled qty + average price.
- Entry: position qty + stops are sized to the ACTUAL filled quantity, and the
  stop/target are recomputed from the real average fill price (preserves R:R).
  filled<=0 → no position opened. Partial → Telegram alert + size to filled.
- Exit: uses the real exit fill price for P&L; a partial close retries the
  remainder once, then alerts if it still can't fully flatten.
- DRY_RUN simulates a full fill at the current ticker price.

## Circuit breaker (kill switch) — risk/circuit_breaker.py

Halts NEW entries (existing positions keep their stops — no panic-sell). Four
independent trip conditions, any one trips: daily loss (CB_MAX_DAILY_LOSS_PCT),
drawdown from high-water mark (CB_MAX_DRAWDOWN_PCT), consecutive losses
(CB_MAX_CONSECUTIVE_LOSSES), market crash (BTC down CB_MARKET_CRASH_PCT in the
lookback window — correlation protection). Auto-resets after CB_COOLDOWN_HOURS;
manual reset/trip from the dashboard. Telegram alert on trip + reset. Status
shown as a red dashboard banner. Checked in try_open_trade via breaker.can_trade().

## Exit handling design (professional-grade)

- On open (live): place exchange-side stop-limit (stop) + limit (take-profit).
  The exchange enforces them instantly even if the bot is slow/down.
- Coinbase has NO native OCO → manual OCO: when one protective order fills, the
  monitor cancels the sibling.
- Trailing stop (live): when price moves favourably, cancel the old exchange stop
  and re-place it tighter.
- Fast monitor thread (_position_monitor_loop) checks exits every
  POSITION_CHECK_SECONDS (45s) — independent of the 5-min entry scan. Guarded by
  _positions_lock so entry loop + monitor never double-close.
- DRY_RUN simulates all protective orders (no real orders placed).
- If a live stop order is rejected → fall back to poll-based monitoring + Telegram
  alert (never silently unprotected).
