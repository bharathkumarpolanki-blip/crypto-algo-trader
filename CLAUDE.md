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
- [ ] Circuit breaker (halt all trading on extreme market-wide drawdown)
- [ ] Partial-fill handling on live orders
- [ ] Stop-limit gap-through safety net (limit may not fill in a fast gap)

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
