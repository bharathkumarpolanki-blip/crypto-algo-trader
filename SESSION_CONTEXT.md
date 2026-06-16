# Session Context — Crypto Bot Edge Investigation

> A reconstruction of the full multi-day working session (substance, not a verbatim
> transcript). Purpose: a memory/context file so work can resume without re-deriving
> everything. Companion to `RESEARCH_FINDINGS.md` (technical results) — this file adds
> the narrative arc, decisions, code changes, and current state.

Last updated: 2026-06-11.

---

## 0. TL;DR — the bottom line

- **No deployable alpha exists** in any freely-available data we tested (price, funding,
  open interest, on-chain, ML, unsupervised, cross-sectional, forced-flow). Proven ~20+ ways
  under strict T+1 execution + fees + slippage + walk-forward + OOS + permutation + bootstrap.
- The **only deployable artifact** is risk-managed crypto-beta via the **200-day SMA**
  (`sma_bot.py`) — drawdown control, NOT alpha. It won a production exposure-engine bake-off
  (best Calmar 0.36, lowest MaxDD −68%, best exposure-efficiency).
- The **1h engine (`bot.py`) is NOT tradeable** — composite signal IC = −0.037 (significant,
  anti-predictive). It is hard-blocked from live trading and kept as a sandbox + dashboard.
- The most valuable thing built is **`validate.py`** — the deployability gate that repeatedly
  caught overfitting (incl. a lookahead bug) before any capital was risked.

---

## 1. How the session started

The user runs a professional-grade crypto bot (Coinbase via CCXT). Two systems:
- `bot.py` — 1h multi-strategy engine (16 components + ML + sentiment) with a Flask dashboard.
- `sma_bot.py` — standalone 200-day SMA daily trend system.

Initial asks evolved from "why is it slow / not trading?" and "the backtest looks bad" into a
full, honest edge investigation. Standing user constraints throughout:
- **Be honest, never passify.** Surface real numbers. Never oversell.
- Professional-grade always (CLAUDE.md): risk-first, robust, no blocking the trade loop.
- Commit only on explicit say-so. Work on `dev`; `main` is protected. Author commits as the
  user only (NO `Co-Authored-By: Claude`).

---

## 2. The investigation arc (what we tested, in order)

1. **1h trend engine** — composite IC **−0.037 (p=0.03)**, anti-predictive; lost on all 13
   watchlist coins after fees.
2. **Shorts bug** — shorts were structurally unreachable (`score>=MIN_SIGNAL_SCORE` rejected
   every short). Fixed with `passes_conviction()` + a daily-200-SMA gate. Result: losses became
   *symmetric*, not positive (no edge either side).
3. **1h mean-reversion rewrite** (`core/mean_reversion.py`) — real positive IC but **sub-fee**
   (gross +11.7% at zero cost → −76% at retail fees).
4. **Edge validation (IC)** — composite IC negative; trend components (ema/ichimoku/regime) are
   the most anti-predictive; only **volume** (+0.039) is significantly positive.
5. **Horizon / signal discovery** — inverting trend signals flips IC positive; edge peaks
   ~48–72h but is sub-cost; confidence scores are non-monotonic (meaningless).
6. **Daily/weekly systems** — Weekly Trend beats Buy&Hold on risk over 6.8y but FAILS a 15-phase
   deployability audit (OOS collapse, bull-beta only, p=0.12 not significant).
7. **Regime-adaptive system** — looked great (Sharpe 1.46) but had a **same-bar LOOKAHEAD bug**;
   corrected to T+1 → Sharpe 0.50, MaxDD −76%. Edge was an artifact. Retracted.
8. **Full T+1 audit of all candidates** — under honest execution, nothing clears Sharpe>1 / +OOS.
   Weekly SMA & Breakout = "drawdown-management only" (B); Rotation & Regime = "no edge" (A).
9. **WeeklySMA deep forensic** — risk-management overlay only; benefit not statistically
   significant (permutation p≈0.12); protects in slow bears, fails fast/choppy ones.
10. **Edge discovery (categories)** — momentum, mean-reversion, vol, breadth, etc.: best gross
    edge ~+0.12% vs 1.30% cost. No category survives costs.
11. **ML interactions** (tree/RF/GBM) — train IC 0.77 → **OOS negative**. Overfitting; no
    nonlinear structure. Gini importance contradicted OOS permutation importance.
12. **Cross-asset lead-lag** — contemporaneous corr 0.82 but lag-1 ≈ 0; the 30d BTC→ETH effect is
    the shared trend factor, fails OOS.
13. **Funding rate** (OKX, 93d) — a hint (IC +0.158) but redundant with price (ΔR² +0.5%, p=0.64),
    negatively skewed (−0.95 = tail premium), data-capped.
14. **Open interest** (OKX, 110d) — theory-consistent quadrant signals but regime-confounded,
    redundant, data-capped.
15. **On-chain MVRV/NUPL** (Coin Metrics, 12y) — biggest in-sample IC (−0.23) but **dead OOS**
    (−0.03). Few cycles → large-sample illusion.
16. **Unsupervised regimes** (HMM/GMM/KMeans/Bayesian) — significant in-sample (p=0.0000),
    **dead OOS** (p 0.18–0.94). 0/4 methods generalize.
17. **Cross-sectional dollar-neutral** — momentum has a REAL gross edge (perm p=0.039, OOS gross
    +0.79) but net −2.05 at spot fees and sub-1.0 Sharpe even gross. The closest real signal.
18. **Forced-flow framing** — the one genuinely-untested frontier (ETF flows, dealer gamma, token
    unlocks, liquidations, index rebalances). Analytically demolished ETF flows (≥90% impact
    pre-publication; flow is majority sentiment-following-price). Could not acquire free data here.
19. **Risk-intelligence audit** — signals predict **volatility** (IC up to 0.25) far more than
    returns (the "category error"). But mostly via trivial **vol-persistence** (autocorrelation).
20. **Fragility Score** — survived one OOS split, then **DESTROYED by a frozen adversarial audit**:
    out-of-universe IC +0.005 (ns); vol-control killed it (raw +0.027 → partial **−0.034**); only
    worked in low-vol; failed small-caps. It was vol-persistence in disguise. Retired.
21. **Production exposure-engine bake-off** — Fragility engine was WORST (CAGR −9%, MaxDD −79%,
    Calmar −0.11, harmful). **200d SMA WON** (Calmar 0.36, MaxDD −68%, exposure-efficiency 44).
22. **"Fix the losing trades" test** (`research/trade_filter_test.py`) — learned a loser-avoidance
    filter on in-sample trades (WR 48%→74%, looked amazing) → froze it → **OOS still LOSES**
    (−$12.50/trade). Proof that winners/losers are not separable at entry; classic overfitting.

---

## 3. Code changes made this session (all on `dev`)

- **`core/strategies.py`** — `passes_conviction()` (direction-aware), `daily_trend_gate()`
  (200d SMA macro gate), wired into `analyse()`. Added `conviction`/`daily_trend` to SignalResult.
- **`bot.py`** — Tier-0 hard block (`ENGINE_1H_LIVE_ENABLED`) so the 1h engine can never place
  real orders; uses `passes_conviction`; fetches daily candles for the gate.
- **`backtest.py`** — daily-gate plumbing; direction-aware entry gate.
- **`ml/signal_predictor.py`** — rebuilt to 3-class multiclass (long/short/sideways); short is now
  a LEARNED class, not 1−P(long); macro OVR AUC; robust to old pickles.
- **`config.py`** (evidence-driven): WATCHLIST → BTC/ETH/SOL; SENTIMENT_WEIGHT → 0;
  ML_SCALE_POSITIONS → False; ENTRY_ORDER_TYPE → maker; MIN_WIN_FEE_MULTIPLE 3→4;
  DAILY_TREND_GATE added; ENGINE_1H_LIVE_ENABLED=False; ML_RETRAIN_HOURS kept at 6 (reverted a
  24h experiment per user). Added SMA_DASHBOARD / SMA_DASHBOARD_PORT (8081) for sma_bot's own UI.
- **`exchange/universe.py`** — skip ambiguous numeric-base market ids (e.g. `00/USD`) that crashed
  ccxt `safeMarket()`; blacklist + numeric-base filter.
- **`sma_bot.py` hardening**:
  - BUG FIX: never liquidate a position on a data-fetch failure
    (`sells = (held & evaluated) − target`).
  - Atomic state writes (tmp + fsync + os.replace) → no `sma_state.json` corruption on crash.
  - 2-attempt resilient data fetch per symbol.
  - Graceful Ctrl+C shutdown; CLI `--backtest`; paper-trading P&L tracking.
- **Dashboard SMA wiring** (`ui/dashboard.py`, `ui/state.py`) — SMA Trend tab: status table,
  backtest button + equity chart, paper-activity panel (reads `sma_state.json`), 5s polling.
- **Repo cleanup** — root went 41 → 5 `.py` files (`bot.py`, `config.py`, `sma_bot.py`,
  `backtest.py`, `validate.py`); 36 research scripts archived under `research/` (+ README).
  Finalized `RESEARCH_FINDINGS.md`.

> IN-PROGRESS at last checkpoint: wiring the dashboard server INTO `sma_bot.py` (so one process
> serves both). Config flags added (`SMA_DASHBOARD`, `SMA_DASHBOARD_PORT`); the `start_server()`
> call in `sma_bot.run()` was NOT yet added. Dashboard imports only `ui.state` at module level
> (safe — won't pull in bot.py). Resume here if continuing that feature.

---

## 4. Key conclusions / mental models established

- **More inputs ≠ more edge.** The price already contains all public analysis (indicators, news,
  sentiment, visible whale moves). Stacking signals adds noise + fees + overfitting, not signal.
- **Frequency × fees is the killer.** ~1.3% round-trip cost destroys any high-frequency signal.
  The only winners trade rarely (sma_bot ~10–30 trades/yr).
- **Returns are unpredictable OOS; risk (volatility) is more predictable** — but mostly via
  trivial vol-autocorrelation, not novel signals.
- **In-sample brilliance ≠ OOS edge.** Every overfit signal (MVRV, ML, Fragility, regime,
  loser-filter) looked spectacular in-sample and collapsed OOS. The discipline that caught this
  (T+1, walk-forward, permutation, frozen audits) is the real product.
- **Professionals' edges are where retail can't reach** — microstructure/order-flow, market-making,
  options/vol-premium, forced-flow — not "better analysis of public data."

---

## 5. Current deployable state

- **Run for real:** `python3 sma_bot.py` (200d SMA; alert-only by default; set
  `SMA_ALERT_ONLY=False` + `DRY_RUN=True` for paper trading with P&L).
- **Dashboard:** `python3 bot.py` → http://localhost:8081 (1h engine is a DRY_RUN sandbox,
  hard-blocked from live orders). [Pending: sma_bot self-serving the dashboard.]
- **Backtests:** `python3 backtest.py --multi` (1h engine); `python3 sma_bot.py --backtest 3 1000`
  (SMA); `python3 validate.py` (deployability gate); `research/*.py` (the full investigation).
- **Safety:** `DRY_RUN=True`, `ENGINE_1H_LIVE_ENABLED=False`. Nothing trades real money.
- **Git:** all work committed/pushed to `dev`. `main` untouched (PR is the user's call).

---

## 6. Open threads / where to go next (honest EV ranking)

1. **Strip `bot.py` to the lean version** — cut zero/negative-IC components (ema_trend, ichimoku,
   regime, macd, adx, supertrend, candles, support), keep volume (+ weak oscillators), re-measure.
   Cleaner & cheaper, but will NOT create edge. (Offered, not yet done.)
2. **Finish wiring the dashboard into `sma_bot.py`** (one-process UI). Mechanical, low-risk.
3. **If chasing the remaining frontier:** forced-flow / vol-premium event studies — but they need
   PAID data (Deribit options, Coinglass funding/OI/liquidations, ETF flow feeds) and have only
   ~15–35% success odds each. Highest-EV single bet ≈ ETF flow-following (cheapest, fastest) or
   token-unlock event study (best forward-visibility). All require data we couldn't access here.
4. **Accept the honest endpoint:** run sma_bot for risk-managed beta; treat bot.py as a sandbox.

---

## 7. Files that matter (post-cleanup)

- Root (the bot): `bot.py`, `config.py`, `sma_bot.py`, `backtest.py`, `validate.py`
- Docs: `RESEARCH_FINDINGS.md` (technical results), `SESSION_CONTEXT.md` (this file), `CLAUDE.md`
- `research/` — 36 archived investigation scripts (+ `research/README.md` index)
- `core/`, `exchange/`, `ml/`, `risk/`, `data/`, `ui/`, `notifications/` — bot packages

---

## 8. Continuation (2026-06-16) — "trade on time, not after the fact" → first real finding

After saving this file, the session continued and produced the **first genuinely
positive, statistically-validated result** of the whole investigation.

### 8.1 The reframe
User internalised "the bot trades AFTER the fact (price already moved, whales already
acted)" and asked how to trade ON TIME. Established there are only 3 escapes:
1. **Be faster** (HFT/latency) — ❌ not reachable for retail.
2. **Know the flow before it happens** (forward-visible/scheduled forced flows) — ✅ best path.
3. **Be the liquidity** (market-making) — ⚠️ possible, competitive.
(+ go slower so the reaction doesn't matter = what sma_bot does.)

### 8.2 "Fix the losing trades" — disproved on the user's own idea
`research/trade_filter_test.py`: learned a loser-avoidance filter on in-sample 1h
trades (WR 48%→74%, looked great) → frozen → OOS still LOSES (−$12.50/trade). Proof
that winners/losers aren't separable at entry; classic overfitting.

### 8.3 The forced-flow pivot + data reality
Ranked forced-flow participants (ETF flows, dealer gamma, token unlocks, liquidations,
index rebalances, bankruptcy, miners, treasury, stablecoins). Highest-EV retail path =
**forward-visible scheduled flows**. Data probing (all $0 attempts):
- Farside 403, SoSoValue 404, CoinGlass needs key, DefiLlama /emissions = **402 (paywalled)**.
- **CoinGecko free WORKS** (price + market_cap) but capped at **365 days**.
- **Deribit options data = FREE** (full chain).

### 8.4 Free tests run
- **Options-expiry calendar** (`research/options_expiry_study.py`) — significant in-sample,
  **collapses OOS** (bull-drift confound). Dead at daily/free level.
- **Token-unlock event study** (`research/token_unlock_study.py` + `_placebo.py`) —
  detect unlocks from CoinGecko supply step-ups (supply = mc/price), measure BTC-relative
  abnormal returns. 27 events: consistent NEGATIVE sign across holdout (no sign-flip!) but
  underpowered (p=0.126). First non-sign-flipping hint.
- **EXPANDED to ~70 tokens** (`research/token_unlock_full.py`) → **326 events, 44 tokens**.

### 8.5 THE FIRST REAL FINDING — unlock-DAY effect
- **Unlock DAY abnormal return = −1.5% vs BTC, p=0.003** (full), **−1.31% p=0.015 on
  HELD-OUT tokens** (OOS-surviving).
- Pre-drift (front-run-into-it thesis) FAILED (sign-flips, ns). Post-drift ~0. Only the DAY.
- **Day-effect placebo = DECISIVE**: real unlock days −1.61% vs random days −0.01%,
  **permutation p=0.0000 → UNLOCK-SPECIFIC** (n=342). Not general alt-weakness.
- First effect to pass ALL gates: significant + OOS holdout + placebo-confirmed +
  forward-visible + sound mechanism (scheduled forced supply → selling).

### 8.6 Remaining gates before "deployable"
1. **Artifact check (critical):** "unlock day" = CoinGecko supply-jump-detected day. Could be
   a timestamp artifact. MUST re-confirm using REAL scheduled unlock dates (DefiLlama).
2. **Tradability:** −1.6% on small alts; round-trip cost 0.5–1.5%; most NOT shortable
   (no borrow/perp). Deployable forms are narrow: short-where-possible / avoid-buying on unlock day.
3. **Multi-regime:** all 365d / one regime; confirm bull vs bear.

### 8.7 Next action (in progress)
User getting **DefiLlama 7-day Pro free trial** → export historical unlock schedules via
**DefiLlama Sheets** (free; CSV is paid-Pro-only) → save as `research/data/unlock_dates.csv`
(cols: `cg_id,date`). Then run **`research/token_unlock_confirm.py`** — the frozen real-date
confirmation harness (already built): re-runs the day-effect placebo on REAL dates. If real
dates also give ~−1.6% (p<0.05) → CONFIRMED, not artifact → tradability study. If ~0 → artifact, lead dead.

DefiLlama tiers: Free shows upcoming unlocks (dashboard only); **Pro $49/mo** (7-day trial,
Sheets export, CSV paid-only); **API $300/mo** (programmatic unlocks endpoint — overkill, skip).

### 8.8 New research files (this continuation)
`research/trade_filter_test.py`, `research/options_expiry_study.py`,
`research/token_unlock_study.py`, `research/token_unlock_placebo.py`,
`research/token_unlock_full.py`, `research/token_unlock_confirm.py` (real-date harness),
`research/data/README.md` (CSV format for the DefiLlama export).

### 8.9 Status of the bottom line
Mostly unchanged — BUT for the first time there is a **live, validated lead** (token-unlock-day
effect) pending one real-date confirmation. Everything else remains: no deployable alpha;
sma_bot (200d SMA) is the deployable risk system; bot.py is a sandbox.
