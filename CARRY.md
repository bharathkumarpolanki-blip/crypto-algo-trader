# Carry Harvester (`carry_bot.py`) — delta-neutral funding harvest

A self-contained service (like `sma_bot.py` / `tradingview_bot.py`) that harvests the
**perpetual-swap funding premium** with a **delta-neutral** book: long spot + short
perp of equal coin quantity per token. Price moves cancel between the legs, so P&L
comes from **funding**, not from predicting direction. This is the cash-and-carry
basis trade every delta-neutral crypto fund runs — a *structural* edge, not a
statistical one.

## Why this exists (the evidence)

After ~26 prediction approaches failed out-of-sample, a multi-year feasibility study
(`research/carry_feasibility.py`, full Binance funding history 2020–2026) showed carry
is real and regime-surviving:

| Token | Net annualized | 2021 bull | **2022 bear** | worst full year | % periods + |
|---|---|---|---|---|---|
| BTC | **+11.9%** | +30.7% | **+4.2%** | +4.2% | 86% |
| ETH | **+14.2%** | +37.6% | **+0.8%** | +0.8% | 86% |
| SOL | +0.1% raw / **+12.7% smart** | +28.7% | −38% / +4.9% smart | smart +2.7% | 72% |

BTC/ETH were positive **every single year, including the 2022 bear** — the test every
directional strategy failed — because the edge isn't directional. Volatile alts (SOL)
**require** the "smart" filter (sit FLAT when funding is negative).

## How it works

- **Book per token:** `long spot_qty coins` + `short perp_qty coins`, equal size →
  delta-neutral. The short perp **earns** funding when funding > 0 (longs pay shorts).
- **Funding accrual:** prorated by elapsed time (`rate / interval_h` per hour). Over a
  day this equals the sum of that day's settlements — same cumulative, smoothed.
- **Smart mode** (`CARRY_SMART`): flip the book **FLAT** when annualized funding ≤
  `CARRY_EXIT_APR`, re-enter when ≥ `CARRY_MIN_APR`. Hysteresis prevents churn around
  zero. **Every flip pays real round-trip fees** — the friction Phase 1 measures.
- **Basis residual:** the small spot−perp price-gap drift is tracked explicitly.
- **Margin health:** the isolated-margin liquidation price of the short is surfaced
  (`liqDist`). Cross-margined against the spot long the book is naturally safe (spot
  gains fund the short's losses); the number is what a live operator must watch.

## Data source

OKX public API (no key, read-only). Binance (451) and Bybit (403) are geo-blocked from
here; OKX serves spot ticker + perp ticker + funding rate fine. USDT-margined perps.

## Run

```bash
python3 carry_bot.py            # live PAPER harvester loop (Ctrl+C to stop; state saved)
python3 carry_bot.py --once     # one cycle, print status, exit
python3 carry_bot.py --status   # print the saved book offline (no network)
```

State persists to `carry_state.json` (atomic writes; gitignored). Let it run for
**days** — funding settles every 8h, so meaningful carry only shows over time.

## Config (`config.py`, `CARRY_*`)

| Key | Default | Meaning |
|---|---|---|
| `CARRY_TOKENS` | `[BTC, ETH, SOL]` | tokens to harvest |
| `CARRY_NOTIONAL_USD` | `1000` | per-leg size per token |
| `CARRY_LEVERAGE` | `2.0` | short-perp leverage (conservative) |
| `CARRY_SMART` | `True` | sit FLAT on negative funding |
| `CARRY_MIN_APR` / `CARRY_EXIT_APR` | `0.0` / `-0.03` | enter / exit funding-APR thresholds (hysteresis) |
| `CARRY_PERP_FEE_PCT` / `CARRY_SPOT_FEE_PCT` | `0.05` / `0.10` | per-leg per-side fees (%) |
| `CARRY_POLL_SECONDS` | `300` | cycle cadence |
| `CARRY_LIVE_ENABLED` | `False` | Phase-2 reserved; **no-op today** |

## ⚠️ PAPER ONLY (Phase 1)

`carry_bot.py` has **no live-execution code path** — it reads public data and simulates
the book. It **physically cannot place a real order**, regardless of any flag.

**What the +12–14% does NOT include — the real, risk-side catches (Phase 2 work):**
1. **Counterparty / exchange failure** — collateral sits on the exchange; an FTX-style
   blow-up can wipe 100%, dwarfing a year of carry. The dominant real risk. Unhedgeable;
   mitigate with tier-1 venues, diversification, and sweeping profits off-exchange.
2. **Short-leg liquidation** — a price rip eats the short's margin; needs conservative
   leverage + auto-deleverage or the hedge breaks and you're left naked long.
3. **Capital efficiency** — a margin buffer means ~12% on notional ≈ **~8–12% on total
   capital**.
4. **Regime compression** — current funding is thin (BTC ≈ +4% APR now vs +30% in 2021
   mania); fat carry appears in bull euphoria, i.e. exactly when blow-up risk is highest.
5. **Tax** — funding is frequent ordinary income.

Phase 2 (live) requires a funded perp venue and hardened margin/liquidation management,
and is gated on explicit owner approval + `DRY_RUN=False` + a real arm switch.
