"""
SMA200 Daily Trend Bot — sma_bot.py

The simple, research-endorsed strategy, as a clean STANDALONE tool (deliberately
NOT part of the complex 1h bot — simplicity is the entire point).

Rule, checked once per day:
  - For each coin: is the daily close above its 200-day SMA (by SMA_BUFFER_PCT)?
      YES → the coin should be HELD
      NO  → the coin should be in CASH
  - Capital is split equally across the coins currently in uptrend.
  - When a coin crosses below its SMA, exit to cash; when it crosses back above,
    re-enter. That's the whole strategy.

Honest expectations: this does NOT beat buy & hold on total return. Its value is
roughly halving the drawdown (≈-40% vs -77%) — a behavioural/risk tool, not a
profit engine. See RESEARCH_FINDINGS.md.

Modes:
  SMA_ALERT_ONLY = True  → only sends Telegram alerts on trend flips (you act)
  SMA_ALERT_ONLY = False → actually trades (paper if DRY_RUN, else live)

Run:  python3 sma_bot.py
"""
from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime, timezone

import config
from exchange.market_data import (fetch_ohlcv, get_quote_balance,
                                  place_market_order_filled)
from notifications.notifier import send_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] sma_bot: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("sma_bot.log")],
)
logger = logging.getLogger("sma_bot")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sma_state.json")


# ── State persistence (which coins we currently hold) ─────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            pass
    return {"holdings": {}, "last_check": None}


def save_state(state: dict) -> None:
    try:
        json.dump(state, open(STATE_FILE, "w"), indent=2)
    except Exception as e:
        logger.warning("Could not save state: %s", e)


# ── Signal: is each coin above its 200-day SMA? ───────────────────────────────

def evaluate() -> dict:
    """
    Return {symbol: {"uptrend": bool, "close": float, "sma": float, "pct": float}}
    pct = how far price is above(+)/below(-) the SMA, in %.
    """
    out = {}
    period = config.SMA_PERIOD
    buf    = config.SMA_BUFFER_PCT / 100.0
    for sym in config.SMA_SYMBOLS:
        df = fetch_ohlcv(sym, "1d", limit=period + 60)
        if df.empty or len(df) < period + 1:
            logger.warning("Not enough daily data for %s (have %d, need %d)",
                           sym, len(df), period + 1)
            continue
        close = float(df["close"].iloc[-1])
        sma   = float(df["close"].rolling(period).mean().iloc[-1])
        pct   = (close - sma) / sma * 100
        # Require a buffer beyond the line to flip (anti-whipsaw)
        uptrend = close > sma * (1 + buf)
        out[sym] = {"uptrend": uptrend, "close": close, "sma": sma, "pct": pct}
    return out


# ── Decide target holdings (equal weight across uptrending coins) ─────────────

def target_holdings(signals: dict) -> set[str]:
    return {s for s, v in signals.items() if v["uptrend"]}


# ── Trading (only when not alert-only) ────────────────────────────────────────

def rebalance(signals: dict, state: dict) -> None:
    target  = target_holdings(signals)
    current = set(state["holdings"].keys())

    sells = current - target          # fell below SMA → exit
    buys  = target - current          # crossed above SMA → enter

    # ── Alert-only mode: notify on any change, don't trade ────────────────────
    if config.SMA_ALERT_ONLY:
        for s in sells:
            _alert(f"🔻 *{s}* dropped below its {config.SMA_PERIOD}-day average "
                   f"({signals[s]['pct']:+.1f}%). Consider moving to CASH.")
        for s in buys:
            _alert(f"🔺 *{s}* climbed above its {config.SMA_PERIOD}-day average "
                   f"({signals[s]['pct']:+.1f}%). Consider BUYING.")
        # Track intended state so we only alert on changes
        state["holdings"] = {s: {"since": _now()} for s in target}
        return

    # ── Trading mode (paper if DRY_RUN, else live) ────────────────────────────
    # 1. SELL everything leaving the target
    for s in sells:
        units = state["holdings"][s].get("units", 0)
        if units > 0:
            fill = place_market_order_filled(s, "sell", units)
            _alert(f"📤 SOLD *{s}* — below {config.SMA_PERIOD}d SMA "
                   f"({signals[s]['pct']:+.1f}%). Exited to cash.")
        del state["holdings"][s]

    # 2. BUY new entrants, equal weight across the full target set
    if buys:
        cash = get_quote_balance() if not config.DRY_RUN else config.TOTAL_CAPITAL_USDT
        # Reserve equal slices for ALL target coins; spend on the new ones
        slice_usd = cash / max(len(target), 1)
        for s in buys:
            price = signals[s]["close"]
            qty   = slice_usd / price
            fill  = place_market_order_filled(s, "buy", qty)
            filled = fill.get("filled", qty)
            avg    = fill.get("average", price)
            state["holdings"][s] = {"units": filled, "entry": avg, "since": _now()}
            _alert(f"📥 BOUGHT *{s}* — above {config.SMA_PERIOD}d SMA "
                   f"({signals[s]['pct']:+.1f}%). Allocated ${slice_usd:,.0f}.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alert(msg: str) -> None:
    logger.info(msg.replace("*", ""))
    try:
        send_telegram(msg)
    except Exception:
        pass


# ── Status print ──────────────────────────────────────────────────────────────

def print_status(signals: dict, state: dict) -> None:
    mode = "ALERT-ONLY" if config.SMA_ALERT_ONLY else ("PAPER" if config.DRY_RUN else "LIVE")
    print(f"\n{'='*58}")
    print(f"  SMA{config.SMA_PERIOD} DAILY TREND BOT   |   mode: {mode}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*58}")
    for s, v in signals.items():
        state_str = "🟢 UPTREND (hold)" if v["uptrend"] else "🔴 DOWN (cash)"
        print(f"  {s:9s}  ${v['close']:>10,.2f}  vs SMA ${v['sma']:>10,.2f}  "
              f"{v['pct']:+5.1f}%   {state_str}")
    held = list(state["holdings"].keys())
    print(f"{'-'*58}")
    print(f"  Currently {'flagged to hold' if config.SMA_ALERT_ONLY else 'holding'}: "
          f"{', '.join(held) if held else 'CASH (nothing in uptrend)'}")
    print(f"  Next check in {config.SMA_CHECK_HOURS}h\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_once() -> None:
    state   = load_state()
    signals = evaluate()
    if not signals:
        logger.warning("No signals computed (data issue) — skipping this cycle")
        return
    rebalance(signals, state)
    state["last_check"] = _now()
    save_state(state)
    print_status(signals, state)


def run() -> None:
    mode = "ALERT-ONLY" if config.SMA_ALERT_ONLY else ("PAPER" if config.DRY_RUN else "LIVE")
    logger.info("SMA%d daily trend bot starting | symbols=%s | mode=%s | check every %dh",
                config.SMA_PERIOD, config.SMA_SYMBOLS, mode, config.SMA_CHECK_HOURS)
    send_telegram(f"📈 *SMA{config.SMA_PERIOD} trend bot started* — "
                  f"watching {', '.join(config.SMA_SYMBOLS)} ({mode} mode).")
    while True:
        try:
            run_once()
        except Exception as e:
            logger.error("Cycle error: %s", e)
        time.sleep(config.SMA_CHECK_HOURS * 3600)


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        run_once()     # single check (useful for cron or testing)
    else:
        run()          # continuous loop
