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


# ── Signal: daily 200-SMA  AND  (optional) weekly 30-SMA regime gate ──────────

def _weekly_uptrend(daily_df) -> tuple[bool, float]:
    """
    Resample daily → weekly and check the weekly trend (close > 30-week SMA).
    Returns (is_up, pct_above). The slow-timeframe macro gate.
    """
    wk = daily_df["close"].resample("1W").last().dropna()
    wp = config.SMA_WEEKLY_PERIOD
    if len(wk) < wp + 1:
        return True, 0.0   # not enough weekly history → don't block (neutral)
    wsma = wk.rolling(wp).mean().iloc[-1]
    wclose = wk.iloc[-1]
    return bool(wclose > wsma), float((wclose - wsma) / wsma * 100)


def evaluate() -> dict:
    """
    Per symbol: {"uptrend", "close", "sma", "pct", "weekly_up", "weekly_pct"}
    uptrend = daily close > 200d SMA  AND  (if enabled) weekly close > 30wk SMA.
    """
    out = {}
    period = config.SMA_PERIOD
    buf    = config.SMA_BUFFER_PCT / 100.0
    use_wk = getattr(config, "SMA_USE_WEEKLY_GATE", False)
    # Need enough daily history for BOTH the 200-day SMA and ~30 weeks of resample
    need = max(period + 60, config.SMA_WEEKLY_PERIOD * 7 + 60)
    for sym in config.SMA_SYMBOLS:
        df = fetch_ohlcv(sym, "1d", limit=need)
        if df.empty or len(df) < period + 1:
            logger.warning("Not enough daily data for %s (have %d, need %d)",
                           sym, len(df), period + 1)
            continue
        close = float(df["close"].iloc[-1])
        sma   = float(df["close"].rolling(period).mean().iloc[-1])
        pct   = (close - sma) / sma * 100
        daily_up = close > sma * (1 + buf)

        weekly_up, weekly_pct = (True, 0.0)
        if use_wk:
            weekly_up, weekly_pct = _weekly_uptrend(df)

        # Top-down confluence: BOTH the daily trend AND the weekly regime must be up
        uptrend = daily_up and weekly_up
        out[sym] = {"uptrend": uptrend, "close": close, "sma": sma, "pct": pct,
                    "daily_up": daily_up, "weekly_up": weekly_up, "weekly_pct": weekly_pct}
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
    fee = getattr(config, "FEE_RATE_PCT", 0.6) / 100.0

    # Track a virtual cash balance so paper P&L is accurate. First run → seed it.
    if "cash" not in state:
        state["cash"] = config.TOTAL_CAPITAL_USDT

    # 1. SELL everything leaving the target → proceeds back to cash
    for s in sells:
        h = state["holdings"][s]
        units = h.get("units", 0)
        if units > 0:
            fill = place_market_order_filled(s, "sell", units)
            px   = fill.get("average") or signals[s]["close"]
            proceeds = units * px * (1 - fee)
            state["cash"] += proceeds
            pnl = proceeds - (h.get("units", 0) * h.get("entry", px))
            _alert(f"📤 SOLD *{s}* @ ${px:,.2f} — below {config.SMA_PERIOD}d SMA. "
                   f"Proceeds ${proceeds:,.0f} (P&L {'+'if pnl>=0 else ''}${pnl:,.0f}).")
        del state["holdings"][s]

    # 2. BUY new entrants, equal weight across the FULL target set, from cash
    if buys:
        # Mark-to-market total equity, split equally across all target coins
        equity = _equity(signals, state)
        slice_usd = equity / max(len(target), 1)
        for s in buys:
            spend = min(slice_usd, state["cash"])
            if spend < 5:
                continue
            price = signals[s]["close"]
            qty   = spend / price
            fill  = place_market_order_filled(s, "buy", qty)
            filled = fill.get("filled", qty)
            avg    = fill.get("average", price)
            state["cash"] -= spend
            state["holdings"][s] = {"units": filled, "entry": avg, "since": _now()}
            _alert(f"📥 BOUGHT *{s}* @ ${avg:,.2f} — above {config.SMA_PERIOD}d SMA. "
                   f"Allocated ${spend:,.0f}.")


def _equity(signals: dict, state: dict) -> float:
    """Total portfolio value = virtual cash + current value of all holdings."""
    cash = state.get("cash", config.TOTAL_CAPITAL_USDT)
    held_val = 0.0
    for s, h in state.get("holdings", {}).items():
        px = signals.get(s, {}).get("close")
        if px and h.get("units"):
            held_val += h["units"] * px
    return cash + held_val


def build_paper_summary(signals: dict, state: dict) -> dict:
    """
    Snapshot of paper/live activity for the dashboard. Written into the state
    file each cycle so the dashboard (a SEPARATE process) can read it.
    """
    mode = "ALERT-ONLY" if config.SMA_ALERT_ONLY else ("PAPER" if config.DRY_RUN else "LIVE")
    start = config.TOTAL_CAPITAL_USDT
    equity = _equity(signals, state) if not config.SMA_ALERT_ONLY else start
    cash = state.get("cash", start)
    pnl = equity - start
    positions = []
    for s, h in state.get("holdings", {}).items():
        px = signals.get(s, {}).get("close")
        units = h.get("units")
        entry = h.get("entry")
        ppnl = (units * (px - entry)) if (units and px and entry) else None
        positions.append({
            "symbol": s, "units": units, "entry": entry, "price": px,
            "value": (units * px) if (units and px) else None,
            "pnl": ppnl, "since": h.get("since"),
        })
    return {
        "mode": mode,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "start_capital": start,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / start * 100, 2) if start else 0.0,
        "positions": positions,
        "updated": _now(),
    }


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
        state_str = "🟢 UPTREND (hold)" if v["uptrend"] else "🔴 CASH"
        wk = ""
        if getattr(config, "SMA_USE_WEEKLY_GATE", False):
            wk = f"  | wk {'↑' if v.get('weekly_up') else '↓'}{v.get('weekly_pct',0):+.0f}%"
        # Show WHY it's cash: daily below, weekly below, or both
        why = ""
        if not v["uptrend"]:
            flags = []
            if not v.get("daily_up", True):  flags.append("daily<200d")
            if not v.get("weekly_up", True): flags.append("weekly<30wk")
            why = f"  ({', '.join(flags)})" if flags else ""
        print(f"  {s:9s}  ${v['close']:>10,.2f}  vs 200dSMA ${v['sma']:>10,.2f}  "
              f"{v['pct']:+5.1f}%{wk}   {state_str}{why}")
    held = list(state["holdings"].keys())
    print(f"{'-'*58}")
    print(f"  Currently {'flagged to hold' if config.SMA_ALERT_ONLY else 'holding'}: "
          f"{', '.join(held) if held else 'CASH (nothing in uptrend)'}")

    # Paper/live P&L (only when actually trading, not alert-only)
    if not config.SMA_ALERT_ONLY:
        equity = _equity(signals, state)
        start  = config.TOTAL_CAPITAL_USDT
        pnl    = equity - start
        pct    = pnl / start * 100 if start else 0
        cash   = state.get("cash", start)
        sign   = "+" if pnl >= 0 else ""
        print(f"  Equity: ${equity:,.2f}  (cash ${cash:,.2f})  "
              f"P&L {sign}${pnl:,.2f} ({sign}{pct:.1f}%)")
    print(f"  Next check in {config.SMA_CHECK_HOURS}h\n")


# ── Backtest (shared by CLI + dashboard) ──────────────────────────────────────

def backtest_sma(years: float = 3.0, capital: float = 1000.0,
                 progress=None) -> dict:
    """
    Walk-forward backtest of the SMA200 (+ optional weekly gate) PORTFOLIO:
    each day, hold equal-weight the coins whose daily close > 200d SMA AND
    (if enabled) weekly close > 30wk SMA; cash otherwise. Trades only the delta.
    Fees applied. Returns metrics + equity curve + buy&hold-BTC benchmark.
    """
    import numpy as np
    import pandas as pd
    fee   = getattr(config, "FEE_RATE_PCT", 0.6) / 100.0
    period = config.SMA_PERIOD
    wp     = config.SMA_WEEKLY_PERIOD
    use_wk = getattr(config, "SMA_USE_WEEKLY_GATE", False)
    need   = int(years * 365 + period + 80)

    closes, dsma, wgate = {}, {}, {}
    for sym in config.SMA_SYMBOLS:
        df = fetch_ohlcv(sym, "1d", limit=need)
        if df.empty or len(df) < period + 30:
            continue
        c = df["close"]
        closes[sym] = c
        dsma[sym]   = c.rolling(period).mean()
        if use_wk:
            wk = c.resample("1W").last()
            wsma = wk.rolling(wp).mean()
            up = (wk > wsma)
            wgate[sym] = up.reindex(c.index, method="ffill").fillna(False)
        else:
            wgate[sym] = pd.Series(True, index=c.index)

    if not closes:
        return {"error": "no data"}

    close_p = pd.DataFrame(closes).sort_index()
    n = len(close_p)
    start_i = period + 5
    syms = list(closes.keys())

    cash, holdings, n_trades, equity = capital, {}, 0, []
    for i in range(start_i, n):
        px = close_p.iloc[i]
        def mark():
            return cash + sum(u * px[s] for s, u in holdings.items()
                              if not pd.isna(px.get(s, float("nan"))))
        # target = coins in uptrend today
        target = set()
        for s in syms:
            c_i = px.get(s); sma_i = dsma[s].iloc[i]
            if pd.isna(c_i) or pd.isna(sma_i):
                continue
            if c_i > sma_i and bool(wgate[s].iloc[i]):
                target.add(s)
        cur = set(holdings.keys())
        for s in cur - target:                       # SELL leavers
            cash += holdings[s] * px[s] * (1 - fee); n_trades += 1; del holdings[s]
        entrants = target - set(holdings.keys())
        if entrants:                                  # BUY entrants equal-weight
            slice_v = mark() / max(len(target), 1)
            for s in entrants:
                if slice_v > 1 and cash >= slice_v * 0.5:
                    holdings[s] = (slice_v * (1 - fee)) / px[s]; cash -= slice_v; n_trades += 1
        equity.append(mark())

    eq = pd.Series(equity, index=close_p.index[start_i:])
    end = eq.iloc[-1]; yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (end / capital) ** (1 / yrs) - 1 if yrs > 0 and end > 0 else -1
    maxdd = ((eq - eq.cummax()) / eq.cummax()).min()
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * (365 ** 0.5) if rets.std() > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0

    # Buy & hold BTC benchmark over same window
    btc = close_p["BTC/USD"].iloc[start_i:] if "BTC/USD" in close_p else eq
    bh_u = (capital * (1 - fee)) / btc.iloc[0]; bh = bh_u * btc
    bh_ret = (bh.iloc[-1] / capital - 1) * 100
    bh_dd  = ((bh - bh.cummax()) / bh.cummax()).min() * 100

    # Downsample equity curve for the chart (~150 pts)
    step = max(1, len(eq) // 150)
    curve = [{"t": str(t.date()), "v": round(float(v), 2)}
             for t, v in zip(eq.index[::step], eq.values[::step])]

    return {
        "years": round(yrs, 1),
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "total_return": round((end / capital - 1) * 100, 1),
        "cagr": round(cagr * 100, 1),
        "max_drawdown": round(maxdd * 100, 1),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "trades": n_trades,
        "end_capital": round(end, 2),
        "bh_return": round(bh_ret, 1),
        "bh_drawdown": round(bh_dd, 1),
        "symbols": syms,
        "weekly_gate": use_wk,
        "curve": curve,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_once() -> None:
    state   = load_state()
    signals = evaluate()
    if not signals:
        logger.warning("No signals computed (data issue) — skipping this cycle")
        return
    rebalance(signals, state)
    state["last_check"] = _now()
    state["paper"] = build_paper_summary(signals, state)   # for the dashboard
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


def print_backtest(res: dict) -> None:
    if res.get("error"):
        print("Backtest error:", res["error"]); return
    beat_dd = res["max_drawdown"] > res["bh_drawdown"]   # less negative = better
    print(f"\n{'='*60}")
    print(f"  SMA{config.SMA_PERIOD} TREND PORTFOLIO BACKTEST")
    print(f"  {res['start']} → {res['end']}  ({res['years']}y, {len(res['symbols'])} coins)")
    print(f"  Weekly gate: {'ON' if res['weekly_gate'] else 'off'}  |  fees included")
    print(f"{'='*60}")
    print(f"  Total return : {res['total_return']:+.1f}%   (Buy & Hold BTC: {res['bh_return']:+.0f}%)")
    print(f"  CAGR         : {res['cagr']:+.1f}%")
    print(f"  Max drawdown : {res['max_drawdown']:.1f}%   (Buy & Hold BTC: {res['bh_drawdown']:.0f}%)"
          f"   {'← gentler ✅' if beat_dd else ''}")
    print(f"  Calmar       : {res['calmar']:.2f}")
    print(f"  Sharpe       : {res['sharpe']:.2f}")
    print(f"  Trades       : {res['trades']}")
    print(f"  End capital  : ${res['end_capital']:,.2f}")
    print(f"{'='*60}")
    print(f"  Honest read: trend-following gives up some RETURN vs holding, but")
    print(f"  cuts the DRAWDOWN — a smoother ride, not more money.\n")


if __name__ == "__main__":
    import sys
    if "--backtest" in sys.argv:
        # python3 sma_bot.py --backtest [years] [capital]
        nums = [a for a in sys.argv if a.replace(".", "").isdigit()]
        years   = float(nums[0]) if len(nums) > 0 else 3.0
        capital = float(nums[1]) if len(nums) > 1 else config.TOTAL_CAPITAL_USDT
        print(f"\nRunning SMA backtest: {years}y, ${capital:,.0f} … (fetching daily history)")
        print_backtest(backtest_sma(years=years, capital=capital))
    elif "--once" in sys.argv:
        run_once()     # single check (useful for cron or testing)
    else:
        run()          # continuous loop
