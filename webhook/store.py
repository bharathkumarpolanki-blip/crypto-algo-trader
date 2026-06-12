"""
Persistent, thread-safe portfolio ledger for the TradingView webhook bot.

This is the single source of truth for the bot's positions, trade history,
equity curve and the raw alert log. It works identically for paper and live
trading: every fill (simulated or real) is recorded here, so the dashboard shows
one coherent portfolio regardless of mode.

Design notes (professional-grade):
  - One global lock guards all reads/writes — the Flask app is threaded and the
    background mark-to-market loop runs concurrently with inbound webhooks.
  - Writes are atomic (temp file + fsync + os.replace) so a crash mid-write can
    never corrupt the state file — the old state survives intact.
  - P&L is TRUE NET: entry and exit fees are folded into cost basis / proceeds,
    matching how close_position accounts for fees elsewhere in the codebase.
"""
from __future__ import annotations

import os
import json
import time
import threading
from datetime import datetime, timezone

import config

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FILE = os.path.join(_PROJECT_ROOT, "webhook_state.json")

_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _blank_state() -> dict:
    cap = float(config.WEBHOOK_START_CAPITAL_USD)
    return {
        "cash":          cap,     # uninvested USD (virtual in paper, ledger in live)
        "start_capital": cap,     # seed capital — basis for total P&L
        "positions":     {},      # symbol -> open position dict
        "trades":        [],      # closed trade records (newest appended last)
        "alerts":        [],      # raw inbound alert log (most recent last)
        "equity_curve":  [],      # [{t, v}] marked-to-market over time
        "seen_alerts":   [],      # idempotency keys already processed
        "realized_today": {"date": _today(), "pnl": 0.0},  # daily kill-switch tally
        "paused":        False,   # dashboard pause toggle (rejects new entries)
        "created":       _now(),
        "updated":       _now(),
    }


_state: dict = _blank_state()


# ── Load / save ───────────────────────────────────────────────────────────────

def load() -> None:
    """Load persisted state from disk (called once at startup)."""
    global _state
    with _lock:
        if not os.path.exists(_FILE):
            _state = _blank_state()
            return
        try:
            with open(_FILE) as f:
                data = json.load(f)
            base = _blank_state()
            base.update(data)          # tolerate older files missing new keys
            _state = base
            # Roll the daily tally if the file is from a previous UTC day.
            if _state.get("realized_today", {}).get("date") != _today():
                _state["realized_today"] = {"date": _today(), "pnl": 0.0}
        except Exception:
            _state = _blank_state()


def _save_locked() -> None:
    """Atomic write. Caller must already hold _lock."""
    _state["updated"] = _now()
    tmp = _FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(_state, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _FILE)
    except Exception:
        # Persistence is best-effort; never crash the trade path on a disk error.
        pass


def snapshot() -> dict:
    """Deep-ish copy of the full state for read-only dashboard use."""
    with _lock:
        return json.loads(json.dumps(_state, default=str))


# ── Alert log + idempotency ───────────────────────────────────────────────────

def already_seen(alert_id: str) -> bool:
    if not alert_id:
        return False
    with _lock:
        return alert_id in _state["seen_alerts"]


def mark_seen(alert_id: str) -> None:
    if not alert_id:
        return
    with _lock:
        _state["seen_alerts"].append(alert_id)
        if len(_state["seen_alerts"]) > 500:
            _state["seen_alerts"] = _state["seen_alerts"][-500:]


def log_alert(raw: dict, source_ip: str, status: str, message: str,
              symbol: str = "", action: str = "") -> None:
    """Record every inbound alert (accepted or rejected) for the dashboard."""
    with _lock:
        _state["alerts"].append({
            "time":    _now(),
            "ip":      source_ip,
            "symbol":  symbol,
            "action":  action,
            "status":  status,        # accepted | rejected | executed | error | duplicate
            "message": message,
            "raw":     raw,
        })
        if len(_state["alerts"]) > 300:
            _state["alerts"] = _state["alerts"][-300:]
        _save_locked()


# ── Portfolio accessors ───────────────────────────────────────────────────────

def is_paused() -> bool:
    with _lock:
        return bool(_state["paused"])


def set_paused(paused: bool) -> None:
    with _lock:
        _state["paused"] = bool(paused)
        _save_locked()


def get_position(symbol: str) -> dict | None:
    with _lock:
        pos = _state["positions"].get(symbol)
        return dict(pos) if pos else None


def open_position_count() -> int:
    with _lock:
        return len(_state["positions"])


def cash() -> float:
    with _lock:
        return float(_state["cash"])


def daily_realized() -> float:
    """Realized P&L so far for the current UTC day (rolls over automatically)."""
    with _lock:
        rt = _state["realized_today"]
        if rt.get("date") != _today():
            rt = {"date": _today(), "pnl": 0.0}
            _state["realized_today"] = rt
        return float(rt["pnl"])


# ── Mutations (called by the engine after a confirmed fill) ───────────────────

def apply_buy(symbol: str, qty: float, price: float, fee: float,
              meta: dict) -> dict:
    """
    Record a long entry / add. cost includes the entry fee. If a position already
    exists it is averaged up. Returns the resulting position dict.
    """
    with _lock:
        notional = qty * price
        cost = notional + fee
        _state["cash"] -= cost
        pos = _state["positions"].get(symbol)
        if pos:
            new_qty  = pos["qty"] + qty
            # entry = quantity-weighted average FILL price (what the user expects);
            # fees are carried separately in `cost` so P&L stays true-net.
            pos["entry"] = (pos["qty"] * pos["entry"] + qty * price) / new_qty if new_qty else price
            pos["qty"]   = new_qty
            pos["cost"]  = pos["cost"] + cost
            pos["last_action"] = _now()
        else:
            pos = {
                "symbol":    symbol,
                "side":      "long",
                "qty":       qty,
                "entry":     price,                          # avg fill price (fees in `cost`)
                "cost":      cost,                            # total USD invested incl fee
                "opened_at": _now(),
                "last_action": _now(),
                "strategy":  meta.get("strategy", ""),
                "alert_id":  meta.get("alert_id", ""),
                "order_id":  meta.get("order_id", ""),
                "live":      meta.get("live", False),
            }
            _state["positions"][symbol] = pos
        _save_locked()
        return dict(pos)


def apply_sell(symbol: str, qty: float, price: float, fee: float,
               reason: str, meta: dict) -> dict:
    """
    Reduce/close a long. Records a closed-trade entry for the portion sold with
    TRUE NET P&L (both-side fees included). Returns the trade record.
    """
    with _lock:
        pos = _state["positions"].get(symbol)
        if not pos:
            return {}
        qty = min(qty, pos["qty"])
        proceeds = qty * price - fee
        _state["cash"] += proceeds
        # Cost basis attributable to the portion being sold.
        frac = qty / pos["qty"] if pos["qty"] else 1.0
        cost_portion = pos["cost"] * frac
        pnl = proceeds - cost_portion

        # Reduce or remove the position.
        pos["qty"]  -= qty
        pos["cost"] -= cost_portion
        closed_fully = pos["qty"] <= 1e-12
        if closed_fully:
            del _state["positions"][symbol]
        else:
            pos["last_action"] = _now()

        trade = {
            "symbol":     symbol,
            "side":       "long",
            "qty":        qty,
            "entry":      pos["entry"],
            "exit":       price,
            "pnl":        round(pnl, 4),
            "pnl_pct":    round(pnl / cost_portion * 100, 3) if cost_portion else 0.0,
            "reason":     reason,
            "opened_at":  pos.get("opened_at"),
            "closed_at":  _now(),
            "strategy":   meta.get("strategy", pos.get("strategy", "")),
            "alert_id":   meta.get("alert_id", ""),
            "order_id":   meta.get("order_id", ""),
            "live":       meta.get("live", False),
        }
        _state["trades"].append(trade)
        if len(_state["trades"]) > 1000:
            _state["trades"] = _state["trades"][-1000:]

        # Daily realized tally (kill switch input).
        rt = _state["realized_today"]
        if rt.get("date") != _today():
            rt = {"date": _today(), "pnl": 0.0}
        rt["pnl"] = round(rt["pnl"] + pnl, 4)
        _state["realized_today"] = rt

        _save_locked()
        return trade


def mark_to_market(prices: dict[str, float]) -> dict:
    """
    Recompute equity using current prices, append an equity-curve point, and
    return a portfolio summary for the dashboard. `prices` maps symbol -> price.

    Equity is honest *realizable* value: cash + Σ liquidation value of positions
    (qty × price minus the estimated exit fee). Unrealized P&L is what you'd bank
    if you closed everything right now.
    """
    fee_rate = config.FEE_RATE_PCT / 100.0
    with _lock:
        positions_out = []
        holdings_value = 0.0
        unrealized = 0.0
        for sym, pos in _state["positions"].items():
            px = prices.get(sym)
            if px:
                liq = pos["qty"] * px * (1 - fee_rate)   # value if sold now (net of fee)
                pnl = liq - pos["cost"]
                holdings_value += liq
                unrealized += pnl
            else:
                liq = pos["cost"]
                pnl = 0.0
                holdings_value += liq
            positions_out.append({
                "symbol":     sym,
                "side":       pos["side"],
                "qty":        pos["qty"],
                "entry":      round(pos["entry"], 6),
                "price":      round(px, 6) if px else None,
                "value":      round(liq, 2),
                "cost":       round(pos["cost"], 2),
                "pnl":        round(pnl, 2),
                "pnl_pct":    round(pnl / pos["cost"] * 100, 2) if pos["cost"] else 0.0,
                "opened_at":  pos.get("opened_at"),
                "strategy":   pos.get("strategy", ""),
                "live":       pos.get("live", False),
            })

        equity = _state["cash"] + holdings_value
        start  = _state["start_capital"]
        # Append an equity point (throttle: skip if value barely moved & recent).
        curve = _state["equity_curve"]
        point = {"t": _now(), "v": round(equity, 2)}
        if not curve or abs(curve[-1]["v"] - point["v"]) > 1e-6 or \
           (time.time() - _ts(curve[-1]["t"])) > 300:
            curve.append(point)
            if len(curve) > 2000:
                _state["equity_curve"] = curve[-2000:]

        realized = sum(t["pnl"] for t in _state["trades"])
        wins   = [t for t in _state["trades"] if t["pnl"] > 0]
        losses = [t for t in _state["trades"] if t["pnl"] <= 0]
        summary = {
            "equity":        round(equity, 2),
            "cash":          round(_state["cash"], 2),
            "start_capital": round(start, 2),
            "holdings_value": round(holdings_value, 2),
            "total_pnl":     round(equity - start, 2),
            "total_pnl_pct": round((equity - start) / start * 100, 2) if start else 0.0,
            "realized_pnl":  round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "realized_today": round(daily_realized(), 2),
            "open_positions": positions_out,
            "open_count":    len(positions_out),
            "closed_count":  len(_state["trades"]),
            "win_rate":      round(len(wins) / len(_state["trades"]) * 100, 1) if _state["trades"] else 0.0,
            "wins":          len(wins),
            "losses":        len(losses),
            "paused":        _state["paused"],
            "updated":       _now(),
        }
        _save_locked()
        return summary


def _ts(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return 0.0


# ── Admin actions (dashboard) ─────────────────────────────────────────────────

def reset(keep_history: bool = False) -> None:
    """Reset the paper portfolio to its seed capital. Optionally keep trade log."""
    global _state
    with _lock:
        old = _state
        _state = _blank_state()
        if keep_history:
            _state["trades"] = old.get("trades", [])
            _state["alerts"] = old.get("alerts", [])
        _save_locked()
