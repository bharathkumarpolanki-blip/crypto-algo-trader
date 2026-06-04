"""
Thread-safe shared state between the bot loop and the web dashboard.
Both run in the same process (bot in a background thread, Flask in main thread).
"""

import threading
import json
import numpy as np
from datetime import datetime, timezone, date
from dataclasses import dataclass, field, asdict
from typing import Any


class _SafeEncoder(json.JSONEncoder):
    """Convert types that json.dumps can't handle by default."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):           return int(obj)
        if isinstance(obj, (np.floating,)):          return float(obj)
        if isinstance(obj, np.ndarray):              return obj.tolist()
        if isinstance(obj, (datetime, date)):        return obj.isoformat()
        # pandas Timestamp and similar — fall back to str
        try:
            return str(obj)
        except Exception:
            return super().default(obj)


_lock = threading.Lock()

# ── Live state ────────────────────────────────────────────────────────────────

_state: dict[str, Any] = {
    "bot_status":       "stopped",      # "running" | "stopped" | "scanning"
    "last_scan":        None,
    "capital_usdt":     0.0,
    "capital_start":    0.0,
    "open_positions":   [],             # list of position dicts
    "trade_history":    [],             # list of closed trade dicts
    "signals":          [],             # latest scan results per symbol
    "equity_curve":     [],             # [{time, value}, …]  for chart
    "scan_count":       0,
    "errors":           [],
    # ── Backtest state ────────────────────────────────────────────────────────
    "backtest_status":  "idle",
    "backtest_progress": "",
    "backtest_results": [],
    "backtest_last_run": None,
    # ── Universe state ────────────────────────────────────────────────────────
    "universe_symbols":  [],            # current dynamic watchlist
    "universe_log":      [],            # history of refreshes
    # ── ML state ──────────────────────────────────────────────────────────────
    "ml_training":       {"active": False, "progress": ""},
    "tuner":             {"active": False, "progress": "", "result": None},
    # ── Auth health ───────────────────────────────────────────────────────────
    "auth_ok":           None,           # None = unchecked, True/False after check
    "auth_msg":          "",
    # ── Circuit breaker ───────────────────────────────────────────────────────
    "circuit_breaker":   {"status": "active", "reason": ""},
}


def get() -> dict:
    with _lock:
        return json.loads(json.dumps(_state, cls=_SafeEncoder))   # deep copy via JSON


def set_bot_status(status: str) -> None:
    with _lock:
        _state["bot_status"] = status
        _state["last_scan"]  = datetime.now(timezone.utc).isoformat()


def set_capital(value: float, initial: bool = False) -> None:
    with _lock:
        _state["capital_usdt"] = round(value, 2)
        if initial:
            _state["capital_start"] = round(value, 2)
        # append to equity curve every time capital changes
        _state["equity_curve"].append({
            "time":  datetime.now(timezone.utc).isoformat(),
            "value": round(value, 2),
        })
        # keep last 1000 data points
        if len(_state["equity_curve"]) > 1000:
            _state["equity_curve"] = _state["equity_curve"][-1000:]


def set_open_positions(positions: list[dict]) -> None:
    with _lock:
        _state["open_positions"] = positions


def add_trade(trade: dict) -> None:
    with _lock:
        _state["trade_history"].append(trade)
        # keep last 500
        if len(_state["trade_history"]) > 500:
            _state["trade_history"] = _state["trade_history"][-500:]


def set_signals(signals: list[dict]) -> None:
    with _lock:
        _state["signals"] = signals
        _state["scan_count"] += 1
        _state["last_scan"] = datetime.now(timezone.utc).isoformat()


def add_error(msg: str) -> None:
    with _lock:
        _state["errors"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "msg":  msg,
        })
        if len(_state["errors"]) > 50:
            _state["errors"] = _state["errors"][-50:]


# ── Backtest helpers ──────────────────────────────────────────────────────────

def set_backtest_status(status: str, progress: str = "") -> None:
    with _lock:
        _state["backtest_status"]   = status
        _state["backtest_progress"] = progress
        if status in ("done", "error"):
            _state["backtest_last_run"] = datetime.now(timezone.utc).isoformat()


def set_backtest_results(results: list[dict]) -> None:
    with _lock:
        _state["backtest_results"] = results


def set_universe(symbols: list[str], log: list[dict]) -> None:
    with _lock:
        _state["universe_symbols"] = symbols
        _state["universe_log"]     = log


def get_universe() -> dict:
    with _lock:
        return {
            "symbols": _state["universe_symbols"],
            "log":     _state["universe_log"],
        }


# ── ML training state ─────────────────────────────────────────────────────────

def set_ml_training(active: bool, progress: str = "") -> None:
    with _lock:
        _state["ml_training"] = {"active": active, "progress": progress}


def get_ml_training() -> dict:
    with _lock:
        return dict(_state["ml_training"])


# ── Auto-tuner state ──────────────────────────────────────────────────────────

def set_tuner(active: bool, progress: str = "", result: dict | None = None) -> None:
    with _lock:
        prev = _state.get("tuner", {})
        _state["tuner"] = {
            "active":   active,
            "progress": progress,
            # keep previous result unless a new one is supplied
            "result":   result if result is not None else prev.get("result"),
        }


def get_tuner() -> dict:
    with _lock:
        return dict(_state["tuner"])


# ── Auth health state ─────────────────────────────────────────────────────────

def set_auth_status(ok: bool, msg: str) -> None:
    with _lock:
        _state["auth_ok"]  = ok
        _state["auth_msg"] = msg


# ── Circuit breaker state ─────────────────────────────────────────────────────

def set_circuit_breaker(status: dict) -> None:
    with _lock:
        _state["circuit_breaker"] = status


def get_circuit_breaker() -> dict:
    with _lock:
        return dict(_state["circuit_breaker"])


def get_backtest() -> dict:
    with _lock:
        raw = {
            "status":    _state["backtest_status"],
            "progress":  _state["backtest_progress"],
            "results":   _state["backtest_results"],
            "last_run":  _state["backtest_last_run"],
        }
        # Round-trip through safe encoder to normalise any non-serialisable values
        return json.loads(json.dumps(raw, cls=_SafeEncoder))