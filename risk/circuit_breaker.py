"""
Circuit Breaker — risk/circuit_breaker.py

Professional-grade kill switch. Halts NEW trade entries when the portfolio or the
market enters dangerous territory. Existing positions keep their exchange-side
protective stops — we do NOT panic-sell at the bottom by default (that usually
locks in the worst price). The breaker's job is to stop *adding* risk.

Independent trip conditions (any one trips the breaker):
  1. Daily loss limit     — realised+unrealised loss today exceeds MAX_DAILY_LOSS_PCT
  2. Max drawdown         — equity falls MAX_DRAWDOWN_PCT from its high-water mark
  3. Consecutive losses   — MAX_CONSECUTIVE_LOSSES stop-outs in a row
  4. Market crash         — BTC drops MARKET_CRASH_PCT within the lookback window
                            (correlation protection — when BTC dumps, everything dumps)

States:
  ACTIVE   — trading allowed
  TRIPPED  — new entries blocked; auto-resets after COOLDOWN_HOURS, or manual reset

All thresholds are configurable in config.py. Thread-safe.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, date

import config

logger = logging.getLogger(__name__)


# ── Defaults (overridable via config) ─────────────────────────────────────────

def _cfg(name: str, default):
    return getattr(config, name, default)


@dataclass
class BreakerState:
    status:           str   = "active"      # "active" | "tripped"
    reason:           str   = ""
    tripped_at:       str   = ""            # ISO timestamp
    high_water_mark:  float = 0.0           # peak equity seen
    day:              str   = ""            # UTC date for daily-loss tracking
    day_start_equity: float = 0.0           # equity at start of UTC day
    consecutive_losses: int = 0
    trips_today:      int   = 0


class CircuitBreaker:
    def __init__(self):
        self._lock  = threading.Lock()
        self._state = BreakerState()
        self._initialised = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def initialise(self, equity: float) -> None:
        """Call once at startup with the starting capital."""
        with self._lock:
            today = date.today().isoformat()
            self._state.high_water_mark  = equity
            self._state.day              = today
            self._state.day_start_equity = equity
            self._initialised = True
        logger.info("Circuit breaker armed | equity=%.2f | daily_limit=%.1f%% "
                    "drawdown_limit=%.1f%% consec_losses=%d crash=%.1f%%",
                    equity, _cfg("CB_MAX_DAILY_LOSS_PCT", 6.0),
                    _cfg("CB_MAX_DRAWDOWN_PCT", 15.0),
                    _cfg("CB_MAX_CONSECUTIVE_LOSSES", 5),
                    _cfg("CB_MARKET_CRASH_PCT", 8.0))

    # ── The gate the bot checks before opening a trade ────────────────────────

    def can_trade(self) -> tuple[bool, str]:
        """Return (allowed, reason). Auto-resets if cooldown elapsed."""
        if not _cfg("CB_ENABLED", True):
            return True, "breaker disabled"
        with self._lock:
            if self._state.status == "tripped":
                # Auto-reset after cooldown
                if self._cooldown_elapsed():
                    self._reset_locked("cooldown elapsed")
                    return True, "reset after cooldown"
                return False, f"HALTED: {self._state.reason}"
            return True, "ok"

    # ── Update — call each scan cycle with current portfolio + market state ───

    def update(self, equity: float, open_unrealised: float,
               btc_change_pct: float | None = None) -> None:
        """
        Re-evaluate all trip conditions.
          equity          : capital + realised P&L (NOT counting open positions)
          open_unrealised : sum of unrealised P&L across open positions
          btc_change_pct  : BTC % change over the crash lookback window (optional)
        """
        if not _cfg("CB_ENABLED", True) or not self._initialised:
            return

        with self._lock:
            # Roll the daily window at UTC midnight
            today = date.today().isoformat()
            if today != self._state.day:
                self._state.day              = today
                self._state.day_start_equity = equity
                self._state.trips_today      = 0

            total_equity = equity + open_unrealised

            # Update high-water mark
            if total_equity > self._state.high_water_mark:
                self._state.high_water_mark = total_equity

            if self._state.status == "tripped":
                if self._cooldown_elapsed():
                    self._reset_locked("cooldown elapsed")
                else:
                    return   # stay tripped

            # ── Evaluate trip conditions ──────────────────────────────────────
            # 1. Daily loss
            daily_limit = _cfg("CB_MAX_DAILY_LOSS_PCT", 6.0)
            day_start   = self._state.day_start_equity or 1e-9
            daily_pl_pct = (total_equity - day_start) / day_start * 100
            if daily_pl_pct <= -daily_limit:
                self._trip_locked(f"daily loss {daily_pl_pct:.1f}% ≤ -{daily_limit:.1f}%")
                return

            # 2. Max drawdown from peak
            dd_limit = _cfg("CB_MAX_DRAWDOWN_PCT", 15.0)
            hwm      = self._state.high_water_mark or 1e-9
            drawdown = (total_equity - hwm) / hwm * 100
            if drawdown <= -dd_limit:
                self._trip_locked(f"drawdown {drawdown:.1f}% ≤ -{dd_limit:.1f}% from peak")
                return

            # 3. Consecutive losses
            max_consec = _cfg("CB_MAX_CONSECUTIVE_LOSSES", 5)
            if self._state.consecutive_losses >= max_consec:
                self._trip_locked(f"{self._state.consecutive_losses} consecutive losses")
                return

            # 4. Market crash (BTC correlation protection)
            crash_limit = _cfg("CB_MARKET_CRASH_PCT", 8.0)
            if btc_change_pct is not None and btc_change_pct <= -crash_limit:
                self._trip_locked(f"market crash: BTC {btc_change_pct:.1f}% ≤ -{crash_limit:.1f}%")
                return

    # ── Trade outcome feed ────────────────────────────────────────────────────

    def record_trade_result(self, pnl: float) -> None:
        """Feed each closed trade so consecutive-loss tracking stays accurate."""
        with self._lock:
            if pnl < 0:
                self._state.consecutive_losses += 1
            else:
                self._state.consecutive_losses = 0

    # ── Manual control ─────────────────────────────────────────────────────────

    def manual_reset(self) -> None:
        with self._lock:
            self._reset_locked("manual reset")

    def manual_trip(self, reason: str = "manual halt") -> None:
        with self._lock:
            self._trip_locked(reason)

    # ── Internal (must hold lock) ──────────────────────────────────────────────

    def _trip_locked(self, reason: str) -> None:
        if self._state.status == "tripped":
            return
        self._state.status     = "tripped"
        self._state.reason     = reason
        self._state.tripped_at = datetime.now(timezone.utc).isoformat()
        self._state.trips_today += 1
        logger.error("🛑 CIRCUIT BREAKER TRIPPED — new entries halted. Reason: %s", reason)
        # Fire Telegram alert (non-blocking)
        try:
            from notifications.notifier import send_telegram
            cooldown = _cfg("CB_COOLDOWN_HOURS", 6)
            send_telegram(
                f"🛑 *CIRCUIT BREAKER TRIPPED*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"New trades halted.\n"
                f"*Reason:* {reason}\n"
                f"Open positions keep their stops.\n"
                f"Auto-resets in {cooldown}h (or reset from dashboard)."
            )
        except Exception:
            pass

    def _reset_locked(self, why: str) -> None:
        was_tripped = self._state.status == "tripped"
        self._state.status            = "active"
        self._state.reason            = ""
        self._state.tripped_at        = ""
        self._state.consecutive_losses = 0
        if was_tripped:
            logger.warning("✅ Circuit breaker RESET (%s) — trading resumed", why)
            try:
                from notifications.notifier import send_telegram
                send_telegram(f"✅ *Circuit breaker reset* ({why}) — trading resumed.")
            except Exception:
                pass

    def _cooldown_elapsed(self) -> bool:
        if not self._state.tripped_at:
            return True
        try:
            tripped = datetime.fromisoformat(self._state.tripped_at)
            hours = (datetime.now(timezone.utc) - tripped).total_seconds() / 3600
            return hours >= _cfg("CB_COOLDOWN_HOURS", 6)
        except Exception:
            return True

    # ── Introspection for dashboard ───────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            s = self._state
            hwm = s.high_water_mark or 1e-9
            return {
                "enabled":            _cfg("CB_ENABLED", True),
                "status":             s.status,
                "reason":             s.reason,
                "tripped_at":         s.tripped_at,
                "high_water_mark":    round(s.high_water_mark, 2),
                "consecutive_losses": s.consecutive_losses,
                "trips_today":        s.trips_today,
                "cooldown_hours":     _cfg("CB_COOLDOWN_HOURS", 6),
                "limits": {
                    "daily_loss_pct":     _cfg("CB_MAX_DAILY_LOSS_PCT", 6.0),
                    "max_drawdown_pct":   _cfg("CB_MAX_DRAWDOWN_PCT", 15.0),
                    "consecutive_losses": _cfg("CB_MAX_CONSECUTIVE_LOSSES", 5),
                    "market_crash_pct":   _cfg("CB_MARKET_CRASH_PCT", 8.0),
                },
            }


# ── Singleton ─────────────────────────────────────────────────────────────────

_breaker: CircuitBreaker | None = None

def get_breaker() -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker()
    return _breaker
