"""
Position sizing, portfolio risk management, and stop/target tracking.

Uses a fixed-fractional Kelly-inspired sizing model:
  position_size = (risk_capital * risk_pct) / (entry - stop_loss)

Tracks open positions and enforces:
  - Max concurrent open positions
  - Max total portfolio risk at once
  - Per-trade max risk
  - Trailing stop updates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    side: str           # "long" | "short"
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    atr_at_entry: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: str = ""
    highest_price: float = 0.0   # for trailing stop (long)
    lowest_price: float  = 0.0   # for trailing stop (short)
    unrealized_pnl: float = 0.0
    status: str = "open"   # open | closed


class RiskManager:
    def __init__(self):
        self.positions: dict[str, Position] = {}   # symbol → Position
        self._total_capital = config.TOTAL_CAPITAL_USDT
        self._restore_positions()   # reload from disk on startup

    def _restore_positions(self) -> None:
        """Reload any open positions saved before a restart."""
        import risk.position_store as position_store
        data = position_store.load()
        for sym, d in data.items():
            try:
                opened_at = datetime.fromisoformat(d["opened_at"])
            except Exception:
                opened_at = datetime.now(timezone.utc)
            pos = Position(
                symbol=d["symbol"], side=d["side"],
                entry_price=d["entry_price"], qty=d["qty"],
                stop_loss=d["stop_loss"], take_profit=d["take_profit"],
                atr_at_entry=d["atr_at_entry"], order_id=d.get("order_id",""),
                opened_at=opened_at,
                highest_price=d.get("highest_price", d["entry_price"]),
                lowest_price=d.get("lowest_price",  d["entry_price"]),
            )
            self.positions[sym] = pos
            logger.info("Restored position: %s %s @ %.6f", d["side"].upper(), sym, d["entry_price"])

    # ── Sizing ────────────────────────────────────────────────────────────────

    def position_size_usdt(self, entry: float, stop: float, score: float) -> float:
        """
        Return USDT amount to deploy.
        Scales between 1× and 1.5× based on signal strength.
        """
        risk_usdt = self._total_capital * (config.RISK_PER_TRADE_PCT / 100)
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return 0.0
        base_qty   = risk_usdt / risk_per_unit       # units
        position_value = base_qty * entry

        # scale up for strong signals, cap at 1.5×
        multiplier = 1.0
        if score >= config.STRONG_SIGNAL_SCORE:
            multiplier = 1.5
        elif score >= config.MIN_SIGNAL_SCORE + 1:
            multiplier = 1.25
        position_value *= multiplier

        # hard cap: single position ≤ 20% of capital
        max_position = self._total_capital * 0.20
        return min(position_value, max_position)

    # ── Portfolio checks ──────────────────────────────────────────────────────

    def can_open(self, symbol: str) -> tuple[bool, str]:
        """Check if a new position can be opened."""
        open_positions = [p for p in self.positions.values() if p.status == "open"]

        if symbol in self.positions and self.positions[symbol].status == "open":
            return False, f"already have open position in {symbol}"

        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            return False, f"max open positions ({config.MAX_OPEN_POSITIONS}) reached"

        total_risk_pct = len(open_positions) * config.RISK_PER_TRADE_PCT
        if total_risk_pct >= config.MAX_PORTFOLIO_RISK_PCT:
            return False, f"max portfolio risk ({config.MAX_PORTFOLIO_RISK_PCT}%) reached"

        return True, "ok"

    # ── Position lifecycle ────────────────────────────────────────────────────

    def open_position(self, symbol: str, side: str, entry: float,
                      qty: float, stop: float, take_profit: float,
                      atr: float, order_id: str = "") -> Position:
        pos = Position(
            symbol=symbol, side=side, entry_price=entry,
            qty=qty, stop_loss=stop, take_profit=take_profit,
            atr_at_entry=atr, order_id=order_id,
            highest_price=entry, lowest_price=entry,
        )
        self.positions[symbol] = pos
        logger.info("Position opened: %s %s @ %.6f  SL=%.6f  TP=%.6f  qty=%.6f",
                    side.upper(), symbol, entry, stop, take_profit, qty)
        import risk.position_store as position_store; position_store.save(self.positions)
        return pos

    def update_position(self, symbol: str, current_price: float) -> dict:
        """Update unrealized PnL and trailing stop. Returns action dict."""
        pos = self.positions.get(symbol)
        if pos is None or pos.status != "open":
            return {"action": "none"}

        # Update extremes
        if pos.side == "long":
            pos.highest_price = max(pos.highest_price, current_price)
        else:
            pos.lowest_price = min(pos.lowest_price or current_price, current_price)

        # Unrealized PnL
        if pos.side == "long":
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.qty
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.qty

        # Check stop loss hit
        if pos.side == "long" and current_price <= pos.stop_loss:
            return {"action": "close", "reason": "stop_loss", "price": current_price}
        if pos.side == "short" and current_price >= pos.stop_loss:
            return {"action": "close", "reason": "stop_loss", "price": current_price}

        # Check take profit hit
        if pos.side == "long" and current_price >= pos.take_profit:
            return {"action": "close", "reason": "take_profit", "price": current_price}
        if pos.side == "short" and current_price <= pos.take_profit:
            return {"action": "close", "reason": "take_profit", "price": current_price}

        # Trailing stop: once price moves 1 ATR in profit direction, trail at 1.5 ATR
        trail_atr = config.TRAILING_STOP_ATR * pos.atr_at_entry
        if pos.side == "long":
            in_profit = current_price - pos.entry_price
            if in_profit > pos.atr_at_entry:   # at least 1 ATR in profit
                new_trail = pos.highest_price - trail_atr
                if new_trail > pos.stop_loss:
                    pos.stop_loss = new_trail
                    logger.info("Trailing stop raised for %s → %.6f", symbol, pos.stop_loss)
        elif pos.side == "short":
            in_profit = pos.entry_price - current_price
            if in_profit > pos.atr_at_entry:
                new_trail = pos.lowest_price + trail_atr
                if new_trail < pos.stop_loss:
                    pos.stop_loss = new_trail

        return {"action": "hold", "unrealized_pnl": pos.unrealized_pnl}

    def close_position(self, symbol: str, exit_price: float, reason: str = "manual") -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - exit_price) * pos.qty
        pos.status = "closed"
        self._total_capital += pnl   # update running capital
        logger.info("Position closed: %s @ %.6f  Reason=%s  PnL=%.4f USDT  Capital=%.2f",
                    symbol, exit_price, reason, pnl, self._total_capital)
        import risk.position_store as position_store; position_store.save(self.positions)
        return pnl

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        open_pos   = [p for p in self.positions.values() if p.status == "open"]
        closed_pos = [p for p in self.positions.values() if p.status == "closed"]
        return {
            "capital_usdt":      round(self._total_capital, 2),
            "open_positions":    len(open_pos),
            "closed_positions":  len(closed_pos),
            "open_details":      [{"symbol": p.symbol, "side": p.side,
                                   "entry": p.entry_price, "sl": p.stop_loss,
                                   "tp": p.take_profit, "pnl": round(p.unrealized_pnl, 4)}
                                  for p in open_pos],
        }