"""
Persist open positions to disk so bot restarts don't lose track.
Saves to positions.json every time a position is opened or closed.
"""
import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
# Resolve to the PROJECT ROOT (parent of the risk/ package dir), so the state
# file is predictable regardless of where this module lives after refactors.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FILE = os.path.join(_PROJECT_ROOT, "positions.json")

# One-time migration: if a stale file exists in the old risk/ location, move it.
_OLD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.json")
if os.path.exists(_OLD_FILE) and not os.path.exists(_FILE):
    try:
        os.replace(_OLD_FILE, _FILE)
    except Exception:
        pass


def save(positions: dict) -> None:
    try:
        data = {}
        for sym, pos in positions.items():
            if pos.status == "open":
                data[sym] = {
                    "symbol":       pos.symbol,
                    "side":         pos.side,
                    "entry_price":  pos.entry_price,
                    "qty":          pos.qty,
                    "stop_loss":    pos.stop_loss,
                    "take_profit":  pos.take_profit,
                    "atr_at_entry": pos.atr_at_entry,
                    "order_id":     pos.order_id,
                    "opened_at":    pos.opened_at.isoformat(),
                    "highest_price":pos.highest_price,
                    "lowest_price": pos.lowest_price,
                    "stop_order_id":pos.stop_order_id,
                    "tp_order_id":  pos.tp_order_id,
                    "protected":    pos.protected,
                }
        with open(_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning("Could not save positions: %s", e)


def load() -> dict:
    if not os.path.exists(_FILE):
        return {}
    try:
        with open(_FILE) as f:
            data = json.load(f)
        logger.info("Loaded %d persisted positions from disk", len(data))
        return data
    except Exception as e:
        logger.warning("Could not load positions: %s", e)
        return {}


def clear() -> None:
    if os.path.exists(_FILE):
        os.remove(_FILE)
