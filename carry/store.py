"""
Atomic state persistence for the carry harvester.

State survives restarts: books (open carry positions + cumulative funding/fees),
the start time, and a config snapshot. Writes are atomic (tmp + fsync + os.replace)
so a crash mid-write can never corrupt carry_state.json — the old state stays intact
until the new one is fully on disk. Mirrors sma_bot.py's persistence.
"""
import os
import json
import time
import logging
import config
from carry.engine import new_book

logger = logging.getLogger("carry.store")
STATE_FILE = config.CARRY_STATE_FILE


def load() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            st = json.load(open(STATE_FILE))
            if "books" in st:
                return st
        except Exception as e:
            logger.warning("Could not read %s (%s) — starting fresh", STATE_FILE, e)
    return {"books": {}, "started_ms": int(time.time() * 1000),
            "last_cycle": None, "cycles": 0}


def ensure_books(state: dict, tokens: list[str]) -> None:
    """Make sure every configured token has a book; leave existing ones intact."""
    for t in tokens:
        state["books"].setdefault(t, new_book(t))


def save(state: dict) -> None:
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        logger.warning("Could not save state: %s", e)
