"""
Daily trend-following strategy — core/daily_strategy.py

Thesis (documented edge): ride sustained uptrends, sit in cash during downtrends.
Long-only. Operates on DAILY candles. Scans the whole liquid universe once a day,
holds the strongest-trending coins, exits anything whose trend breaks.

Signal:  in uptrend while close > SMA(TREND_MA) AND SMA(FAST_MA) > SMA(TREND_MA)
Strength: ranking score for picking the BEST trends when many qualify —
          uses risk-adjusted momentum (return / volatility) so we prefer smooth,
          strong trends over violent ones.

This module is pure functions (no I/O) so it can be used by both the live bot
and the backtest identically.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ── Parameters (tunable) ──────────────────────────────────────────────────────
TREND_MA      = 200    # long-term trend filter (days)
FAST_MA       = 50     # faster confirmation
MOMENTUM_DAYS = 90     # lookback for the strength ranking
VOL_DAYS      = 30     # volatility window for risk-adjusting the strength


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def in_uptrend(df: pd.DataFrame) -> pd.Series:
    """
    Boolean Series: True on days the coin is in a confirmed uptrend.
    close > 200-day SMA  AND  50-day SMA > 200-day SMA.
    """
    close = df["close"]
    trend = _sma(close, TREND_MA)
    fast  = _sma(close, FAST_MA)
    return (close > trend) & (fast > trend)


def trend_strength(df: pd.DataFrame) -> pd.Series:
    """
    Risk-adjusted momentum used to RANK uptrending coins.
    = (MOMENTUM_DAYS return) / (annualised volatility).
    Higher = stronger, smoother uptrend → preferred for the portfolio.
    """
    close = df["close"]
    mom   = close.pct_change(MOMENTUM_DAYS)
    vol   = close.pct_change().rolling(VOL_DAYS).std() * np.sqrt(365)
    return mom / vol.replace(0, np.nan)


def latest_signal(df: pd.DataFrame) -> dict:
    """
    Convenience for the live bot: returns the current day's signal for one coin.
    {"uptrend": bool, "strength": float, "close": float, "sma200": float}
    """
    if df is None or len(df) < TREND_MA + 5:
        return {"uptrend": False, "strength": float("-inf"), "close": 0.0, "sma200": 0.0}
    up  = bool(in_uptrend(df).iloc[-1])
    stg = trend_strength(df).iloc[-1]
    stg = float(stg) if not (isinstance(stg, float) and np.isnan(stg)) else float("-inf")
    return {
        "uptrend":  up,
        "strength": stg,
        "close":    float(df["close"].iloc[-1]),
        "sma200":   float(_sma(df["close"], TREND_MA).iloc[-1]),
    }
