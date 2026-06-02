"""
Candlestick pattern detection — 23 patterns.

All functions receive the last N rows of an enriched OHLCV DataFrame
and return a score in [-3, +3]:
  +3  = very strong bullish signal
  +2  = strong bullish
  +1  = mild bullish
   0  = neutral / no pattern
  -1  = mild bearish
  -2  = strong bearish
  -3  = very strong bearish signal

The master function `score_candle_patterns(df)` runs all detectors
and returns the combined score (capped at [-3, +3]).

Patterns implemented:
  Reversal (bullish) : Hammer, Inverted Hammer, Bullish Engulfing,
                       Morning Star, Piercing Line, Bullish Harami,
                       Tweezer Bottom
  Reversal (bearish) : Shooting Star, Hanging Man, Bearish Engulfing,
                       Evening Star, Dark Cloud Cover, Bearish Harami,
                       Tweezer Top
  Continuation (bull): Three White Soldiers, Rising Three Methods,
                       Three Inside Up, Green Marubozu
  Continuation (bear): Three Black Crows, Falling Three Methods,
                       Three Inside Down, Red Marubozu
  Indecision         : Doji, Spinning Top
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _c(df: pd.DataFrame, i: int = 0) -> dict:
    """Return OHLCV dict for row at position i from the end (0 = latest)."""
    row   = df.iloc[-(i + 1)]
    o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]
    body  = abs(c - o)
    rng   = h - l if (h - l) > 0 else 1e-9
    upper = h - max(c, o)
    lower = min(c, o) - l
    return {
        "o": o, "h": h, "l": l, "c": c, "v": v,
        "body": body, "range": rng,
        "upper_wick": upper, "lower_wick": lower,
        "is_bull": c >= o, "is_bear": c < o,
        "body_pct": body / rng,          # body as % of total range
    }


def _avg_body(df: pd.DataFrame, n: int = 10) -> float:
    """Average candle body size over last n candles."""
    tail = df.tail(n)
    return (tail["close"] - tail["open"]).abs().mean()


def _prior_trend(df: pd.DataFrame, n: int = 5) -> str:
    """Rough prior trend: 'up', 'down', or 'flat'."""
    if len(df) < n + 1:
        return "flat"
    start = df["close"].iloc[-(n + 1)]
    end   = df["close"].iloc[-2]          # candle before the current one
    chg   = (end - start) / start * 100
    if chg >  1.5: return "up"
    if chg < -1.5: return "down"
    return "flat"


# ══════════════════════════════════════════════════════════════════════════════
#  REVERSAL — BULLISH
# ══════════════════════════════════════════════════════════════════════════════

def detect_hammer(df: pd.DataFrame) -> float:
    """
    Hammer: small body near the top, long lower wick (≥2× body),
    tiny upper wick. After a downtrend → bullish reversal.
    """
    if len(df) < 2: return 0.0
    c0 = _c(df, 0)
    if c0["body"] == 0: return 0.0
    if c0["lower_wick"] < 2 * c0["body"]: return 0.0
    if c0["upper_wick"] > c0["body"]: return 0.0          # upper wick must be tiny
    if c0["body_pct"] > 0.35: return 0.0                  # body must be small
    return 2.0 if _prior_trend(df) == "down" else 1.0


def detect_inverted_hammer(df: pd.DataFrame) -> float:
    """
    Inverted Hammer: small body near the bottom, long upper wick (≥2× body).
    After downtrend → potential bullish reversal.
    """
    if len(df) < 2: return 0.0
    c0 = _c(df, 0)
    if c0["body"] == 0: return 0.0
    if c0["upper_wick"] < 2 * c0["body"]: return 0.0
    if c0["lower_wick"] > c0["body"]: return 0.0
    if c0["body_pct"] > 0.35: return 0.0
    return 1.5 if _prior_trend(df) == "down" else 0.5


def detect_bullish_engulfing(df: pd.DataFrame) -> float:
    """
    Bullish Engulfing: large green candle completely engulfs prior red candle.
    """
    if len(df) < 3: return 0.0
    c0, c1 = _c(df, 0), _c(df, 1)
    if not (c0["is_bull"] and c1["is_bear"]): return 0.0
    if not (c0["o"] < c1["c"] and c0["c"] > c1["o"]): return 0.0
    if c0["body"] < c1["body"]: return 0.0
    return 2.5 if _prior_trend(df) == "down" else 1.5


def detect_morning_star(df: pd.DataFrame) -> float:
    """
    Morning Star (3-candle):
      Day 1: large bearish candle
      Day 2: small body (doji-like) — uncertainty
      Day 3: large bullish candle closing above Day 1 midpoint
    """
    if len(df) < 4: return 0.0
    c0, c1, c2 = _c(df, 0), _c(df, 1), _c(df, 2)
    avg = _avg_body(df)
    if not (c2["is_bear"] and c2["body"] > avg * 0.6): return 0.0   # day1 big bear
    if c1["body_pct"] > 0.40: return 0.0                             # day2 small
    if not (c0["is_bull"] and c0["body"] > avg * 0.6): return 0.0   # day3 big bull
    midpoint = (c2["o"] + c2["c"]) / 2
    if c0["c"] < midpoint: return 0.0                                # must close above mid
    return 3.0 if _prior_trend(df) == "down" else 2.0


def detect_piercing_line(df: pd.DataFrame) -> float:
    """
    Piercing Line: bearish candle followed by bullish candle that opens below
    the prior low but closes above the prior candle's midpoint.
    """
    if len(df) < 3: return 0.0
    c0, c1 = _c(df, 0), _c(df, 1)
    if not (c1["is_bear"] and c0["is_bull"]): return 0.0
    if c0["o"] >= c1["c"]: return 0.0                               # must gap down
    midpoint = (c1["o"] + c1["c"]) / 2
    if c0["c"] <= midpoint: return 0.0                              # must close above mid
    if c0["c"] >= c1["o"]: return 0.0                              # but NOT above prior open
    return 2.0 if _prior_trend(df) == "down" else 1.0


def detect_bullish_harami(df: pd.DataFrame) -> float:
    """
    Bullish Harami: large bearish candle followed by small bullish candle
    contained within the prior body.
    """
    if len(df) < 3: return 0.0
    c0, c1 = _c(df, 0), _c(df, 1)
    avg = _avg_body(df)
    if not (c1["is_bear"] and c0["is_bull"]): return 0.0
    if c1["body"] < avg * 0.8: return 0.0                           # prior must be large
    if not (c0["o"] >= c1["c"] and c0["c"] <= c1["o"]): return 0.0 # current contained (≥ not >)
    if c0["body"] > c1["body"] * 0.6: return 0.0                   # current must be small
    return 1.5 if _prior_trend(df) == "down" else 0.5


def detect_tweezer_bottom(df: pd.DataFrame) -> float:
    """
    Tweezer Bottom: two candles with nearly identical lows (within 0.1% of each other).
    First bearish, second bullish — strong support level.
    """
    if len(df) < 3: return 0.0
    c0, c1 = _c(df, 0), _c(df, 1)
    if not (c1["is_bear"] and c0["is_bull"]): return 0.0
    if abs(c0["l"] - c1["l"]) / c1["l"] > 0.001: return 0.0       # lows within 0.1%
    return 2.0 if _prior_trend(df) == "down" else 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  REVERSAL — BEARISH
# ══════════════════════════════════════════════════════════════════════════════

def detect_shooting_star(df: pd.DataFrame) -> float:
    """
    Shooting Star: small body near the bottom, long upper wick (≥2× body).
    After an uptrend → bearish reversal.
    """
    if len(df) < 2: return 0.0
    c0 = _c(df, 0)
    if c0["body"] == 0: return 0.0
    if c0["upper_wick"] < 2 * c0["body"]: return 0.0
    if c0["lower_wick"] > c0["body"]: return 0.0
    if c0["body_pct"] > 0.35: return 0.0
    return -2.0 if _prior_trend(df) == "up" else -1.0


def detect_hanging_man(df: pd.DataFrame) -> float:
    """
    Hanging Man: looks like a Hammer but appears after an uptrend → bearish warning.
    Small body near top, long lower wick.
    """
    if len(df) < 2: return 0.0
    c0 = _c(df, 0)
    if c0["body"] == 0: return 0.0
    if c0["lower_wick"] < 2 * c0["body"]: return 0.0
    if c0["upper_wick"] > c0["body"]: return 0.0
    if c0["body_pct"] > 0.35: return 0.0
    return -1.5 if _prior_trend(df) == "up" else -0.5


def detect_bearish_engulfing(df: pd.DataFrame) -> float:
    """
    Bearish Engulfing: large red candle completely engulfs prior green candle.
    """
    if len(df) < 3: return 0.0
    c0, c1 = _c(df, 0), _c(df, 1)
    if not (c0["is_bear"] and c1["is_bull"]): return 0.0
    if not (c0["o"] > c1["c"] and c0["c"] < c1["o"]): return 0.0
    if c0["body"] < c1["body"]: return 0.0
    return -2.5 if _prior_trend(df) == "up" else -1.5


def detect_evening_star(df: pd.DataFrame) -> float:
    """
    Evening Star (3-candle — mirror of Morning Star):
      Day 1: large bullish candle
      Day 2: small body — uncertainty
      Day 3: large bearish candle closing below Day 1 midpoint
    """
    if len(df) < 4: return 0.0
    c0, c1, c2 = _c(df, 0), _c(df, 1), _c(df, 2)
    avg = _avg_body(df)
    if not (c2["is_bull"] and c2["body"] > avg * 0.6): return 0.0
    if c1["body_pct"] > 0.40: return 0.0
    if not (c0["is_bear"] and c0["body"] > avg * 0.6): return 0.0
    midpoint = (c2["o"] + c2["c"]) / 2
    if c0["c"] > midpoint: return 0.0
    return -3.0 if _prior_trend(df) == "up" else -2.0


def detect_dark_cloud_cover(df: pd.DataFrame) -> float:
    """
    Dark Cloud Cover: after an uptrend, green candle followed by red candle
    opening above prior high but closing below prior midpoint.
    """
    if len(df) < 3: return 0.0
    c0, c1 = _c(df, 0), _c(df, 1)
    if not (c1["is_bull"] and c0["is_bear"]): return 0.0
    if c0["o"] <= c1["c"]: return 0.0                              # gap up required
    midpoint = (c1["o"] + c1["c"]) / 2
    if c0["c"] >= midpoint: return 0.0                             # must close below mid
    if c0["c"] <= c1["o"]: return 0.0                             # but NOT below prior open
    return -2.0 if _prior_trend(df) == "up" else -1.0


def detect_bearish_harami(df: pd.DataFrame) -> float:
    """
    Bearish Harami: large bullish candle followed by small bearish candle
    contained within the prior body.
    """
    if len(df) < 3: return 0.0
    c0, c1 = _c(df, 0), _c(df, 1)
    avg = _avg_body(df)
    if not (c1["is_bull"] and c0["is_bear"]): return 0.0
    if c1["body"] < avg * 0.8: return 0.0
    if not (c0["o"] <= c1["c"] and c0["c"] >= c1["o"]): return 0.0  # ≤ not <
    if c0["body"] > c1["body"] * 0.6: return 0.0
    return -1.5 if _prior_trend(df) == "up" else -0.5


def detect_tweezer_top(df: pd.DataFrame) -> float:
    """
    Tweezer Top: two candles with nearly identical highs (within 0.1%).
    First bullish, second bearish — strong resistance level.
    """
    if len(df) < 3: return 0.0
    c0, c1 = _c(df, 0), _c(df, 1)
    if not (c1["is_bull"] and c0["is_bear"]): return 0.0
    if abs(c0["h"] - c1["h"]) / c1["h"] > 0.001: return 0.0
    return -2.0 if _prior_trend(df) == "up" else -1.0


# ══════════════════════════════════════════════════════════════════════════════
#  CONTINUATION — BULLISH
# ══════════════════════════════════════════════════════════════════════════════

def detect_three_white_soldiers(df: pd.DataFrame) -> float:
    """
    Three White Soldiers: 3 consecutive large bullish candles,
    each closing higher than the previous, with small upper wicks.
    """
    if len(df) < 4: return 0.0
    c0, c1, c2 = _c(df, 0), _c(df, 1), _c(df, 2)
    avg = _avg_body(df)
    for cx in [c0, c1, c2]:
        if not cx["is_bull"]: return 0.0
        if cx["body"] < avg * 0.6: return 0.0               # each must be sizeable
        if cx["upper_wick"] > cx["body"] * 0.4: return 0.0  # small upper wicks
    if not (c0["c"] > c1["c"] > c2["c"]): return 0.0        # each close higher
    # Each candle opens within the prior candle's range (not too gappy)
    if c0["o"] > c1["h"] or c1["o"] > c2["h"]: return 0.0
    return 3.0


def detect_green_marubozu(df: pd.DataFrame) -> float:
    """
    Green Marubozu: large bullish candle with no (or minimal) wicks.
    Pure buying pressure — very strong bullish signal.
    """
    if len(df) < 2: return 0.0
    c0 = _c(df, 0)
    avg = _avg_body(df)
    if not c0["is_bull"]: return 0.0
    if c0["body"] < avg * 1.2: return 0.0               # must be larger than average
    if c0["upper_wick"] > c0["body"] * 0.10: return 0.0  # ≤10% wick allowed
    if c0["lower_wick"] > c0["body"] * 0.10: return 0.0
    return 2.5


def detect_rising_three_methods(df: pd.DataFrame) -> float:
    """
    Rising Three Methods: large bullish candle, 3 small bearish candles
    (staying within range of first candle), then another large bullish candle.
    """
    if len(df) < 6: return 0.0
    c0, c1, c2, c3, c4 = _c(df,0), _c(df,1), _c(df,2), _c(df,3), _c(df,4)
    avg = _avg_body(df)
    if not (c4["is_bull"] and c4["body"] > avg): return 0.0   # first big bull
    if not (c1["is_bear"] and c2["is_bear"] and c3["is_bear"]): return 0.0
    for cx in [c1, c2, c3]:
        if cx["h"] > c4["h"] or cx["l"] < c4["l"]: return 0.0  # stay within range
        if cx["body"] > avg * 0.6: return 0.0                    # must be small
    if not (c0["is_bull"] and c0["body"] > avg): return 0.0    # final big bull
    if c0["c"] <= c4["c"]: return 0.0                          # must close above first candle
    return 2.5


def detect_three_inside_up(df: pd.DataFrame) -> float:
    """
    Three Inside Up: bearish candle, small bullish candle inside its body (Harami),
    then a third bullish candle closing above the first candle's open.
    """
    if len(df) < 4: return 0.0
    c0, c1, c2 = _c(df, 0), _c(df, 1), _c(df, 2)
    if not c2["is_bear"]: return 0.0
    if not (c1["is_bull"] and c1["o"] >= c2["c"] and c1["c"] <= c2["o"]): return 0.0
    if not (c0["is_bull"] and c0["c"] > c2["o"]): return 0.0
    return 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  CONTINUATION — BEARISH
# ══════════════════════════════════════════════════════════════════════════════

def detect_three_black_crows(df: pd.DataFrame) -> float:
    """
    Three Black Crows: 3 consecutive large bearish candles,
    each opening within the prior body and closing near its low.
    """
    if len(df) < 4: return 0.0
    c0, c1, c2 = _c(df, 0), _c(df, 1), _c(df, 2)
    avg = _avg_body(df)
    for cx in [c0, c1, c2]:
        if not cx["is_bear"]: return 0.0
        if cx["body"] < avg * 0.7: return 0.0
        if cx["lower_wick"] > cx["body"] * 0.3: return 0.0
    if not (c0["c"] < c1["c"] < c2["c"]): return 0.0
    return -3.0


def detect_red_marubozu(df: pd.DataFrame) -> float:
    """
    Red Marubozu: large bearish candle with no (or minimal) wicks.
    Pure selling pressure — very strong bearish signal.
    """
    if len(df) < 2: return 0.0
    c0 = _c(df, 0)
    avg = _avg_body(df)
    if not c0["is_bear"]: return 0.0
    if c0["body"] < avg * 1.2: return 0.0
    if c0["upper_wick"] > c0["body"] * 0.10: return 0.0  # ≤10% wick allowed
    if c0["lower_wick"] > c0["body"] * 0.10: return 0.0
    return -2.5


def detect_falling_three_methods(df: pd.DataFrame) -> float:
    """
    Falling Three Methods: mirror of Rising Three Methods.
    Large bearish candle, 3 small bullish candles within range, then large bearish.
    """
    if len(df) < 6: return 0.0
    c0, c1, c2, c3, c4 = _c(df,0), _c(df,1), _c(df,2), _c(df,3), _c(df,4)
    avg = _avg_body(df)
    if not (c4["is_bear"] and c4["body"] > avg): return 0.0
    if not (c1["is_bull"] and c2["is_bull"] and c3["is_bull"]): return 0.0
    for cx in [c1, c2, c3]:
        if cx["h"] > c4["h"] or cx["l"] < c4["l"]: return 0.0
        if cx["body"] > avg * 0.6: return 0.0
    if not (c0["is_bear"] and c0["body"] > avg): return 0.0
    if c0["c"] >= c4["c"]: return 0.0
    return -2.5


def detect_three_inside_down(df: pd.DataFrame) -> float:
    """
    Three Inside Down: bullish candle, small bearish Harami inside it,
    then a third bearish candle closing below the first candle's open.
    """
    if len(df) < 4: return 0.0
    c0, c1, c2 = _c(df, 0), _c(df, 1), _c(df, 2)
    if not c2["is_bull"]: return 0.0
    if not (c1["is_bear"] and c1["o"] <= c2["c"] and c1["c"] >= c2["o"]): return 0.0
    if not (c0["is_bear"] and c0["c"] < c2["o"]): return 0.0
    return -2.0


# ══════════════════════════════════════════════════════════════════════════════
#  INDECISION
# ══════════════════════════════════════════════════════════════════════════════

def detect_doji(df: pd.DataFrame) -> float:
    """
    Doji: open ≈ close (body < 10% of range).
    After a trend = potential reversal coming.
    Score direction based on prior trend: bearish doji after uptrend, bullish after down.
    """
    if len(df) < 2: return 0.0
    c0 = _c(df, 0)
    if c0["body_pct"] > 0.10: return 0.0
    trend = _prior_trend(df)
    if trend == "up":   return -1.0   # exhaustion at the top
    if trend == "down": return 1.0    # exhaustion at the bottom
    return 0.0


def detect_spinning_top(df: pd.DataFrame) -> float:
    """
    Spinning Top: small body (10-30% of range) with long wicks on both sides.
    Indecision — especially meaningful after a strong trend.
    """
    if len(df) < 2: return 0.0
    c0 = _c(df, 0)
    if not (0.10 < c0["body_pct"] < 0.35): return 0.0
    if c0["upper_wick"] < c0["body"] * 0.7: return 0.0
    if c0["lower_wick"] < c0["body"] * 0.7: return 0.0
    trend = _prior_trend(df)
    if trend == "up":   return -0.5
    if trend == "down": return 0.5
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER SCORER
# ══════════════════════════════════════════════════════════════════════════════

_DETECTORS = [
    # Bullish reversal
    detect_hammer,
    detect_inverted_hammer,
    detect_bullish_engulfing,
    detect_morning_star,
    detect_piercing_line,
    detect_bullish_harami,
    detect_tweezer_bottom,
    # Bearish reversal
    detect_shooting_star,
    detect_hanging_man,
    detect_bearish_engulfing,
    detect_evening_star,
    detect_dark_cloud_cover,
    detect_bearish_harami,
    detect_tweezer_top,
    # Bullish continuation
    detect_three_white_soldiers,
    detect_green_marubozu,
    detect_rising_three_methods,
    detect_three_inside_up,
    # Bearish continuation
    detect_three_black_crows,
    detect_red_marubozu,
    detect_falling_three_methods,
    detect_three_inside_down,
    # Indecision
    detect_doji,
    detect_spinning_top,
]


def score_candle_patterns(df: pd.DataFrame) -> tuple[float, dict]:
    """
    Run all 23 detectors. Return:
      (combined_score, detail_dict)
    combined_score is capped at [-3, +3].
    detail_dict maps pattern name → score for the dashboard.
    """
    if df is None or len(df) < 6:
        return 0.0, {}

    detail: dict[str, float] = {}
    total = 0.0

    for fn in _DETECTORS:
        try:
            score = fn(df)
        except Exception:
            score = 0.0
        if score != 0.0:
            detail[fn.__name__.replace("detect_", "")] = score
            total += score

    # Cap at ±3 so one mega-pattern can't dominate everything
    return round(max(-3.0, min(3.0, total)), 2), detail


def active_patterns(df: pd.DataFrame) -> list[str]:
    """Return human-readable list of currently active patterns (non-zero)."""
    _, detail = score_candle_patterns(df)
    names = {
        "hammer":               "🔨 Hammer",
        "inverted_hammer":      "🔨 Inverted Hammer",
        "bullish_engulfing":    "🟢 Bullish Engulfing",
        "morning_star":         "⭐ Morning Star",
        "piercing_line":        "📈 Piercing Line",
        "bullish_harami":       "🟢 Bullish Harami",
        "tweezer_bottom":       "🟢 Tweezer Bottom",
        "shooting_star":        "🌠 Shooting Star",
        "hanging_man":          "🪢 Hanging Man",
        "bearish_engulfing":    "🔴 Bearish Engulfing",
        "evening_star":         "🌙 Evening Star",
        "dark_cloud_cover":     "☁️  Dark Cloud Cover",
        "bearish_harami":       "🔴 Bearish Harami",
        "tweezer_top":          "🔴 Tweezer Top",
        "three_white_soldiers": "⬆️  Three White Soldiers",
        "green_marubozu":       "💚 Green Marubozu",
        "rising_three_methods": "📊 Rising Three Methods",
        "three_inside_up":      "🟢 Three Inside Up",
        "three_black_crows":    "⬇️  Three Black Crows",
        "red_marubozu":         "❤️  Red Marubozu",
        "falling_three_methods":"📉 Falling Three Methods",
        "three_inside_down":    "🔴 Three Inside Down",
        "doji":                 "➕ Doji",
        "spinning_top":         "🔄 Spinning Top",
    }
    return [names.get(k, k) for k in detail]
