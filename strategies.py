"""
Multi-strategy signal engine.

Each strategy returns a score component in [-2, +2].
Final score normalised to 0-10 for display.

Key design changes (v2):
 - 4h trend is now a HARD GATE: LONG blocked when 4h bearish, SHORT blocked when 4h bullish
 - RSI no longer rewards oversold bounces in downtrend (regime-aware)
 - Volume is required (not just a bonus) — no volume = no trade
 - Bollinger bands now distinguish bounce vs breakdown correctly
 - ADX threshold enforced harder; ranging markets = no signal
 - Market regime carries a larger weight to veto counter-trend entries
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import config
from sentiment import get_sentiment
from candle_patterns import score_candle_patterns, active_patterns

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    symbol: str
    direction: str        # "long" | "short" | "neutral"
    score: float          # 0-10
    components: dict      = field(default_factory=dict)
    entry_price: float    = 0.0
    stop_loss: float      = 0.0
    take_profit: float    = 0.0
    risk_reward: float    = 0.0
    sentiment: dict       = field(default_factory=dict)
    timeframe: str        = ""
    atr: float            = 0.0
    note: str             = ""
    candle_patterns: dict = field(default_factory=dict)  # active pattern detail


def _last(series: pd.Series, n: int = 1):
    return series.iloc[-n] if len(series) >= n else np.nan


def _safe(val):
    return val if not (isinstance(val, float) and np.isnan(val)) else 0.0


# ── Hard trend gate (4h) ──────────────────────────────────────────────────────

def trend_direction_4h(df4h: pd.DataFrame) -> str:
    """
    Classify the higher-timeframe trend as 'bull', 'bear', or 'neutral'.
    Used as a hard gate: LONG is blocked when bear, SHORT is blocked when bull.
    Falls back to reading the 1h regime when no higher-tf data is available.
    """
    if df4h is None or len(df4h) < 60:
        return "unknown"   # conservative: unknown blocks both directions

    close  = _last(df4h["close"])
    e21    = _last(df4h["ema21"])
    e50    = _last(df4h["ema50"])
    e200   = _last(df4h["ema200"])
    adx    = _last(df4h["adx"])
    dip    = _last(df4h["di_plus"])
    dim    = _last(df4h["di_minus"])
    st_dir = _last(df4h["supertrend_dir"])   # -1 = bullish, 1 = bearish

    if any(np.isnan(v) for v in [close, e21, e50, e200]):
        return "neutral"

    bull_points = 0
    bear_points = 0

    # EMA ordering
    if close > e21 > e50:     bull_points += 2
    elif close < e21 < e50:   bear_points += 2
    if e50 > e200:            bull_points += 1
    elif e50 < e200:          bear_points += 1

    # EMA21 slope — if EMA21 is falling, downgrade bull bias
    e21_prev = _last(df4h["ema21"], 3)
    if not np.isnan(e21_prev):
        if e21 < e21_prev:    bear_points += 1   # EMA21 declining
        elif e21 > e21_prev:  bull_points += 1   # EMA21 rising

    # ADX + DI direction (only counts when trend is strong)
    if not np.isnan(adx) and adx >= config.ADX_THRESHOLD:
        if not np.isnan(dip) and not np.isnan(dim):
            if dip > dim:     bull_points += 2
            else:             bear_points += 2

    # Supertrend
    if not np.isnan(st_dir):
        if st_dir == -1:      bull_points += 1
        elif st_dir == 1:     bear_points += 1

    # 7-day momentum check (10 × 6h bars ≈ 2.5 days, 28 bars ≈ 7 days)
    # If price has been falling consistently, demote bull → neutral
    close_28ago = _last(df4h["close"], 28) if len(df4h) >= 28 else np.nan
    if not np.isnan(close_28ago) and close_28ago > 0:
        roc_7d = (close - close_28ago) / close_28ago * 100
        if roc_7d < -5:    bear_points += 2   # coin down >5% in 7 days = bear pressure
        elif roc_7d > 5:   bull_points += 2   # coin up >5% in 7 days = bull pressure

    if bull_points >= 4 and bull_points > bear_points + 1:
        return "bull"
    elif bear_points >= 4 and bear_points > bull_points + 1:
        return "bear"
    return "neutral"


# ── Individual strategy scores ────────────────────────────────────────────────

def score_ema_trend(df: pd.DataFrame) -> float:
    """EMA alignment. Full stack alignment required for max score."""
    c   = _last(df["close"])
    e9  = _last(df["ema9"])
    e21 = _last(df["ema21"])
    e50 = _last(df["ema50"])
    e200= _last(df["ema200"])

    if any(np.isnan(v) for v in [c, e9, e21, e50, e200]):
        return 0.0

    # Full bearish stack — hard negative
    if e9 < e21 < e50 < e200 and c < e50:
        return -2.0

    score = 0.0
    if e9  > e21:  score += 0.5
    if e21 > e50:  score += 0.5
    if e50 > e200: score += 0.5
    if c   > e50:  score += 0.25
    if c   > e200: score += 0.25
    return min(score, 2.0)


def score_macd(df: pd.DataFrame) -> float:
    """MACD: bullish cross + expanding histogram. Both required for full score."""
    hist_now  = _last(df["macd_hist"])
    hist_prev = _last(df["macd_hist"], 2)
    hist_prev2= _last(df["macd_hist"], 3)
    macd_now  = _last(df["macd"])
    sig_now   = _last(df["macd_signal"])

    if any(np.isnan(v) for v in [hist_now, macd_now, sig_now]):
        return 0.0

    score = 0.0
    if macd_now > sig_now:                   score += 1.0
    if hist_now > 0:                         score += 0.5
    # Histogram must be expanding for 2 consecutive bars
    if not np.isnan(hist_prev) and hist_now > hist_prev > 0:
        score += 0.5
    if macd_now < sig_now:                   score -= 1.0
    if hist_now < 0:                         score -= 0.5
    if not np.isnan(hist_prev) and hist_now < hist_prev < 0:
        score -= 0.5   # expanding negative histogram
    return max(-2.0, min(score, 2.0))


def score_rsi(df: pd.DataFrame, trend: str = "neutral") -> float:
    """
    RSI — regime-aware.
    In bull trend: reward momentum (RSI 50-70), penalise if dropping below 45.
    In bear trend: reward RSI falling/staying below 50 strongly for shorts.
    """
    rsi      = _last(df["rsi"])
    rsi_prev = _last(df["rsi"], 3)
    if np.isnan(rsi):
        return 0.0

    if trend == "bull":
        if 50 <= rsi <= 70:  return 1.5   # healthy bull momentum
        if rsi > 70:         return 0.0   # overbought — avoid chasing
        if rsi < 45:         return -1.5  # losing momentum fast — stronger warning
        return 0.5

    elif trend == "bear":
        # Short signals: RSI below 50 and falling = strong bear confirmation
        if rsi < 45 and not np.isnan(rsi_prev) and rsi < rsi_prev:
            return 2.0   # RSI falling below 45 = strong short signal
        if rsi < 50:         return 1.5   # below midline = bearish
        if rsi > 55:         return -1.5  # recovering above 55 = bad for shorts
        return 0.5

    else:
        # Neutral regime: classic mean-reversion
        if rsi < config.RSI_OVERSOLD and not np.isnan(rsi_prev) and rsi > rsi_prev:
            return 1.5   # bouncing up from oversold
        if rsi < config.RSI_OVERSOLD:
            return 0.5
        if 50 <= rsi < config.RSI_OVERBOUGHT:
            return 1.0
        if rsi >= config.RSI_OVERBOUGHT:
            return -0.5
        return 0.0


def score_stochastic(df: pd.DataFrame) -> float:
    """%K/%D cross. Only strong crosses (oversold/overbought zones) score highly."""
    k      = _last(df["stoch_k"])
    d      = _last(df["stoch_d"])
    k_prev = _last(df["stoch_k"], 2)
    d_prev = _last(df["stoch_d"], 2)
    if any(np.isnan(v) for v in [k, d, k_prev, d_prev]):
        return 0.0

    just_crossed_up   = k > d and k_prev <= d_prev
    just_crossed_down = k < d and k_prev >= d_prev

    if k < 25 and just_crossed_up:    return 2.0   # golden cross deeply oversold
    if k > 75 and just_crossed_down:  return -2.0  # death cross deeply overbought
    if k < 35:                        return 0.5
    if k > 65:                        return -0.5
    return 0.0


def score_bollinger(df: pd.DataFrame, trend: str = "neutral") -> float:
    """
    Bollinger Bands — trend-aware.
    In uptrend: riding upper half of band is bullish, not a sell signal.
    In downtrend: price near upper band = resistance, near lower = continuation down.
    """
    pct   = _last(df["bb_pct"])
    width = _last(df["bb_width"])
    close = _last(df["close"])
    bb_up = _last(df["bb_up"])
    bb_mid= _last(df["bb_mid"])

    if any(np.isnan(v) for v in [pct, width]):
        return 0.0

    score = 0.0
    is_squeeze = width < df["bb_width"].rolling(50, min_periods=10).mean().iloc[-1] * 0.7

    if trend == "bull":
        if pct > 0.5:            score += 1.0   # riding upper half = momentum
        if close > bb_up:        score += 0.5   # breakout above band
        if is_squeeze:           score += 0.5   # squeeze before expansion
        if pct < 0.2:            score -= 0.5   # price dropping to lower band in uptrend = concern
    elif trend == "bear":
        if pct < 0.2:            score += 1.5   # deep in lower band = strong bear momentum
        elif pct < 0.4:          score += 1.0   # riding lower half
        bb_low_val = _last(df["bb_low"]) if "bb_low" in df.columns else np.nan
        if not np.isnan(bb_low_val) and close < bb_low_val:
            score += 0.5         # breakdown below lower band
        if is_squeeze:           score += 0.5
        if pct > 0.7:            score -= 1.0   # recovering toward upper band = bad for shorts
    else:
        if pct < 0.2:            score += 1.0   # near lower band = potential bounce
        elif 0.4 < pct < 0.7:   score += 0.5   # midline riding
        if close > bb_up:        score += 0.5
        if is_squeeze:           score += 0.5

    return max(-2.0, min(score, 2.0))


def score_adx(df: pd.DataFrame) -> float:
    """ADX trend strength. Below threshold = hard penalise (ranging market)."""
    adx = _last(df["adx"])
    dip = _last(df["di_plus"])
    dim = _last(df["di_minus"])
    if any(np.isnan(v) for v in [adx, dip, dim]):
        return 0.0

    if adx < config.ADX_THRESHOLD:
        return -1.0   # flat/ranging — stronger penalty than before

    if dip > dim:
        return 1.5 if adx > 40 else 1.0
    else:
        return -1.5 if adx > 40 else -1.0


def score_volume(df: pd.DataFrame, trend: str = "neutral") -> float:
    """
    Volume confirmation. Returns 0 if volume is below average — no volume = no trade signal.
    Rewards volume surges in the direction of the trend.
    """
    vol_ratio = _last(df["vol_ratio"])
    close     = _last(df["open"])
    open_     = _last(df["open"])
    cmf       = _last(df["cmf"])
    obv       = _last(df["obv"])
    obv_ema   = _last(df["obv_ema"])

    if np.isnan(vol_ratio):
        return 0.0

    # Volume below average = no confirmation at all
    if vol_ratio < 0.8:
        return -0.5

    score = 0.0
    green_candle = _last(df["close"]) > _last(df["open"])
    red_candle   = not green_candle

    if trend in ("bull", "neutral") and green_candle:
        if vol_ratio > config.VOLUME_SURGE_MULTIPLIER: score += 1.5
        elif vol_ratio > 1.0:                          score += 0.75
    elif trend == "bear" and red_candle:
        if vol_ratio > config.VOLUME_SURGE_MULTIPLIER: score += 1.5
        elif vol_ratio > 1.0:                          score += 0.75

    # CMF: money flow direction
    if not np.isnan(cmf):
        if trend in ("bull", "neutral") and cmf > 0.05:  score += 0.5
        elif trend == "bear" and cmf < -0.05:             score += 0.5

    # OBV above EMA = accumulation
    if not np.isnan(obv) and not np.isnan(obv_ema):
        if trend in ("bull", "neutral") and obv > obv_ema: score += 0.5
        elif trend == "bear" and obv < obv_ema:            score += 0.5

    return min(score, 2.0)


def score_supertrend(df: pd.DataFrame) -> float:
    """Supertrend flip or continuation."""
    direction      = _last(df["supertrend_dir"])
    direction_prev = _last(df["supertrend_dir"], 2)
    if np.isnan(direction):
        return 0.0
    # -1 = bullish (price above supertrend line), 1 = bearish
    if direction == -1 and direction_prev == 1:   return 2.0   # fresh flip bullish
    elif direction == -1:                          return 1.0   # ongoing bullish
    elif direction == 1 and direction_prev == -1:  return -2.0  # fresh flip bearish
    else:                                          return -1.0


def score_ichimoku(df: pd.DataFrame) -> float:
    """Ichimoku: price vs cloud + TK cross."""
    close    = _last(df["close"])
    tenkan   = _last(df["tenkan"])
    kijun    = _last(df["kijun"])
    senkou_a = _last(df["senkou_a"])
    senkou_b = _last(df["senkou_b"])
    if any(np.isnan(v) for v in [tenkan, kijun, senkou_a, senkou_b]):
        return 0.0

    cloud_top = max(senkou_a, senkou_b)
    cloud_bot = min(senkou_a, senkou_b)
    score = 0.0
    if close > cloud_top:      score += 1.0
    elif close < cloud_bot:    score -= 1.0
    # Inside cloud = uncertain, discount
    else:                      score -= 0.5
    if tenkan > kijun:         score += 0.75
    elif tenkan < kijun:       score -= 0.75
    if senkou_a > senkou_b:    score += 0.25
    return max(-2.0, min(score, 2.0))


def score_support_bounce(df: pd.DataFrame) -> float:
    """Price bounced from a pivot support level."""
    from indicators import find_pivot_levels
    try:
        _, lows = find_pivot_levels(df)
        if not lows:
            return 0.0
        close = _last(df["close"])
        atr   = _last(df["atr"])
        if np.isnan(atr) or atr == 0:
            return 0.0
        for level in sorted(lows, reverse=True):
            if abs(close - level) < atr * 0.4 and close >= level:
                return 1.5
    except Exception:
        pass
    return 0.0


def score_market_regime(df: pd.DataFrame) -> float:
    """
    Strong regime filter — bigger weight than before.
    Bear regime applies a -3 raw penalty, effectively killing most long signals.
    """
    close = _last(df["close"])
    e200  = _last(df["ema200"])
    e50   = _last(df["ema50"])
    e21   = _last(df["ema21"])
    if any(np.isnan(v) for v in [close, e200, e50, e21]):
        return 0.0

    if close > e200 and e50 > e200 and e21 > e50:
        return 1.5   # strong bull regime

    if close > e200 and e50 > e200:
        return 0.5   # moderate bull

    if close < e200 and e50 < e200 and e21 < e50:
        return -3.0  # confirmed bear regime (raised from -2 to -3)

    if close < e200:
        return -1.0  # below 200 EMA = caution
    return 0.0


# ── Main scoring function ─────────────────────────────────────────────────────

_MAX_RAW_SCORE = 25.0   # +3 for candle patterns component


def analyse(symbol: str,
            df_primary: pd.DataFrame,
            df_trend: pd.DataFrame | None = None,
            include_sentiment: bool = True) -> SignalResult:
    """
    Compute a composite signal score for `symbol`.
    df_primary = enriched 1h candles
    df_trend   = enriched 4h candles — used as hard directional gate
    """
    result = SignalResult(symbol=symbol, direction="neutral", score=0.0, timeframe=config.TF_PRIMARY)

    if df_primary is None or len(df_primary) < 60:
        result.note = "insufficient data"
        return result

    # ── Hard gate 1: higher-tf trend direction ───────────────────────────────
    trend4h = trend_direction_4h(df_trend)

    # ── Hard gate 2: 1h short-term momentum must agree with the trade ────────
    # Require EMA9 > EMA21 on 1h for LONG, EMA9 < EMA21 for SHORT.
    # Also require the last 10 candles' ROC to be positive for LONG.
    e9_now  = _last(df_primary["ema9"])
    e21_now = _last(df_primary["ema21"])
    e50_now = _last(df_primary["ema50"])
    roc_now = _last(df_primary["roc"]) if "roc" in df_primary.columns else 0.0

    short_term_bull = (not np.isnan(e9_now) and not np.isnan(e21_now)
                       and e9_now > e21_now and (np.isnan(roc_now) or roc_now > -1.0))
    short_term_bear = (not np.isnan(e9_now) and not np.isnan(e21_now)
                       and e9_now < e21_now and (np.isnan(roc_now) or roc_now < 1.0))

    result.note = f"6h:{trend4h} 1h:{'bull' if short_term_bull else 'bear' if short_term_bear else 'mix'}"

    components: dict[str, float] = {}
    components["ema_trend"]   = _safe(score_ema_trend(df_primary))
    components["macd"]        = _safe(score_macd(df_primary))
    components["rsi"]         = _safe(score_rsi(df_primary, trend4h))
    components["stochastic"]  = _safe(score_stochastic(df_primary))
    components["bollinger"]   = _safe(score_bollinger(df_primary, trend4h))
    components["adx"]         = _safe(score_adx(df_primary))
    components["volume"]      = _safe(score_volume(df_primary, trend4h))
    components["supertrend"]  = _safe(score_supertrend(df_primary))
    components["ichimoku"]    = _safe(score_ichimoku(df_primary))
    components["support"]     = _safe(score_support_bounce(df_primary))
    components["regime"]      = _safe(score_market_regime(df_primary))

    # ── Candlestick patterns (14th component) ─────────────────────────────────
    cp_score, cp_detail = score_candle_patterns(df_primary)
    components["candle_patterns"] = _safe(cp_score)
    result.candle_patterns = cp_detail

    # 4h trend contributes a weighted directional bonus
    if df_trend is not None and len(df_trend) >= 60:
        raw_4h = score_ema_trend(df_trend) + score_supertrend(df_trend)
        components["trend_4h"] = _safe(raw_4h * 0.75)
    else:
        components["trend_4h"] = 0.0

    # News sentiment
    sentiment = {"score": 0.0, "label": "neutral", "source": "none"}
    if include_sentiment:
        try:
            sentiment = get_sentiment(symbol)
            components["sentiment"] = sentiment["score"] * config.SENTIMENT_WEIGHT
        except Exception:
            components["sentiment"] = 0.0
    result.sentiment = sentiment

    # ── Aggregate raw score ───────────────────────────────────────────────────
    raw = sum(components.values())
    normalized = round((raw + _MAX_RAW_SCORE) / (2 * _MAX_RAW_SCORE) * 10, 2)
    normalized = max(0.0, min(10.0, normalized))

    result.components = components
    result.score      = normalized

    # ── Direction — score threshold + 4h hard gate ───────────────────────────
    raw_long_ok  = normalized >= config.MIN_SIGNAL_SCORE and raw > 0
    raw_short_ok = normalized <= (10 - config.MIN_SIGNAL_SCORE) and raw < 0

    # Hard gate: block trades that conflict with either the 6h trend OR 1h momentum.
    want_long  = raw_long_ok  and short_term_bull
    want_short = raw_short_ok and short_term_bear

    strong_long  = want_long  and normalized >= config.STRONG_SIGNAL_SCORE
    strong_short = want_short and normalized <= (10 - config.STRONG_SIGNAL_SCORE)

    if want_long and trend4h in ("bull", "neutral"):
        result.direction = "long"
    elif want_short and trend4h in ("bear", "neutral"):
        result.direction = "short"
    # Allow a strong 1h short even in a 6h bull regime (trend divergence)
    elif strong_short and trend4h == "bull":
        result.direction = "short"
    # Allow a strong 1h long even in a 6h bear regime (counter-trend bounce)
    elif strong_long and trend4h == "bear":
        result.direction = "long"
    elif (want_long or want_short) and trend4h == "unknown":
        if strong_long:  result.direction = "long"
        elif strong_short: result.direction = "short"
        else: result.direction = "neutral"
    else:
        result.direction = "neutral"

    # ── Trade levels ──────────────────────────────────────────────────────────
    close = _last(df_primary["close"])
    atr   = _last(df_primary["atr"])
    result.entry_price = close
    result.atr         = atr if not np.isnan(atr) else 0.0

    if not np.isnan(atr) and atr > 0:
        if result.direction == "long":
            result.stop_loss   = round(close - config.ATR_STOP_MULTIPLIER   * atr, 6)
            result.take_profit = round(close + config.ATR_TARGET_MULTIPLIER * atr, 6)
        elif result.direction == "short":
            result.stop_loss   = round(close + config.ATR_STOP_MULTIPLIER   * atr, 6)
            result.take_profit = round(close - config.ATR_TARGET_MULTIPLIER * atr, 6)

        if result.direction != "neutral":
            risk   = abs(close - result.stop_loss)
            reward = abs(result.take_profit - close)
            result.risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    return result