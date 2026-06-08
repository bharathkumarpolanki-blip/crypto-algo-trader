"""
Mean-reversion signal engine (1h) — the evidence-driven rewrite.

WHY THIS EXISTS
---------------
The trend-following engine (core/strategies.py) was proven to have a statistically
significant *negative* Information Coefficient on 1h crypto (IC=-0.037, p=0.03):
its highest-weighted components (ema_trend, ichimoku, regime) actively
anti-predict forward returns because 1h crypto MEAN-REVERTS — a trend signal
fires after the move, exactly when price snaps back.

The only components with POSITIVE IC were the mean-reversion family
(volume +0.039 sig; rsi/bollinger/stochastic weakly +). This module rebuilds the
thesis around them:

  - DROP the trend components entirely (they have negative predictive value).
  - FADE extremes instead of chasing them: stretched-below-mean → long,
    stretched-above-mean → short.
  - Trade ONLY in ranging conditions (low ADX). Mean-reversion dies in strong
    trends, so a strong trend is a HARD BLOCK (the inverse of the old engine).
  - Volume confirmation is central (capitulation/exhaustion at the extreme).
  - GEOMETRY FLIP: target = back toward the mean (tight); stop = beyond the
    extreme (wider). High win rate / small wins — the opposite of trend geometry.

This is a HYPOTHESIS under test. It does not trade live until edge_validation
shows a positive, significant, cost-surviving IC. Same bar as everything else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from core.strategies import SignalResult, _last, _safe


# ── Mean-reversion component scores (positive = expect price UP / fade-from-below)
# Each returns a signed value; positive ⇒ bullish reversion, negative ⇒ bearish.

def score_zscore(df: pd.DataFrame, period: int = 20) -> float:
    """Price z-score vs its rolling mean. Stretched LOW → +, stretched HIGH → -."""
    close = df["close"]
    mean  = close.rolling(period).mean().iloc[-1]
    std   = close.rolling(period).std().iloc[-1]
    c     = _last(close)
    if np.isnan(mean) or np.isnan(std) or std == 0:
        return 0.0
    z = (c - mean) / std
    # Fade the deviation: signal = -z, scaled so |z|=2 → ~±2.0
    return float(np.clip(-z, -2.0, 2.0))


def score_rsi_revert(df: pd.DataFrame) -> float:
    """Oversold RSI → +, overbought → -. Reward turning, not just level."""
    rsi      = _last(df["rsi"])
    rsi_prev = _last(df["rsi"], 3)
    if np.isnan(rsi):
        return 0.0
    s = (50.0 - rsi) / 25.0          # rsi 25 → +1.0 ; rsi 75 → -1.0
    # bonus when it's actually turning back toward the mean
    if not np.isnan(rsi_prev):
        if rsi < 35 and rsi > rsi_prev:  s += 0.5   # bouncing up from oversold
        if rsi > 65 and rsi < rsi_prev:  s -= 0.5   # rolling over from overbought
    return float(np.clip(s, -2.0, 2.0))


def score_bbpct_revert(df: pd.DataFrame) -> float:
    """Bollinger %B fade: below lower band → +, above upper band → -."""
    pct = _last(df["bb_pct"])
    if np.isnan(pct):
        return 0.0
    # %B 0.5 = mid → 0 ; %B 0 (lower band) → +1 ; %B 1 (upper) → -1 ; beyond → ±2
    s = (0.5 - pct) * 2.0
    return float(np.clip(s * 1.0, -2.0, 2.0))


def score_stoch_revert(df: pd.DataFrame) -> float:
    """Stochastic fade with cross confirmation."""
    k = _last(df["stoch_k"]); d = _last(df["stoch_d"])
    kp = _last(df["stoch_k"], 2); dp = _last(df["stoch_d"], 2)
    if any(np.isnan(v) for v in [k, d, kp, dp]):
        return 0.0
    s = (50.0 - k) / 30.0
    crossed_up   = k > d and kp <= dp
    crossed_down = k < d and kp >= dp
    if k < 25 and crossed_up:   s += 0.75
    if k > 75 and crossed_down: s -= 0.75
    return float(np.clip(s, -2.0, 2.0))


def score_volume_confirm(df: pd.DataFrame, raw_dir: float) -> float:
    """
    Volume confirmation (the only positive-IC component). Mean-reversion entries
    are strongest when the extreme prints on a volume SURGE (capitulation /
    exhaustion). Returns a magnitude bonus aligned with the reversion direction.
    """
    vr = _last(df["vol_ratio"]); cmf = _last(df["cmf"])
    if np.isnan(vr):
        return 0.0
    s = 0.0
    if vr > config.VOLUME_SURGE_MULTIPLIER:   s += 1.0   # capitulation/exhaustion spike
    elif vr > 1.0:                            s += 0.4
    # CMF divergence supports reversion (selling exhausted at a low → cmf turning up)
    if not np.isnan(cmf):
        if raw_dir > 0 and cmf > -0.05:  s += 0.3
        if raw_dir < 0 and cmf < 0.05:   s += 0.3
    # align magnitude with the reversion direction
    return float(np.sign(raw_dir) * min(s, 1.5)) if raw_dir != 0 else 0.0


def is_ranging(df: pd.DataFrame) -> bool:
    """Mean-reversion only works in RANGES. Strong trend (high ADX) = hard block."""
    adx = _last(df["adx"])
    if np.isnan(adx):
        return False
    return adx < config.MR_ADX_MAX        # e.g. < 25 = not strongly trending


_MR_MAX_RAW = 9.5   # z(2) + rsi(2) + bb(2) + stoch(2) + vol(1.5)


def analyse_mr(symbol: str,
               df_primary: pd.DataFrame,
               df_daily: pd.DataFrame | None = None) -> SignalResult:
    """
    Mean-reversion composite. Returns a SignalResult with direction/score/levels.
    Geometry is FLIPPED vs the trend engine: tight target (revert to mean),
    wider stop (beyond the extreme).
    """
    result = SignalResult(symbol=symbol, direction="neutral", score=0.0,
                          timeframe=config.TF_PRIMARY)
    if df_primary is None or len(df_primary) < 60:
        result.note = "insufficient data"
        return result

    comp = {}
    comp["zscore"]     = _safe(score_zscore(df_primary))
    comp["rsi_revert"] = _safe(score_rsi_revert(df_primary))
    comp["bb_revert"]  = _safe(score_bbpct_revert(df_primary))
    comp["stoch"]      = _safe(score_stoch_revert(df_primary))
    raw_dir = comp["zscore"] + comp["rsi_revert"] + comp["bb_revert"] + comp["stoch"]
    comp["volume"]     = _safe(score_volume_confirm(df_primary, raw_dir))

    raw = sum(comp.values())
    normalized = max(0.0, min(10.0, round((raw + _MR_MAX_RAW) / (2 * _MR_MAX_RAW) * 10, 2)))
    result.components = comp
    result.score = normalized
    result.conviction = round(abs(normalized - 5.0) / 5.0, 3)

    ranging = is_ranging(df_primary)
    result.note = f"MR raw={raw:+.2f} {'ranging' if ranging else 'TRENDING(block)'}"

    long_ok  = normalized >= config.MR_MIN_SCORE        and raw > 0
    short_ok = normalized <= (10 - config.MR_MIN_SCORE) and raw < 0

    direction = "neutral"
    if ranging:                       # HARD BLOCK: no MR trades in strong trends
        if long_ok:    direction = "long"
        elif short_ok: direction = "short"
    result.direction = direction

    # ── Trade levels — FLIPPED geometry (tight target / wide stop) ────────────
    close = _last(df_primary["close"]); atr = _last(df_primary["atr"])
    result.entry_price = close
    result.atr = atr if not np.isnan(atr) else 0.0
    if not np.isnan(atr) and atr > 0 and direction != "neutral":
        tgt_mult = config.MR_ATR_TARGET   # small — revert to mean
        stp_mult = config.MR_ATR_STOP     # wide — beyond the extreme
        if direction == "long":
            result.stop_loss   = round(close - stp_mult * atr, 6)
            result.take_profit = round(close + tgt_mult * atr, 6)
        else:
            result.stop_loss   = round(close + stp_mult * atr, 6)
            result.take_profit = round(close - tgt_mult * atr, 6)
        risk   = abs(close - result.stop_loss)
        reward = abs(result.take_profit - close)
        result.risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    return result
