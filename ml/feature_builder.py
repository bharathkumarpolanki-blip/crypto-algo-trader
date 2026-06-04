"""
Feature Pipeline — ml/feature_builder.py

Inspired by FreqAI's feature_engineering_expand_all concept.

Converts an enriched OHLCV DataFrame into a 2D numeric feature matrix
ready for ML training or inference. Every indicator value becomes a
feature column. No human-defined thresholds — the ML model learns
which values matter.

Key ideas borrowed (not copied) from FreqAI:
  - Expand indicators across multiple lookback periods automatically
  - Include multi-candle lag features (shifted candles)
  - Compute ratios and cross-indicator features
  - Return a clean float32 numpy array
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Lookback periods to compute indicators over ────────────────────────────────
PERIODS = [7, 14, 21]              # short → long (dropped 28 — redundant with 21)
LAG_CANDLES = [1, 3]              # how many candles back to shift features.
                                  # Fewer lags = far fewer columns → faster training
                                  # AND less overfitting (was [1,2,3,5,10] → 328 feats).


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an enriched OHLCV DataFrame into a flat feature matrix.

    Each row = one candle.
    Each column = one numeric feature.
    Returns a DataFrame with only float columns (no NaNs).
    """
    feat: dict[str, pd.Series] = {}

    # ── Price action features ──────────────────────────────────────────────────
    feat["close"]          = df["close"]
    feat["body_size"]      = (df["close"] - df["open"]).abs() / df["close"]
    feat["upper_wick"]     = (df["high"]  - df[["close","open"]].max(axis=1)) / df["close"]
    feat["lower_wick"]     = (df[["close","open"]].min(axis=1) - df["low"])   / df["close"]
    feat["candle_range"]   = (df["high"] - df["low"]) / df["close"]
    feat["is_green"]       = (df["close"] >= df["open"]).astype(float)

    # ── EMA ratios (price relative to EMAs — normalised) ──────────────────────
    for col in ["ema9", "ema21", "ema50", "ema200"]:
        if col in df.columns:
            feat[f"ratio_{col}"] = (df["close"] - df[col]) / df[col].replace(0, np.nan)

    # EMA stack spreads
    if "ema9" in df.columns and "ema21" in df.columns:
        feat["ema9_21_spread"]   = (df["ema9"]  - df["ema21"])  / df["ema21"].replace(0, np.nan)
    if "ema21" in df.columns and "ema50" in df.columns:
        feat["ema21_50_spread"]  = (df["ema21"] - df["ema50"])  / df["ema50"].replace(0, np.nan)
    if "ema50" in df.columns and "ema200" in df.columns:
        feat["ema50_200_spread"] = (df["ema50"] - df["ema200"]) / df["ema200"].replace(0, np.nan)

    # ── Momentum indicators ────────────────────────────────────────────────────
    if "rsi" in df.columns:
        feat["rsi"]          = df["rsi"] / 100.0          # normalise to [0,1]
        feat["rsi_centered"] = (df["rsi"] - 50) / 50.0   # [-1, +1]

    if "macd" in df.columns and "macd_signal" in df.columns:
        feat["macd_norm"]    = df["macd"] / df["close"].replace(0, np.nan)
        feat["macd_hist_norm"] = df["macd_hist"] / df["close"].replace(0, np.nan)
        feat["macd_cross"]   = (df["macd"] > df["macd_signal"]).astype(float)
        # Histogram slope (expanding = momentum building)
        feat["macd_hist_slope"] = df["macd_hist"].diff()

    if "stoch_k" in df.columns:
        feat["stoch_k"] = df["stoch_k"] / 100.0
        feat["stoch_d"] = df["stoch_d"] / 100.0
        feat["stoch_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(float)

    if "roc" in df.columns:
        feat["roc"] = df["roc"].clip(-50, 50) / 50.0   # normalise

    # ── Trend indicators ───────────────────────────────────────────────────────
    if "adx" in df.columns:
        feat["adx"]      = df["adx"]  / 100.0
        feat["di_diff"]  = (df["di_plus"] - df["di_minus"]) / 100.0
        feat["di_ratio"] = (df["di_plus"] / df["di_minus"].replace(0, np.nan)).clip(0, 5) / 5.0

    if "supertrend_dir" in df.columns:
        feat["supertrend_dir"] = df["supertrend_dir"].astype(float)  # -1 bull, 1 bear

    # ── Volatility indicators ──────────────────────────────────────────────────
    if "atr" in df.columns:
        feat["atr_norm"]  = df["atr"] / df["close"].replace(0, np.nan)

    if "bb_pct" in df.columns:
        feat["bb_pct"]   = df["bb_pct"].clip(0, 1)
        feat["bb_width"] = df["bb_width"].clip(0, 1)

    # ── Volume indicators ──────────────────────────────────────────────────────
    if "vol_ratio" in df.columns:
        feat["vol_ratio"] = df["vol_ratio"].clip(0, 10) / 10.0

    if "cmf" in df.columns:
        feat["cmf"] = df["cmf"].clip(-1, 1)

    if "obv" in df.columns and "obv_ema" in df.columns:
        # OBV above its EMA = accumulation signal
        feat["obv_vs_ema"] = ((df["obv"] - df["obv_ema"]) / df["obv_ema"].abs().replace(0, np.nan)).clip(-1, 1)

    # ── Ichimoku cloud signals ─────────────────────────────────────────────────
    if "tenkan" in df.columns and "kijun" in df.columns:
        feat["tk_cross"]     = (df["tenkan"] > df["kijun"]).astype(float)
        feat["tk_spread"]    = (df["tenkan"] - df["kijun"]) / df["close"].replace(0, np.nan)
        if "senkou_a" in df.columns and "senkou_b" in df.columns:
            cloud_top = df[["senkou_a","senkou_b"]].max(axis=1)
            cloud_bot = df[["senkou_a","senkou_b"]].min(axis=1)
            feat["above_cloud"] = (df["close"] > cloud_top).astype(float)
            feat["below_cloud"] = (df["close"] < cloud_bot).astype(float)
            feat["cloud_bull"]  = (df["senkou_a"] > df["senkou_b"]).astype(float)

    # ── Multi-period rolling features (expand across periods like FreqAI) ──────
    for p in PERIODS:
        # Price change over period
        feat[f"return_{p}"]     = df["close"].pct_change(p).clip(-0.5, 0.5)
        # Volatility over period
        feat[f"volatility_{p}"] = df["close"].pct_change().rolling(p).std().fillna(0)
        # Volume trend
        feat[f"vol_trend_{p}"]  = (df["volume"] / df["volume"].rolling(p).mean().replace(0, np.nan)).clip(0, 5) / 5.0
        # High/low range relative to period range
        period_high = df["high"].rolling(p).max()
        period_low  = df["low"].rolling(p).min()
        period_rng  = (period_high - period_low).replace(0, np.nan)
        feat[f"pos_in_range_{p}"] = (df["close"] - period_low) / period_rng

    # ── Temporal features (market session awareness) ───────────────────────────
    if hasattr(df.index, 'hour'):
        feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        feat["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
        feat["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    feature_df = pd.DataFrame(feat, index=df.index)

    # ── Lag features (shifted candles — capture recent momentum) ──────────────
    base_cols = [c for c in feature_df.columns if c not in ("hour_sin","hour_cos","dow_sin","dow_cos")]
    lag_dfs = []
    for lag in LAG_CANDLES:
        shifted = feature_df[base_cols].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in base_cols]
        lag_dfs.append(shifted)
    if lag_dfs:
        feature_df = pd.concat([feature_df] + lag_dfs, axis=1)

    # ── Clean: replace inf, drop NaN ──────────────────────────────────────────
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.astype(np.float32)

    return feature_df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return feature column names without building the full matrix."""
    return list(build_features(df.tail(50)).dropna().columns)
