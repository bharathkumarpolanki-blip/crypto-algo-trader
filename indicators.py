"""
All technical indicators computed on a OHLCV DataFrame.
Uses the `ta` library where convenient, computes manually where more control is needed.
"""

import numpy as np
import pandas as pd
import config


# ── Trend ─────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df["ema9"]   = ema(df["close"], config.EMA_FAST)
    df["ema21"]  = ema(df["close"], config.EMA_MID)
    df["ema50"]  = ema(df["close"], config.EMA_SLOW)
    df["ema200"] = ema(df["close"], config.EMA_TREND)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    fast  = ema(df["close"], config.MACD_FAST)
    slow  = ema(df["close"], config.MACD_SLOW)
    macd  = fast - slow
    signal = ema(macd, config.MACD_SIGNAL)
    df["macd"]        = macd
    df["macd_signal"] = signal
    df["macd_hist"]   = macd - signal
    return df


def add_adx(df: pd.DataFrame, period: int = config.ADX_PERIOD) -> pd.DataFrame:
    """Average Directional Index — measures trend strength (not direction)."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr_s    = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_s
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_s
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    df["adx"]      = dx.ewm(alpha=1/period, adjust=False).mean()
    df["di_plus"]  = di_plus
    df["di_minus"] = di_minus
    return df


# ── Momentum ──────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.DataFrame:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    k, d, smooth = config.STOCH_K, config.STOCH_D, config.STOCH_SMOOTH
    low_min  = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    k_raw    = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = k_raw.rolling(smooth).mean()
    df["stoch_d"] = df["stoch_k"].rolling(d).mean()
    return df


def add_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Rate of Change — momentum proxy."""
    df["roc"] = df["close"].pct_change(period) * 100
    return df


# ── Volatility ────────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/period, adjust=False).mean()
    return df


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    mid          = df["close"].rolling(config.BB_PERIOD).mean()
    std          = df["close"].rolling(config.BB_PERIOD).std()
    df["bb_mid"] = mid
    df["bb_up"]  = mid + config.BB_STD * std
    df["bb_low"] = mid - config.BB_STD * std
    df["bb_pct"] = (df["close"] - df["bb_low"]) / (df["bb_up"] - df["bb_low"]).replace(0, np.nan)
    df["bb_width"] = (df["bb_up"] - df["bb_low"]) / mid.replace(0, np.nan)
    return df


# ── Volume ────────────────────────────────────────────────────────────────────

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_ma"]    = df["volume"].rolling(config.VOLUME_MA_PERIOD).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    # On-Balance Volume
    direction = np.sign(df["close"].diff())
    df["obv"]  = (direction * df["volume"]).cumsum()
    df["obv_ema"] = ema(df["obv"], 20)

    # Chaikin Money Flow
    mf_mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) \
              / (df["high"] - df["low"]).replace(0, np.nan)
    mf_vol  = mf_mult * df["volume"]
    df["cmf"] = mf_vol.rolling(20).sum() / df["volume"].rolling(20).sum().replace(0, np.nan)
    return df


# ── Support / Resistance ──────────────────────────────────────────────────────

def find_pivot_levels(df: pd.DataFrame, left: int = 10, right: int = 10) -> tuple[list, list]:
    """Detect pivot highs and lows as S/R levels."""
    highs, lows = [], []
    for i in range(left, len(df) - right):
        window_h = df["high"].iloc[i - left: i + right + 1]
        window_l = df["low"].iloc[i - left: i + right + 1]
        if df["high"].iloc[i] == window_h.max():
            highs.append(df["high"].iloc[i])
        if df["low"].iloc[i] == window_l.min():
            lows.append(df["low"].iloc[i])
    return highs[-5:], lows[-5:]


# ── Supertrend ────────────────────────────────────────────────────────────────

def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    hl2    = (df["high"] + df["low"]) / 2
    atr_   = df["high"].combine(df["low"], lambda h, l: h - l)  # rough; refined below
    tr     = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_s  = tr.rolling(period).mean()

    upper_band = hl2 + multiplier * atr_s
    lower_band = hl2 - multiplier * atr_s

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        prev_upper = upper_band.iloc[i - 1]
        prev_lower = lower_band.iloc[i - 1]
        curr_close = df["close"].iloc[i]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]

        upper_band.iloc[i] = curr_upper if curr_upper < prev_upper or df["close"].iloc[i - 1] > prev_upper else prev_upper
        lower_band.iloc[i] = curr_lower if curr_lower > prev_lower or df["close"].iloc[i - 1] < prev_lower else prev_lower

        if pd.isna(supertrend.iloc[i - 1]):
            direction.iloc[i] = 1
        elif supertrend.iloc[i - 1] == prev_upper:
            direction.iloc[i] = -1 if curr_close > upper_band.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if curr_close < lower_band.iloc[i] else -1

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == -1 else upper_band.iloc[i]

    df["supertrend"]     = supertrend
    df["supertrend_dir"] = direction   # -1 = bullish, 1 = bearish (price above/below)
    return df


# ── Ichimoku Cloud ────────────────────────────────────────────────────────────

def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    nine_period_high  = df["high"].rolling(9).max()
    nine_period_low   = df["low"].rolling(9).min()
    df["tenkan"]      = (nine_period_high + nine_period_low) / 2

    period26_high     = df["high"].rolling(26).max()
    period26_low      = df["low"].rolling(26).min()
    df["kijun"]       = (period26_high + period26_low) / 2

    df["senkou_a"]    = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
    period52_high     = df["high"].rolling(52).max()
    period52_low      = df["low"].rolling(52).min()
    df["senkou_b"]    = ((period52_high + period52_low) / 2).shift(26)
    df["chikou"]      = df["close"].shift(-26)
    return df


# ── Master enrichment ─────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to a raw OHLCV dataframe."""
    if len(df) < 60:
        return df
    df = add_emas(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_stochastic(df)
    df = add_atr(df)
    df = add_bollinger(df)
    df = add_adx(df)
    df = add_volume_indicators(df)
    df = add_supertrend(df)
    df = add_ichimoku(df)
    df = add_roc(df)
    return df