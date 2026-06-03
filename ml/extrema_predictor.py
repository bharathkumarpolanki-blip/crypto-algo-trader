"""
Extrema Predictor — ml/extrema_predictor.py

Concept adapted (own implementation) from FreqAI's "spice rack" add-spice-rack branch.

Instead of predicting "will the next N candles be profitable" (noisy),
this predicts a smarter target: "is the current candle near a local BOTTOM
(buy) or local TOP (sell)?"

Labelling:
  - scipy.signal.argrelextrema finds significant peaks/troughs over a window
  - local minima (troughs) → label -1  (good place to BUY)
  - local maxima (peaks)   → label +1  (good place to SELL)
  - everything else        →  0

Model:
  - Gradient-boosting regressor (sklearn) predicts a continuous value
  - Output near -1 → we are at/near a bottom → bullish (buy the dip)
  - Output near +1 → we are at/near a top    → bearish (sell the rip)

The continuous output becomes a strategy score component in [-2, +2]:
  bottom (predict ≈ -1) → +2 (bullish: buy low)
  top    (predict ≈ +1) → -2 (bearish: sell high)

This is "mean-reversion aware" — it complements the trend-following components.
"""

from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone

from scipy.signal import argrelextrema
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from ml.feature_builder import build_features
from ml.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
EXTREMA_ORDER = 20      # candles on each side to qualify as a local extremum
MIN_CANDLES   = 200
MAX_MAE       = 0.45    # if model can't predict extrema (MAE too high) → ignore


@dataclass
class ExtremaPrediction:
    value:      float   # continuous: -1 (bottom) … +1 (top)
    signal:     str     # "buy_zone" | "sell_zone" | "neutral"
    score:      float   # [-2, +2] for strategy integration (inverted: bottom=+2)
    confidence: float   # how strong the extrema signal is (0-1)
    reliable:   bool


class ExtremaPredictor:
    """
    One regressor per symbol that learns to predict proximity to local tops/bottoms.
    """

    def __init__(self):
        self._models: dict[str, dict] = {}    # symbol → {model, preprocessor, mae, ...}
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._load_all()

    # ── Labelling ──────────────────────────────────────────────────────────────

    def _label_extrema(self, df: pd.DataFrame) -> pd.Series:
        """
        Label each candle: -1 at local troughs, +1 at local peaks, 0 elsewhere.
        Then smooth into a continuous target by linear ramp toward each extremum,
        so the model learns 'how close to a top/bottom' rather than a hard flag.
        """
        close = df["close"].values
        labels = np.zeros(len(close), dtype=np.float32)

        min_idx = argrelextrema(close, np.less,    order=EXTREMA_ORDER)[0]
        max_idx = argrelextrema(close, np.greater, order=EXTREMA_ORDER)[0]

        labels[min_idx] = -1.0
        labels[max_idx] = +1.0

        # Smooth: ramp the label between consecutive extrema so the target is
        # continuous (e.g. halfway between a bottom and top ≈ 0).
        series = pd.Series(labels, index=df.index)
        # Forward/backward fill the non-zero anchors, then interpolate linearly
        anchors = series.replace(0, np.nan)
        smoothed = anchors.interpolate(method="linear", limit_direction="both")
        smoothed = smoothed.fillna(0).clip(-1, 1)
        return smoothed.astype(np.float32)

    # ── Training ────────────────────────────────────────────────────────────────

    def train(self, symbol: str, df: pd.DataFrame) -> bool:
        if len(df) < MIN_CANDLES:
            return False
        try:
            feature_df = build_features(df)
            targets    = self._label_extrema(df)

            # argrelextrema can't confirm extrema near the very end (needs ORDER
            # candles of lookahead), so drop the last EXTREMA_ORDER rows.
            feature_df = feature_df.iloc[:-EXTREMA_ORDER]
            targets    = targets.iloc[:-EXTREMA_ORDER]

            proc = Preprocessor(test_size=0.2, weight_decay=0.92)
            data = proc.fit_transform(feature_df, targets)

            # early_stopping disabled — see signal_predictor for rationale
            # (degenerate internal validation split on skewed data).
            model = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=42,
            )
            model.fit(data.X_train, data.y_train, sample_weight=data.weights)

            test_pred = model.predict(data.X_test)
            mae = float(mean_absolute_error(data.y_test, test_pred))

            logger.info("Extrema model %s | MAE=%.3f  features=%d  samples=%d",
                        symbol, mae, data.X_train.shape[1], data.n_train)

            self._models[symbol] = {
                "model":        model,
                "preprocessor": proc,
                "mae":          mae,
                "X_train_ref":  data.X_train,
                "trained_at":   datetime.now(timezone.utc).isoformat(),
            }
            self._save(symbol)
            return True
        except Exception as e:
            logger.error("Extrema training failed for %s: %s", symbol, e)
            return False

    # ── Prediction ──────────────────────────────────────────────────────────────

    def predict(self, symbol: str, df: pd.DataFrame) -> ExtremaPrediction:
        neutral = ExtremaPrediction(0.0, "neutral", 0.0, 0.0, False)

        m = self._models.get(symbol)
        if m is None:
            return neutral

        # Guard: if the model can't predict extrema well, don't use it
        if m["mae"] > MAX_MAE:
            return neutral

        try:
            feature_df = build_features(df)
            X_live = m["preprocessor"].transform_live(feature_df.iloc[[-1]])
            if X_live is None:
                return neutral

            # Reliability via dissimilarity
            reliable = True
            if m.get("X_train_ref") is not None:
                d = m["preprocessor"].dissimilarity_score(m["X_train_ref"], X_live)
                if d > 1.0:
                    reliable = False

            value = float(np.clip(m["model"].predict(X_live)[0], -1, 1))

            # Interpret: near -1 = bottom (buy), near +1 = top (sell)
            if value < -0.3:
                signal = "buy_zone"
            elif value > 0.3:
                signal = "sell_zone"
            else:
                signal = "neutral"

            # Score is INVERTED: at a bottom (value≈-1) we want to go LONG (+2)
            score      = -value * 2.0          # -1→+2 (buy),  +1→-2 (sell)
            confidence = min(abs(value), 1.0)

            return ExtremaPrediction(
                value=round(value, 3),
                signal=signal,
                score=round(score, 3),
                confidence=round(confidence, 3),
                reliable=reliable,
            )
        except Exception as e:
            logger.warning("Extrema predict failed for %s: %s", symbol, e)
            return neutral

    def needs_retrain(self, symbol: str, hours: int = 12) -> bool:
        m = self._models.get(symbol)
        if m is None:
            return True
        try:
            t = datetime.fromisoformat(m["trained_at"])
            return (datetime.now(timezone.utc) - t).total_seconds() / 3600 >= hours
        except Exception:
            return True

    def model_stats(self, symbol: str) -> dict:
        m = self._models.get(symbol)
        if m is None:
            return {}
        return {"mae": round(m["mae"], 3), "trained_at": m["trained_at"]}

    # ── Persistence ──────────────────────────────────────────────────────────────

    def _save(self, symbol: str) -> None:
        path = os.path.join(MODEL_DIR, f"extrema_{symbol.replace('/','_')}.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(self._models[symbol], f)
        except Exception as e:
            logger.warning("Could not save extrema model %s: %s", symbol, e)

    def _load_all(self) -> None:
        if not os.path.exists(MODEL_DIR):
            return
        for fname in os.listdir(MODEL_DIR):
            if not fname.startswith("extrema_") or not fname.endswith(".pkl"):
                continue
            try:
                symbol = fname.replace("extrema_", "").replace(".pkl", "").replace("_", "/", 1)
                with open(os.path.join(MODEL_DIR, fname), "rb") as f:
                    self._models[symbol] = pickle.load(f)
            except Exception as e:
                logger.warning("Could not load extrema model %s: %s", fname, e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_extrema: ExtremaPredictor | None = None

def get_extrema_predictor() -> ExtremaPredictor:
    global _extrema
    if _extrema is None:
        _extrema = ExtremaPredictor()
    return _extrema
