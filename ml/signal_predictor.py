"""
ML Signal Predictor — ml/signal_predictor.py

Inspired by FreqAI's LightGBM Regressor concept:
  - Train a model to predict expected future return
  - Retrain periodically as market conditions change
  - Feed prediction confidence back into the signal score
  - Persist model to disk, reload on restart

Our implementation:
  - Target: 3-class outcome over the next N candles — long / short / sideways.
    Each class is LEARNED (short is NOT a complement of long).
  - Model: sklearn HistGradientBoosting multiclass (no system deps)
  - Retraining: background thread, every RETRAIN_HOURS
  - Integration: adds ML component to strategy scoring (component 15)
  - Confidence: prediction probability used to scale position size

All logic written from scratch.
"""

from __future__ import annotations

import os
import pickle
import logging
import threading
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from ml.feature_builder import build_features
from ml.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR        = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH       = os.path.join(MODEL_DIR, "signal_predictor.pkl")
RETRAIN_HOURS    = 12           # retrain every 12 hours
MIN_TRAIN_CANDLES = 200         # minimum history needed to train
FORWARD_CANDLES  = 6            # predict outcome N candles ahead
PROFIT_THRESHOLD = 0.005        # 0.5% minimum to call a trade "profitable"
MIN_TEST_AUC     = 0.53         # below this the model doesn't beat random — ignore it
DIRECTION_MIN_PROB = 0.40       # min learned probability for a directional class to
                                # fire (3-class: ~0.33 is chance, so 0.40 = real lead)


def _macro_ovr_auc(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """
    Macro one-vs-rest AUC across all classes present in `y`. Robust to a class
    being absent from a split (skips it) — so it works for the 3-class
    long/short/sideways model even on skewed crypto data. Falls back to 0.5.
    """
    aucs = []
    for idx, c in enumerate(classes):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yc, proba[:, idx]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.5


# ── Prediction result ─────────────────────────────────────────────────────────

@dataclass
class MLPrediction:
    direction:   str    # "long" | "short" | "neutral"
    confidence:  float  # 0.0 → 1.0
    expected_return: float  # expected % return (positive = up, negative = down)
    score:       float  # normalised to [-2, +2] for strategy integration
    reliable:    bool   # False if dissimilarity too high (like do_predict in FreqAI)
    trained_at:  str    = ""


# ── Per-symbol model state ────────────────────────────────────────────────────

class _SymbolModel:
    def __init__(self, symbol: str):
        self.symbol     = symbol
        self.model      = None          # fitted LightGBM model
        self.preprocessor = Preprocessor(test_size=0.2, weight_decay=0.92)
        self.feature_names: list[str] = []
        self.trained_at: str = ""
        self.X_train_ref: Optional[np.ndarray] = None   # for dissimilarity check
        self.train_auc:  float = 0.0
        self.test_auc:   float = 0.0
        self.importances: dict[str, float] = {}
        self.classes_:   list[float] = []      # learned class order (-1/0/+1)

    def is_trained(self) -> bool:
        return self.model is not None

    def needs_retrain(self) -> bool:
        if not self.trained_at:
            return True
        try:
            trained = datetime.fromisoformat(self.trained_at)
            age_hours = (datetime.now(timezone.utc) - trained).total_seconds() / 3600
            return age_hours >= RETRAIN_HOURS
        except Exception:
            return True


# ── Main predictor class ──────────────────────────────────────────────────────

class SignalPredictor:
    """
    Manages one LightGBM model per symbol.
    Trains in a background thread, serves predictions synchronously.
    """

    def __init__(self):
        self._models:  dict[str, _SymbolModel] = {}
        self._lock     = threading.Lock()
        self._training = threading.Event()
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._load_all()

    # ── Feature + target preparation ─────────────────────────────────────────

    def _prepare_targets(self, df: pd.DataFrame) -> pd.Series:
        """
        Target: sign of the future return N candles ahead.
        +1 = price went up by > PROFIT_THRESHOLD (profitable long)
        -1 = price went down by > PROFIT_THRESHOLD (profitable short)
         0 = sideways (not traded)

        Used directly as a 3-class multiclass target (long / short / sideways).
        """
        future_return = df["close"].shift(-FORWARD_CANDLES) / df["close"] - 1
        target = pd.Series(0, index=df.index, dtype=np.float32)
        target[future_return >  PROFIT_THRESHOLD] = 1.0   # long profitable
        target[future_return < -PROFIT_THRESHOLD] = -1.0  # short profitable
        return target

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Train a gradient-boosting classifier on historical data for a symbol.
        Uses sklearn HistGradientBoostingClassifier (no system deps).
        Returns True if training succeeded.
        """
        if len(df) < MIN_TRAIN_CANDLES:
            logger.debug("Not enough candles to train %s (%d < %d)",
                         symbol, len(df), MIN_TRAIN_CANDLES)
            return False

        try:
            # Build features
            feature_df = build_features(df)
            targets    = self._prepare_targets(df)   # -1 short, 0 sideways, +1 long

            # Align and drop last FORWARD_CANDLES (no future target yet)
            feature_df = feature_df.iloc[:-FORWARD_CANDLES]
            targets    = targets.iloc[:-FORWARD_CANDLES].astype(np.float32)

            # MULTICLASS target: the model now LEARNS all three outcomes
            # (long / short / sideways) directly, instead of the old binary
            # "long vs not-long" where short was just 1 - P(long). Short is now
            # a learned class with its own probability — no information missed.
            proc = Preprocessor(test_size=0.2, weight_decay=0.92)
            data = proc.fit_transform(feature_df, targets)

            # Need at least two classes present to train a classifier
            if len(np.unique(data.y_train)) < 2:
                logger.debug("Only one class for %s — skipping", symbol)
                return False

            # Gradient-boosting binary classifier (sklearn, pure Python deps).
            # early_stopping disabled: its internal validation split can land a
            # single class on skewed crypto data and leave the model half-fitted
            # (missing _baseline_prediction). Fixed iterations + L2 reg is safe.
            model = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=42,
            )
            model.fit(data.X_train, data.y_train, sample_weight=data.weights)

            # Evaluate with macro one-vs-rest AUC across ALL present classes
            # (long/short/sideways) — robust to a class missing from a split.
            train_auc = _macro_ovr_auc(data.y_train,
                                       model.predict_proba(data.X_train), model.classes_)
            test_auc  = _macro_ovr_auc(data.y_test,
                                       model.predict_proba(data.X_test),  model.classes_)

            logger.info("ML trained %s | train_AUC=%.3f  test_AUC=%.3f  features=%d  samples=%d",
                        symbol, train_auc, test_auc, data.X_train.shape[1], data.n_train)

            # Permutation importance (dashboard display only) — EXPENSIVE, so it's
            # optional and cheap by default. n_repeats=1 + small subsample.
            importances: dict[str, float] = {}
            try:
                import config as _cfg
                if getattr(_cfg, "ML_COMPUTE_IMPORTANCE", True):
                    from sklearn.inspection import permutation_importance
                    n = min(120, len(data.X_test))
                    pi = permutation_importance(
                        model, data.X_test[:n], data.y_test[:n],
                        n_repeats=1, random_state=42, scoring="roc_auc", n_jobs=-1,
                    )
                    importances = {
                        name: float(imp)
                        for name, imp in zip(data.feature_names, pi.importances_mean)
                        if imp > 0
                    }
            except Exception as e:
                logger.debug("Permutation importance skipped: %s", e)

            # Store
            with self._lock:
                sym_model = self._models.setdefault(symbol, _SymbolModel(symbol))
                sym_model.model          = model
                sym_model.preprocessor   = proc
                sym_model.feature_names  = data.feature_names
                sym_model.trained_at     = datetime.now(timezone.utc).isoformat()
                sym_model.X_train_ref    = data.X_train
                sym_model.train_auc      = train_auc
                sym_model.test_auc       = test_auc
                sym_model.importances    = importances
                sym_model.classes_       = [float(c) for c in model.classes_]

            self._save(symbol)
            return True

        except Exception as e:
            logger.error("ML training failed for %s: %s", symbol, e)
            return False

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, symbol: str, df: pd.DataFrame) -> MLPrediction:
        """
        Generate an ML prediction for the latest candle.
        Returns MLPrediction with direction, confidence, and score.
        If model not trained, returns neutral prediction.
        """
        neutral = MLPrediction("neutral", 0.5, 0.0, 0.0, False)

        with self._lock:
            sym_model = self._models.get(symbol)

        if sym_model is None or not sym_model.is_trained():
            return neutral

        # Guard: a model that doesn't beat random on the test set is worse than
        # useless. Only trust models with test AUC above MIN_TEST_AUC.
        if sym_model.test_auc < MIN_TEST_AUC:
            return MLPrediction("neutral", 0.5, 0.0, 0.0, False,
                                trained_at=sym_model.trained_at)

        try:
            feature_df = build_features(df)
            # Use only the last row (current candle)
            last_row = feature_df.iloc[[-1]]

            X_live = sym_model.preprocessor.transform_live(last_row)
            if X_live is None:
                return neutral

            # Check dissimilarity — is current market like training data?
            reliable = True
            if sym_model.X_train_ref is not None:
                dissim = sym_model.preprocessor.dissimilarity_score(
                    sym_model.X_train_ref, X_live)
                if dissim > 1.0:
                    logger.debug("ML prediction for %s unreliable (dissimilarity=%.2f)", symbol, dissim)
                    reliable = False

            # Per-class probabilities — long / short / sideways are ALL learned
            # (short is no longer 1 - P(long)). Map by the model's class order.
            proba   = sym_model.model.predict_proba(X_live)[0]
            classes = getattr(sym_model, "classes_", None) or [float(c) for c in sym_model.model.classes_]
            probs   = {float(c): float(p) for c, p in zip(classes, proba)}
            p_long  = probs.get(1.0,  0.0)
            p_short = probs.get(-1.0, 0.0)
            p_side  = probs.get(0.0,  0.0)

            # Direction = the dominant LEARNED directional class. It must beat
            # sideways AND clear DIRECTION_MIN_PROB. Sideways winning → neutral.
            if p_long >= p_short and p_long > p_side and p_long >= DIRECTION_MIN_PROB:
                direction, confidence = "long", p_long
            elif p_short > p_long and p_short > p_side and p_short >= DIRECTION_MIN_PROB:
                direction, confidence = "short", p_short
            else:
                direction, confidence = "neutral", max(p_long, p_short, p_side)

            # Signed score in [-2, +2] from the long↔short edge (both learned).
            score = float(np.clip((p_long - p_short) * 2.0, -2.0, 2.0))
            expected_return = (p_long - p_short) * 0.03
            if direction == "neutral":
                score = 0.0
                expected_return = 0.0

            return MLPrediction(
                direction=direction,
                confidence=round(confidence, 3),
                expected_return=round(expected_return, 4),
                score=round(score, 3),
                reliable=reliable,
                trained_at=sym_model.trained_at,
            )

        except Exception as e:
            logger.warning("ML predict failed for %s: %s", symbol, e)
            return neutral

    # ── Background retraining ─────────────────────────────────────────────────

    def retrain_if_needed(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Schedule retraining in a background thread if model is stale.
        Non-blocking — returns immediately.
        """
        with self._lock:
            sym_model = self._models.get(symbol)

        needs = (sym_model is None) or sym_model.needs_retrain()
        if not needs:
            return

        def _train():
            logger.info("ML background retraining: %s", symbol)
            self.train(symbol, df)

        t = threading.Thread(target=_train, daemon=True, name=f"ml-train-{symbol}")
        t.start()

    def train_all_symbols(self, symbols_data: dict[str, pd.DataFrame]) -> None:
        """Train models for all symbols in parallel background threads."""
        for symbol, df in symbols_data.items():
            self.retrain_if_needed(symbol, df)

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self, symbol: str, top_n: int = 10) -> dict[str, float]:
        """
        Return top N most important features (permutation importance).
        HistGradientBoosting has no native gain importance, so we use
        the stored importances computed at training time (if available).
        """
        with self._lock:
            sym_model = self._models.get(symbol)
        if sym_model is None or not sym_model.is_trained():
            return {}
        imp = getattr(sym_model, "importances", None)
        if not imp:
            return {}
        pairs = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        return {k: round(float(v), 4) for k, v in pairs[:top_n]}

    def model_stats(self, symbol: str) -> dict:
        """Return training stats for a symbol."""
        with self._lock:
            sym_model = self._models.get(symbol)
        if sym_model is None:
            return {}
        return {
            "trained_at": sym_model.trained_at,
            "train_auc":  round(sym_model.train_auc, 3),
            "test_auc":   round(sym_model.test_auc, 3),
            "n_features": len(sym_model.feature_names),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, symbol: str) -> None:
        path = os.path.join(MODEL_DIR, f"signal_{symbol.replace('/','_')}.pkl")
        with self._lock:
            sym_model = self._models.get(symbol)
        if sym_model is None:
            return
        try:
            with open(path, "wb") as f:
                pickle.dump(sym_model, f)
        except Exception as e:
            logger.warning("Could not save ML model for %s: %s", symbol, e)

    def _load_all(self) -> None:
        """Load saved signal-predictor models from disk on startup."""
        if not os.path.exists(MODEL_DIR):
            return
        for fname in os.listdir(MODEL_DIR):
            # Only load our own signal_ files — ignore extrema_/regime_/scaler
            if not fname.startswith("signal_") or not fname.endswith(".pkl"):
                continue
            try:
                # "signal_BTC_USD.pkl" → "BTC/USD"
                symbol = fname.replace("signal_", "").replace(".pkl", "").replace("_", "/", 1)
                path   = os.path.join(MODEL_DIR, fname)
                with open(path, "rb") as f:
                    sym_model = pickle.load(f)
                self._models[symbol] = sym_model
                logger.info("Loaded ML model for %s (trained: %s, AUC=%.3f)",
                            symbol, sym_model.trained_at[:10], sym_model.test_auc)
            except Exception as e:
                logger.warning("Could not load model %s: %s", fname, e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_predictor: SignalPredictor | None = None

def get_predictor() -> SignalPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SignalPredictor()
    return _predictor
