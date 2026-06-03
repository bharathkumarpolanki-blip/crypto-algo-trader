"""
ML Regime Classifier — ml/regime_classifier.py

Inspired by FreqAI's adaptive market regime detection.
Replaces our hard-coded EMA slope rules with a trained classifier.

Our implementation:
  - Label regimes from historical data using FORWARD returns
    bull  = next 24 candles return > +3%
    bear  = next 24 candles return < -3%
    sideways = between -3% and +3%
  - Train a LightGBM classifier on those labels
  - Returns regime probabilities + dominant regime

Advantages over rule-based approach:
  - Learns non-linear regime boundaries from actual price behaviour
  - Adapts to different market structures (trending vs ranging)
  - Provides probability rather than binary yes/no
"""

from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from ml.feature_builder import build_features
from ml.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "regime_classifier.pkl")

FORWARD_WINDOW  = 24     # candles to look ahead for labelling
BULL_THRESHOLD  = 0.030  # +3% = bull
BEAR_THRESHOLD  = -0.030 # -3% = bear
MIN_CANDLES     = 150


@dataclass
class RegimePrediction:
    regime:        str    # "bull" | "bear" | "sideways"
    bull_prob:     float
    bear_prob:     float
    sideways_prob: float
    confidence:    float  # probability of the dominant regime
    reliable:      bool


class RegimeClassifier:
    """
    3-class LightGBM classifier: bull / bear / sideways.
    Trained once per symbol, reused for all predictions.
    """

    def __init__(self):
        self.model        = None
        self.preprocessor = Preprocessor(test_size=0.2, weight_decay=0.85)
        self.encoder      = LabelEncoder()
        self.trained_at   = ""
        self.X_train_ref  = None
        self.test_acc     = 0.0

    def _label_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Label each candle with the regime that followed it.
        Uses actual forward returns — no manual rule tuning needed.
        """
        future_return = df["close"].shift(-FORWARD_WINDOW) / df["close"] - 1
        labels = pd.Series("sideways", index=df.index)
        labels[future_return >  BULL_THRESHOLD] = "bull"
        labels[future_return < BEAR_THRESHOLD]  = "bear"
        return labels

    def train(self, df: pd.DataFrame) -> bool:
        if len(df) < MIN_CANDLES + FORWARD_WINDOW:
            logger.debug("Not enough candles to train regime classifier")
            return False
        try:
            feature_df = build_features(df)
            labels     = self._label_regimes(df)

            # Drop last FORWARD_WINDOW rows (no future label)
            feature_df = feature_df.iloc[:-FORWARD_WINDOW]
            labels     = labels.iloc[:-FORWARD_WINDOW]

            # Encode labels: bull/bear/sideways → integers
            y_encoded = self.encoder.fit_transform(labels)

            # Preprocess
            data = self.preprocessor.fit_transform(feature_df, pd.Series(y_encoded, index=feature_df.index))

            # Guard: need at least 2 distinct classes in the TRAIN split, else
            # the classifier can't fit (e.g. a strong one-sided market produces
            # all-'bear' or all-'sideways' labels → degenerate model).
            n_classes_train = len(np.unique(data.y_train))
            if n_classes_train < 2:
                logger.info("Regime classifier skipped — only %d class in training "
                            "window (one-sided market). Keeping previous model.",
                            n_classes_train)
                return False

            # Build into a LOCAL variable — only swap into self.model on full
            # success, so a failed retrain never discards a previously-good model.
            # Disable early_stopping: its internal validation split can land a
            # single class (degenerate) and leave the model without
            # _baseline_prediction. Fixed iteration count is safe here.
            model = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=42,
            )
            model.fit(data.X_train, data.y_train, sample_weight=data.weights)

            # Verify the model is fully fitted before trusting it
            pred_cls = model.predict(data.X_test)

            # Commit — only now does the new model become live
            self.model       = model
            self.test_acc    = float((pred_cls == data.y_test.astype(int)).mean())
            self.X_train_ref = data.X_train
            self.trained_at  = pd.Timestamp.now(tz="UTC").isoformat()

            from collections import Counter
            dist = Counter(labels)
            logger.info("Regime classifier trained | accuracy=%.2f | distribution=%s",
                        self.test_acc, dict(dist))
            return True

        except Exception as e:
            logger.warning("Regime classifier training skipped (keeping previous model): %s", e)
            return False

    def predict(self, df: pd.DataFrame) -> RegimePrediction:
        """Predict regime for the current candle."""
        neutral = RegimePrediction("sideways", 0.33, 0.33, 0.34, 0.34, False)

        if self.model is None:
            return neutral

        try:
            feature_df = build_features(df)
            last_row   = feature_df.iloc[[-1]]
            X_live     = self.preprocessor.transform_live(last_row)
            if X_live is None:
                return neutral

            # Dissimilarity check
            reliable = True
            if self.X_train_ref is not None:
                d = self.preprocessor.dissimilarity_score(self.X_train_ref, X_live)
                if d > 1.0:
                    reliable = False

            probs = self.model.predict_proba(X_live)[0]   # aligned with model.classes_

            # IMPORTANT: map probabilities using the model's OWN learned classes
            # (encoded ints), then decode each to its label. If the model only
            # saw 2 of 3 regimes, model.classes_ has 2 entries — using
            # encoder.classes_ (3 entries) here would misalign the columns.
            model_class_ids = self.model.classes_            # encoded ints actually learned
            labels_for_ids  = self.encoder.inverse_transform(model_class_ids.astype(int))

            prob_map = {lbl: float(probs[i]) for i, lbl in enumerate(labels_for_ids)}
            bull_p   = prob_map.get("bull",     0.0)
            bear_p   = prob_map.get("bear",     0.0)
            side_p   = prob_map.get("sideways", 0.0)

            regime     = max(prob_map, key=prob_map.get)
            confidence = prob_map[regime]

            return RegimePrediction(
                regime=regime,
                bull_prob=round(bull_p, 3),
                bear_prob=round(bear_p, 3),
                sideways_prob=round(side_p, 3),
                confidence=round(confidence, 3),
                reliable=reliable,
            )

        except Exception as e:
            logger.warning("Regime predict failed: %s", e)
            return neutral

    def save(self) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "RegimeClassifier | None":
        if not os.path.exists(MODEL_PATH):
            return None
        try:
            with open(MODEL_PATH, "rb") as f:
                obj = pickle.load(f)
            logger.info("Loaded regime classifier (acc=%.2f)", obj.test_acc)
            return obj
        except Exception as e:
            logger.warning("Could not load regime classifier: %s", e)
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────

_classifier: RegimeClassifier | None = None

def get_regime_classifier() -> RegimeClassifier:
    global _classifier
    if _classifier is None:
        _classifier = RegimeClassifier.load() or RegimeClassifier()
    return _classifier
