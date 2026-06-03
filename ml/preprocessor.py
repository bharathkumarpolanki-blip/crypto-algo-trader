"""
Data Preprocessor — ml/preprocessor.py

Inspired by FreqAI's DataKitchen pipeline concepts:
  - Exponential sample weighting (recent data matters more)
  - StandardScaler normalisation
  - IQR-based outlier clipping (our version of SVM outlier detection)
  - Train/test split with configurable ratio
  - Dissimilarity detection (warn when live data is unlike training data)

All logic written from scratch — no FreqAI code used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
import pickle
import os
from dataclasses import dataclass, field

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")


@dataclass
class ProcessedData:
    X_train:   np.ndarray
    X_test:    np.ndarray
    y_train:   np.ndarray
    y_test:    np.ndarray
    weights:   np.ndarray       # sample weights for training
    feature_names: list[str]
    scaler:    StandardScaler   # fitted scaler — save for live inference
    n_train:   int = 0
    n_test:    int = 0


class Preprocessor:
    """
    Clean, scale and split feature matrices for ML training.
    Fits on training data, transforms both train and test consistently.
    """

    def __init__(self,
                 test_size: float = 0.2,
                 weight_decay: float = 0.9,
                 use_pca: bool = False,
                 pca_variance: float = 0.95):
        """
        test_size    : fraction of data held out for evaluation
        weight_decay : exponential decay for sample weights
                       0.9 = recent candles weighted ~10x more than old ones
        use_pca      : reduce feature dimensions to explain pca_variance of variance
        """
        self.test_size    = test_size
        self.weight_decay = weight_decay
        self.use_pca      = use_pca
        self.pca_variance = pca_variance
        self.scaler: StandardScaler | None = None
        self.pca:    PCA | None = None

    # ── Outlier clipping (IQR method) ─────────────────────────────────────────

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip each feature column at [Q1 - 3×IQR, Q3 + 3×IQR].
        More robust than z-score for fat-tailed financial data.
        """
        Q1  = df.quantile(0.25)
        Q3  = df.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        return df.clip(lower=lower, upper=upper, axis=1)

    # ── Sample weighting ──────────────────────────────────────────────────────

    def _compute_weights(self, n: int) -> np.ndarray:
        """
        Exponentially increasing weights — the most recent candle has weight 1.0,
        candles further back decay by `weight_decay` per step.
        Inspired by FreqAI's weight_factor concept.
        """
        indices = np.arange(n)
        weights = self.weight_decay ** (n - 1 - indices)
        weights = weights / weights.sum()   # normalise to sum=1
        return weights.astype(np.float32)

    # ── Dissimilarity score ───────────────────────────────────────────────────

    def dissimilarity_score(self, X_train: np.ndarray, X_live: np.ndarray) -> float:
        """
        Measure how different a live feature row is from the training distribution.

        Properly calibrated for high dimensions:
          1. Compute the per-row euclidean norm of every training row (z-scored)
          2. Compute the live row's norm the same way
          3. Return how many standard deviations the live norm is above the
             training mean norm, mapped to [0, 1] via /4 (so 4 sigma = 1.0)

        Inspired by FreqAI's dissimilarity index concept.
        Score > 1.0 means the live data is a strong outlier (unreliable).

        Note: X_train and X_live are ALREADY StandardScaler output, so we
        compute euclidean row-norms directly — no re-standardization (which
        would divide by ~0 for features constant in training and explode).
        """
        try:
            # Row norms in the already-scaled space
            train_norms = np.linalg.norm(X_train, axis=1)
            norm_mean   = train_norms.mean()
            norm_std    = train_norms.std() + 1e-8

            live_norm   = np.linalg.norm(X_live, axis=1).mean()

            # How many sigma above the typical training-row norm?
            sigma_away = (live_norm - norm_mean) / norm_std
            # Map: at/below mean → 0, 4 sigma above → 1.0
            return float(max(0.0, sigma_away / 4.0))
        except Exception:
            return 0.0

    # ── Main preprocessing pipeline ───────────────────────────────────────────

    def fit_transform(self, feature_df: pd.DataFrame,
                      targets: pd.Series) -> ProcessedData:
        """
        Full pipeline: clean → clip → split → weight → scale → (PCA).
        Fits scaler on training split only to prevent data leakage.
        """
        # Step 1: align features and targets, drop any remaining NaNs
        combined = feature_df.copy()
        combined["__target__"] = targets
        combined = combined.dropna()

        X = combined.drop("__target__", axis=1).astype(np.float32)
        y = combined["__target__"].values.astype(np.float32)
        feature_names = list(X.columns)

        if len(X) < 50:
            raise ValueError(f"Too few clean samples: {len(X)} (need ≥ 50)")

        # Step 2: clip outliers (fit on full data before splitting)
        X = self._clip_outliers(X)

        # Step 3: time-based train/test split (NO shuffle — preserves time order)
        split_idx = int(len(X) * (1 - self.test_size))
        X_train_raw = X.iloc[:split_idx].values
        X_test_raw  = X.iloc[split_idx:].values
        y_train     = y[:split_idx]
        y_test      = y[split_idx:]

        # Step 4: sample weights (exponential, only for training)
        weights = self._compute_weights(len(X_train_raw))

        # Step 5: StandardScaler fit on TRAINING only, transform both
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test  = self.scaler.transform(X_test_raw).astype(np.float32)

        # Step 6: optional PCA dimensionality reduction
        if self.use_pca and X_train.shape[1] > 20:
            self.pca = PCA(n_components=self.pca_variance, svd_solver="full")
            X_train  = self.pca.fit_transform(X_train).astype(np.float32)
            X_test   = self.pca.transform(X_test).astype(np.float32)
            logger.info("PCA reduced features: %d → %d (%.0f%% variance)",
                        len(feature_names), X_train.shape[1], self.pca_variance*100)

        logger.info("Preprocessed: %d train / %d test samples, %d features",
                    len(X_train), len(X_test), X_train.shape[1])

        return ProcessedData(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            weights=weights,
            feature_names=feature_names,
            scaler=self.scaler,
            n_train=len(X_train),
            n_test=len(X_test),
        )

    def transform_live(self, feature_df: pd.DataFrame) -> np.ndarray | None:
        """
        Transform a single live row (or small batch) using the fitted scaler.
        Returns None if scaler not fitted yet.
        """
        if self.scaler is None:
            return None
        try:
            X = feature_df.fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(np.float32)
            X_scaled = self.scaler.transform(X).astype(np.float32)
            # Clip scaled values to ±10 sigma — a single extreme live feature
            # must not be allowed to dominate distance / model calculations.
            X_scaled = np.clip(X_scaled, -10.0, 10.0)
            if self.pca is not None:
                X_scaled = self.pca.transform(X_scaled).astype(np.float32)
            return X_scaled
        except Exception as e:
            logger.warning("Live transform failed: %s", e)
            return None

    def save(self, path: str = SCALER_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "pca": self.pca}, f)

    def load(self, path: str = SCALER_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data.get("scaler")
        self.pca    = data.get("pca")
        return True
