"""
sequence_generator.py
=====================
Sliding-window sequence generation for LSTM and GRU models.

Temporal integrity guarantee
-----------------------------
All sequences are constructed such that sequence[i] uses only observations
from timesteps ≤ t.  The target y[i] corresponds to timestep t+1 (one-step-
ahead prediction).  No future information is ever included in any window.

Window construction
-------------------
For a lookback window of W days and time index t:
  X[i] = feature_matrix[t-W : t]     shape (W, n_features)
  y[i] = target[t]                    scalar

The first valid sequence index is W (the first t where t-W ≥ 0).
Windows that would extend into a different split are never created;
the split boundary is enforced before any sequence extraction.

Two lookback windows
--------------------
W=30 days — matches the lag feature window; serves as the baseline.
W=60 days — allows the LSTM to capture monsoon onset dynamics
             that unfold over ~4-6 weeks before the first heavy rains.

Weighted loss support
---------------------
Each sequence carries a sample weight derived from the target value.
Days where actual rainfall > 20 mm receive weight = HEAVY_RAIN_WEIGHT (5×).
This weight is passed to the PyTorch DataLoader and applied in the loss
computation to force the model to learn extreme-event patterns.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOOKBACK_SHORT: int   = 30     # days — baseline window
LOOKBACK_LONG:  int   = 60     # days — extended monsoon window
HEAVY_RAIN_THRESHOLD: float = 20.0   # mm — weight escalation threshold
HEAVY_RAIN_WEIGHT:    float = 5.0    # multiplier for heavy-rain samples
BATCH_SIZE:           int   = 128
NUM_WORKERS:          int   = 0      # avoid multiprocessing issues


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class RainfallSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping a sliding-window feature matrix and targets.

    Parameters
    ----------
    X         : Feature array, shape (T, n_features), already scaled.
    y         : Target array, shape (T,).
    weights   : Sample weight array, shape (T,).
    lookback  : Number of timesteps per window.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        lookback: int,
    ) -> None:
        self.X        = torch.tensor(X, dtype=torch.float32)
        self.y        = torch.tensor(y, dtype=torch.float32)
        self.weights  = torch.tensor(weights, dtype=torch.float32)
        self.lookback = lookback
        # Valid indices: from lookback to len(y) (exclusive)
        # At index i: window = X[i-lookback : i], target = y[i]
        self.valid_start = lookback

    def __len__(self) -> int:
        return len(self.y) - self.lookback

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t       = idx + self.lookback          # actual timestep
        x_seq   = self.X[t - self.lookback: t] # (lookback, n_features)
        target  = self.y[t]                    # scalar
        weight  = self.weights[t]              # scalar
        return x_seq, target, weight


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

class SequenceGenerator:
    """
    Builds LSTM/GRU-ready DataLoaders for train, val, and test splits.

    Parameters
    ----------
    lookback   : Window size in days (30 or 60).
    target_col : Column to predict.  'LOG_RAINFALL' for direct prediction,
                 'SARIMAX_residual' for hybrid residual learning.
    batch_size : Mini-batch size for training DataLoader.
    """

    def __init__(
        self,
        lookback:   int   = LOOKBACK_LONG,
        target_col: str   = "LOG_RAINFALL",
        batch_size: int   = BATCH_SIZE,
    ) -> None:
        self.lookback   = lookback
        self.target_col = target_col
        self.batch_size = batch_size
        self.feature_cols_: List[str] = []
        self.n_features_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  pd.DataFrame,
        residuals: Optional[pd.Series] = None,
    ) -> Dict[str, DataLoader]:
        """
        Build DataLoaders for all three splits.

        Parameters
        ----------
        train_df   : Scaled ML feature set, training split.
        val_df     : Scaled ML feature set, validation split.
        test_df    : Scaled ML feature set, test split.
        residuals  : SARIMAX training residuals (log scale).  When provided
                     and target_col starts with 'residual', this overrides
                     the y target for the training split.

        Returns
        -------
        dict with keys 'train', 'val', 'test' → DataLoader
        Also sets self.feature_cols_, self.n_features_.
        """
        # Feature columns: numeric, exclude targets and non-predictive cols
        exclude = {
            "RAINFALL", "LOG_RAINFALL", "RAIN_OCCURRENCE",
            "RAINFALL_WET_ONLY", "SPLIT",
        }
        self.feature_cols_ = [
            c for c in train_df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(train_df[c])
        ]
        self.n_features_ = len(self.feature_cols_)

        logger.info(
            f"[SequenceGenerator] lookback={self.lookback} | "
            f"target='{self.target_col}' | features={self.n_features_}"
        )

        loaders = {}
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            X, y, w = self._extract_arrays(df, split_name, residuals)
            ds = RainfallSequenceDataset(X, y, w, self.lookback)
            shuffle = (split_name == "train")
            loaders[split_name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,   # NEVER shuffle — preserve temporal order
                num_workers=NUM_WORKERS,
                drop_last=False,
            )
            logger.info(
                f"  {split_name}: {len(ds):,} sequences "
                f"({len(ds) // self.batch_size + 1} batches) | "
                f"heavy-rain samples: {(w[self.lookback:] > 1.0).sum():,}"
            )

        return loaders

    def get_arrays(
        self,
        df: pd.DataFrame,
        split_name: str = "test",
        residuals: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X_sequences, y_targets, sample_weights) as numpy arrays.
        Used for evaluation without DataLoader overhead.
        """
        X_flat, y_flat, w_flat = self._extract_arrays(df, split_name, residuals)
        n = len(y_flat) - self.lookback
        X_seq = np.stack([
            X_flat[i: i + self.lookback]
            for i in range(n)
        ], axis=0)                         # (n, lookback, n_features)
        y_out = y_flat[self.lookback:]    # (n,)
        w_out = w_flat[self.lookback:]    # (n,)
        return X_seq, y_out, w_out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _extract_arrays(
        self,
        df: pd.DataFrame,
        split_name: str,
        residuals: Optional[pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract feature matrix, target vector, and weight vector."""
        X = df[self.feature_cols_].values.astype(np.float32)

        # Target selection
        if self.target_col == "LOG_RAINFALL":
            y_raw_mm = df["RAINFALL"].values.astype(np.float32)
            y = np.log1p(y_raw_mm)
        elif self.target_col == "SARIMAX_residual" and split_name == "train":
            # Use SARIMAX residuals as target for hybrid model
            if residuals is None:
                raise ValueError(
                    "residuals must be provided when target_col='SARIMAX_residual'"
                )
            # Align residuals to df index
            aligned = residuals.reindex(df.index)
            y = aligned.values.astype(np.float32)
            y_raw_mm = df["RAINFALL"].values.astype(np.float32)
        else:
            # For val/test in hybrid mode: use actual log-rainfall
            # (we evaluate the hybrid as a whole, not the residual alone)
            y_raw_mm = df["RAINFALL"].values.astype(np.float32)
            y = np.log1p(y_raw_mm)

        # Sample weights: escalate for heavy-rain days
        weights = _compute_sample_weights(
            df["RAINFALL"].values.astype(np.float32),
            threshold=HEAVY_RAIN_THRESHOLD,
            heavy_weight=HEAVY_RAIN_WEIGHT,
        )

        return X, y, weights


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def _compute_sample_weights(
    rainfall: np.ndarray,
    threshold: float = HEAVY_RAIN_THRESHOLD,
    heavy_weight: float = HEAVY_RAIN_WEIGHT,
) -> np.ndarray:
    """
    Return per-sample weights.

    Weight = heavy_weight if rainfall > threshold, else 1.0.
    This biases the loss toward heavy-rain events without discarding
    the dry-day signal that anchors the classifier.
    """
    weights = np.ones_like(rainfall, dtype=np.float32)
    weights[rainfall > threshold] = heavy_weight
    return weights
