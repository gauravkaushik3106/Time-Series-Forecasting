"""
hybrid_model.py
===============
Hybrid SARIMAX + LSTM residual model.

Architecture
------------
  SARIMAX prediction (saved from Phase 4)
      ↓
  LSTM trained on SARIMAX training residuals
      ↓
  LSTM predicted residual
      ↓
  Final prediction = SARIMAX prediction + LSTM residual prediction
                     (back-transformed to mm/day scale)

Design rationale
----------------
The Phase 4 diagnostic pass showed:
  - SARIMAX residual lag-1 ACF = 0.117 (above significance threshold)
  - XGBoost residual lag-1 ACF = 0.019 (effectively zero)

This confirms that SARIMAX leaves exploitable temporal structure in its
residuals — structure that a sequence model can learn.  Rather than
training a standalone LSTM on raw rainfall (which would compete directly
with the XGBoost on the zero-inflation problem), the hybrid LSTM's sole
task is to predict what SARIMAX got wrong.

Residual target
---------------
The LSTM is trained on log-scale SARIMAX residuals:
  residual_t = log(1 + actual_t) − log(1 + SARIMAX_prediction_t)

This keeps the target bounded and approximately symmetric, which suits
the LSTM's MSE objective better than raw-scale residuals.

At inference time, the predicted residual is added back on the log scale
before expm1 back-transformation:
  y_hybrid = expm1( log(1 + y_sarimax) + residual_hat )
           = expm1( log_sarimax_pred + residual_hat )

This means the LSTM correction can both increase and decrease the SARIMAX
prediction — it learns both the underprediction bias during monsoon onset
and the overprediction during dry spells.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path
from src.models.lstm import LSTMModel
from src.models.sequence_generator import SequenceGenerator, LOOKBACK_LONG

logger = logging.getLogger(__name__)


class HybridSARIMAXLSTM:
    """
    SARIMAX + LSTM residual hybrid model.

    Parameters
    ----------
    lookback : Sequence window for the LSTM component.
    """

    name: str = "Hybrid_SARIMAX_LSTM"

    def __init__(self, lookback: int = LOOKBACK_LONG) -> None:
        self.lookback       = lookback
        self.lstm_model_:   Optional[LSTMModel] = None
        self.seq_gen_:      Optional[SequenceGenerator] = None
        self._train_index:  Optional[pd.DatetimeIndex] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_ml:  pd.DataFrame,
        val_ml:    pd.DataFrame,
        sarimax_train_residuals: pd.Series,
        sarimax_val_predictions: pd.DataFrame,
    ) -> "HybridSARIMAXLSTM":
        """
        Train the LSTM component on SARIMAX residuals.

        Parameters
        ----------
        train_ml                 : Scaled ML feature set, training split.
        val_ml                   : Scaled ML feature set, validation split.
        sarimax_train_residuals  : Series of SARIMAX in-sample residuals
                                   (log scale) on the training set.
        sarimax_val_predictions  : DataFrame with [actual, predicted] for
                                   the val split from SARIMAX.
        """
        # Build log-scale residuals for training
        # residual_t = log(1 + actual_t) − log(1 + sarimax_pred_t)
        # We have the raw SARIMAX residuals from Phase 4; use them directly.
        train_residuals_log = sarimax_train_residuals.copy()

        # Build log-scale validation residuals
        val_actual_log = np.log1p(sarimax_val_predictions["actual"].values)
        val_pred_log   = np.log1p(
            np.clip(sarimax_val_predictions["predicted"].values, 0.0, None)
        )
        val_residuals_log = pd.Series(
            val_actual_log - val_pred_log,
            index=sarimax_val_predictions.index,
            name="SARIMAX_val_residuals",
        )

        logger.info(
            f"[{self.name}] Training residuals: "
            f"mean={train_residuals_log.mean():.4f}, "
            f"std={train_residuals_log.std():.4f}"
        )

        # Build sequences with residual target
        self.seq_gen_ = SequenceGenerator(
            lookback=self.lookback,
            target_col="SARIMAX_residual",
        )

        # Augment val_ml with residual target
        val_ml_aug = val_ml.copy()
        val_ml_aug["SARIMAX_residual"] = val_residuals_log.reindex(val_ml.index).values

        # Override target for val DataLoader
        val_ml_aug["LOG_RAINFALL"] = val_residuals_log.reindex(val_ml.index).values

        # Build DataLoaders: train target = sarimax residuals, val target = val residuals
        # We temporarily reassign LOG_RAINFALL for the val to residuals
        train_ml_aug = train_ml.copy()

        loaders = self.seq_gen_.build(
            train_df=train_ml_aug,
            val_df=val_ml_aug,
            test_df=val_ml_aug,   # placeholder; not used
            residuals=train_residuals_log,
        )

        # Initialise LSTM
        self.lstm_model_ = LSTMModel(
            n_features=self.seq_gen_.n_features_,
            lookback=self.lookback,
            name="Hybrid_LSTM_residual",
        )

        self.lstm_model_.fit(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
        )

        self._train_index = train_ml.index
        logger.info(f"[{self.name}] Training complete")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        ml_df:             pd.DataFrame,
        sarimax_pred_df:   pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate hybrid predictions for a split.

        Parameters
        ----------
        ml_df           : Scaled ML feature set for the target split.
        sarimax_pred_df : DataFrame with [actual, predicted] from SARIMAX.

        Returns
        -------
        DataFrame with columns [actual, predicted, sarimax_pred, lstm_residual].
        """
        if self.lstm_model_ is None or self.seq_gen_ is None:
            raise RuntimeError("Call .fit() before .predict()")

        # Build sequences from the CONTINUOUS series
        # We need lookback context before the split.
        # The full_features.parquet holds the entire time series; use it.
        X_seq, _, _ = self.seq_gen_.get_arrays(ml_df)

        # Predict LSTM residual (log scale)
        lstm_residual_log = self.lstm_model_.predict_log(X_seq)

        # Align: the first `lookback` rows have no sequence → pad with 0.0
        n_total     = len(ml_df)
        n_seq       = len(lstm_residual_log)
        n_pad       = n_total - n_seq

        residual_full = np.concatenate([
            np.zeros(n_pad, dtype=np.float32),
            lstm_residual_log,
        ])

        # SARIMAX predictions (already in mm/day)
        sarimax_mm = sarimax_pred_df["predicted"].reindex(ml_df.index).values
        sarimax_mm = np.clip(sarimax_mm, 0.0, None)

        # Combine on log scale
        log_sarimax = np.log1p(sarimax_mm)
        log_hybrid  = log_sarimax + residual_full
        hybrid_mm   = np.clip(np.expm1(log_hybrid), 0.0, None)

        actual_mm   = sarimax_pred_df["actual"].reindex(ml_df.index).values

        return pd.DataFrame({
            "actual":         actual_mm,
            "predicted":      hybrid_mm,
            "sarimax_pred":   sarimax_mm,
            "lstm_residual":  residual_full,
        }, index=ml_df.index)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self) -> None:
        out_dir = abs_path("outputs/models")
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.lstm_model_ is not None:
            self.lstm_model_.save(out_dir / "hybrid_lstm_residual.pt")
        logger.info(f"[{self.name}] Saved LSTM component")

    @classmethod
    def load(cls, lookback: int = LOOKBACK_LONG) -> "HybridSARIMAXLSTM":
        obj = cls(lookback=lookback)
        lstm_path = abs_path("outputs/models/hybrid_lstm_residual.pt")
        obj.lstm_model_ = LSTMModel.load(lstm_path)
        return obj


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_hybrid(
    train_ml:   pd.DataFrame,
    val_ml:     pd.DataFrame,
    test_ml:    pd.DataFrame,
    sarimax_train_residuals: pd.Series,
    sarimax_val:  pd.DataFrame,
    sarimax_test: pd.DataFrame,
    lookback:   int = LOOKBACK_LONG,
) -> Tuple["HybridSARIMAXLSTM", Dict[str, pd.DataFrame]]:
    """
    Fit the hybrid model and generate val/test predictions.

    Returns (fitted_model, predictions_dict).
    """
    model = HybridSARIMAXLSTM(lookback=lookback)
    model.fit(
        train_ml=train_ml,
        val_ml=val_ml,
        sarimax_train_residuals=sarimax_train_residuals,
        sarimax_val_predictions=sarimax_val,
    )
    model.save()

    out_dir = abs_path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for split_name, ml_split, sarimax_split in [
        ("val",  val_ml,  sarimax_val),
        ("test", test_ml, sarimax_test),
    ]:
        pred_df = model.predict(ml_split, sarimax_split)
        pred_df.to_parquet(out_dir / f"hybrid_{split_name}.parquet")
        predictions[split_name] = pred_df
        logger.info(
            f"[Hybrid] {split_name} predictions saved ({len(pred_df):,} rows)"
        )

    return model, predictions
