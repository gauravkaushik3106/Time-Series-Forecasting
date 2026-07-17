"""
mc_dropout.py
=============
Monte Carlo (MC) Dropout for epistemic uncertainty estimation on the GRU model.

Theory
------
At training time, Dropout randomly zeros neuron activations with probability p.
At test time, standard inference DISABLES dropout (deterministic forward pass).

MC Dropout (Gal & Ghahramani, 2016) re-enables dropout at test time and runs
T stochastic forward passes through the same network.  The resulting T
predictions form an approximate posterior predictive distribution:

  p(y* | x*, X_train) ≈ (1/T) Σ_{t=1}^{T} p(y* | x*, ŵ_t)

where ŵ_t are the weights sampled by the t-th dropout mask.

Why GRU and not XGBoost?
------------------------
XGBoost is a deterministic tree ensemble — there is no natural dropout
mechanism.  The GRU has explicit dropout layers trained to regularise the
network; activating them at inference time provides a principled Bayesian
approximation.

Outputs per prediction
----------------------
  - mean:  point estimate (mean of T samples)
  - std:   epistemic uncertainty (std dev of T samples)
  - lower: lower bound of (1-α) prediction interval
  - upper: upper bound of (1-α) prediction interval

Coverage guarantee
------------------
MC Dropout intervals are NOT guaranteed to have exact frequentist coverage —
they are approximate.  Calibration analysis (calibration.py) measures the
actual empirical coverage and compares it to nominal levels.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path
from src.models.gru import GRUModel
from src.models.sequence_generator import SequenceGenerator, LOOKBACK_LONG

logger = logging.getLogger(__name__)

MC_SAMPLES:    int   = 100    # T forward passes per prediction
DROPOUT_RATE:  float = 0.30   # must match training dropout
ALPHA_LEVELS         = [0.10, 0.20, 0.30, 0.40]  # 90%, 80%, 70%, 60% intervals


# ---------------------------------------------------------------------------
# Activate dropout at inference time
# ---------------------------------------------------------------------------

def _enable_dropout(model: nn.Module) -> None:
    """
    Re-enable dropout layers for MC inference.
    PyTorch's model.eval() disables BatchNorm updates AND Dropout.
    We selectively re-enable only Dropout.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

class MCDropoutPredictor:
    """
    Wraps a trained GRUModel to perform MC Dropout inference.

    Parameters
    ----------
    gru_model   : Fitted GRUModel instance.
    n_samples   : Number of stochastic forward passes (T).
    """

    def __init__(
        self,
        gru_model: GRUModel,
        n_samples: int = MC_SAMPLES,
    ) -> None:
        self.gru    = gru_model
        self.T      = n_samples
        self.device = gru_model.device

    @torch.no_grad()
    def predict_distribution(
        self,
        X_seq: np.ndarray,
        batch_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """
        Run T stochastic forward passes and return full sample distribution.

        Parameters
        ----------
        X_seq      : Sequence array (N, lookback, n_features).
        batch_size : Mini-batch size for efficient GPU/CPU throughput.

        Returns
        -------
        dict with keys:
          'samples'     : (T, N) raw log-scale predictions
          'mean_log'    : (N,) mean log-scale prediction
          'std_log'     : (N,) std dev of log-scale predictions
          'mean_mm'     : (N,) back-transformed mean (mm/day)
          'std_mm'      : (N,) back-transformed std (mm/day)
          'lower_<p>'   : (N,) lower bound at each alpha level
          'upper_<p>'   : (N,) upper bound at each alpha level
        """
        # Set model to eval (disables BatchNorm update) then re-enable Dropout
        self.gru.net_.eval()
        _enable_dropout(self.gru.net_)

        dataset = torch.tensor(X_seq, dtype=torch.float32)
        N = len(dataset)

        # Collect T samples: shape (T, N)
        all_samples = np.zeros((self.T, N), dtype=np.float32)

        for t in range(self.T):
            preds = []
            for i in range(0, N, batch_size):
                batch = dataset[i: i + batch_size].to(self.device)
                out   = self.gru.net_(batch).cpu().numpy()
                preds.append(out)
            all_samples[t] = np.concatenate(preds)

        # Reset: disable dropout again
        self.gru.net_.eval()

        mean_log = all_samples.mean(axis=0)
        std_log  = all_samples.std(axis=0)

        result = {
            "samples":  all_samples,
            "mean_log": mean_log,
            "std_log":  std_log,
            "mean_mm":  np.clip(np.expm1(mean_log), 0.0, None),
            "std_mm":   np.clip(np.expm1(mean_log + std_log) - np.expm1(mean_log), 0.0, None),
        }

        # Prediction intervals at multiple alpha levels
        for alpha in ALPHA_LEVELS:
            lo_pct = 100 * alpha / 2
            hi_pct = 100 * (1 - alpha / 2)
            lo_log = np.percentile(all_samples, lo_pct, axis=0)
            hi_log = np.percentile(all_samples, hi_pct, axis=0)
            result[f"lower_{int((1-alpha)*100)}"] = np.clip(np.expm1(lo_log), 0.0, None)
            result[f"upper_{int((1-alpha)*100)}"] = np.clip(np.expm1(hi_log), 0.0, None)

        return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_mc_dropout(
    test_ml:   pd.DataFrame,
    train_ml:  pd.DataFrame,
    val_ml:    pd.DataFrame,
    n_samples: int = MC_SAMPLES,
    save: bool = True,
) -> Tuple[MCDropoutPredictor, pd.DataFrame]:
    """
    Load the trained GRU, run MC Dropout on the test set, and return
    a prediction DataFrame with mean, std, and interval columns.

    Parameters
    ----------
    test_ml  : Scaled ML feature set, test split.
    train_ml : Training split (for sequence context).
    val_ml   : Validation split.
    n_samples: Number of MC forward passes.

    Returns
    -------
    (predictor, prediction_df)
    """
    # Load GRU
    gru_path  = abs_path("outputs/models/gru_model.pt")
    gru_model = GRUModel.load(gru_path)
    logger.info(
        f"[MC Dropout] GRU loaded | T={n_samples} stochastic passes | "
        f"dropout_rate={DROPOUT_RATE}"
    )

    # Build sequences
    seq_gen = SequenceGenerator(lookback=LOOKBACK_LONG, target_col="LOG_RAINFALL")
    seq_gen.build(train_ml, val_ml, test_ml)
    X_seq, y_true, _ = seq_gen.get_arrays(test_ml, split_name="test")
    valid_index = test_ml.index[LOOKBACK_LONG:]

    logger.info(f"[MC Dropout] Running on {len(X_seq):,} test sequences")

    predictor = MCDropoutPredictor(gru_model, n_samples=n_samples)
    dist = predictor.predict_distribution(X_seq)

    actual_mm  = np.expm1(y_true)
    pred_df = pd.DataFrame(
        {
            "actual":   actual_mm,
            "mean_mm":  dist["mean_mm"],
            "std_mm":   dist["std_mm"],
            "mean_log": dist["mean_log"],
            "std_log":  dist["std_log"],
        },
        index=valid_index,
    )

    # Add interval columns
    for alpha in ALPHA_LEVELS:
        level = int((1 - alpha) * 100)
        pred_df[f"lower_{level}"] = dist[f"lower_{level}"]
        pred_df[f"upper_{level}"] = dist[f"upper_{level}"]

    if save:
        out_dir = abs_path("outputs/uncertainty")
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(out_dir / "mc_dropout_predictions.parquet")
        logger.info("[MC Dropout] Predictions saved")

    return predictor, pred_df
