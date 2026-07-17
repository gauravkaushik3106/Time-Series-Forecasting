"""
lstm.py
=======
Multi-layer LSTM for rainfall forecasting.

Architecture
------------
  Input  →  [BatchNorm1d on features]
         →  LSTM (num_layers=2, hidden_size=128, dropout=0.3)
         →  Dropout(0.3) on final hidden state
         →  Linear(hidden → 64) → ReLU → Linear(64 → 1)
         →  Output: log(1 + rainfall) scalar

The final hidden state (last timestep) is used for prediction, not
the full sequence output.  This is appropriate for one-step-ahead
forecasting where only the compressed sequence summary is needed.

Loss function: Weighted MSE
---------------------------
  loss = mean( weight_i × (pred_i − target_i)² )

where weight_i = HEAVY_RAIN_WEIGHT (5.0) if actual_rainfall_i > 20 mm,
else 1.0.  This directly addresses the diagnostic finding that all models
fail catastrophically on >20 mm events.

Training protocol
-----------------
  1. AdamW optimiser with weight decay = 1e-4
  2. ReduceLROnPlateau scheduler (factor=0.5, patience=5)
  3. Early stopping on validation weighted-MSE (patience=15 epochs)
  4. Gradient clipping (max_norm=1.0) to prevent exploding gradients
     during monsoon-onset sequences where targets spike rapidly

Batch normalisation is applied to the INPUT features (not recurrent
states) so that the LSTM sees normalised inputs independent of whether
the external StandardScaler was trained on the same distribution as
the current window.  This improves generalisation to out-of-sample
seasonal regimes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LSTM_HIDDEN_SIZE:  int   = 128
LSTM_NUM_LAYERS:   int   = 2
LSTM_DROPOUT:      float = 0.30
FC_HIDDEN_SIZE:    int   = 64
LEARNING_RATE:     float = 1e-3
WEIGHT_DECAY:      float = 1e-4
MAX_EPOCHS:        int   = 100
EARLY_STOP_PAT:    int   = 15
LR_PATIENCE:       int   = 5
LR_FACTOR:         float = 0.5
GRAD_CLIP:         float = 1.0


# ---------------------------------------------------------------------------
# Neural network definition
# ---------------------------------------------------------------------------

class LSTMNet(nn.Module):
    """
    Multi-layer LSTM with input normalisation and fully-connected head.

    Parameters
    ----------
    n_features   : Number of input features per timestep.
    hidden_size  : LSTM hidden state dimension.
    num_layers   : Number of stacked LSTM layers.
    dropout      : Dropout applied between LSTM layers and on final state.
    fc_hidden    : Intermediate fully-connected layer dimension.
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int   = LSTM_HIDDEN_SIZE,
        num_layers:  int   = LSTM_NUM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
        fc_hidden:   int   = FC_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Input batch normalisation — stabilises feature scale across sequences
        self.input_bn = nn.BatchNorm1d(n_features)

        # LSTM: processes the (lookback, n_features) sequence
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,    # input shape: (batch, seq, features)
        )

        # Dropout on the final hidden state before the FC head
        self.dropout = nn.Dropout(dropout)

        # Fully-connected prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, lookback, n_features)

        Returns
        -------
        Tensor of shape (batch,) — predicted log(1+rainfall) per sequence.
        """
        # Apply batch norm across the feature dimension
        # BN1d expects (batch, features) — apply to each timestep
        batch, seq_len, n_feat = x.shape
        x_flat = x.reshape(batch * seq_len, n_feat)
        x_bn   = self.input_bn(x_flat)
        x      = x_bn.reshape(batch, seq_len, n_feat)

        # LSTM forward
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers, batch, hidden_size)
        # Use the last layer's final hidden state
        last_hidden = h_n[-1]                    # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)

        # Prediction head → squeeze to (batch,)
        out = self.fc(last_hidden).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# Weighted MSE loss
# ---------------------------------------------------------------------------

class WeightedMSELoss(nn.Module):
    """MSE loss with per-sample weights."""

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        sq_err = (pred - target) ** 2
        return (sq_err * weight).mean()


# ---------------------------------------------------------------------------
# LSTM trainer / wrapper
# ---------------------------------------------------------------------------

class LSTMModel:
    """
    Training and inference wrapper for LSTMNet.

    Parameters
    ----------
    n_features  : Input feature count (from SequenceGenerator).
    lookback    : Sequence window length.
    name        : Model label used in logging and file naming.
    """

    def __init__(
        self,
        n_features: int,
        lookback:   int,
        name:       str = "LSTM",
    ) -> None:
        self.name       = name
        self.lookback   = lookback
        self.n_features = n_features
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net_        = LSTMNet(n_features=n_features).to(self.device)
        self.loss_fn_    = WeightedMSELoss()
        self.history_:   Dict[str, List[float]] = {"train": [], "val": []}
        self.best_epoch_: int  = 0
        self.best_val_loss_: float = np.inf

        logger.info(
            f"[{self.name}] Initialised on {self.device} | "
            f"n_features={n_features} | lookback={lookback}"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> "LSTMModel":
        """Train with early stopping and learning-rate scheduling."""
        optimiser = torch.optim.AdamW(
            self.net_.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
        )

        best_state = None
        no_improve = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = self._train_epoch(train_loader, optimiser)
            val_loss   = self._eval_epoch(val_loader)
            scheduler.step(val_loss)

            self.history_["train"].append(train_loss)
            self.history_["val"].append(val_loss)

            if val_loss < self.best_val_loss_ - 1e-6:
                self.best_val_loss_ = val_loss
                self.best_epoch_    = epoch
                best_state          = {
                    k: v.cpu().clone() for k, v in self.net_.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch <= 3:
                lr_now = optimiser.param_groups[0]["lr"]
                logger.info(
                    f"[{self.name}] Epoch {epoch:3d}/{MAX_EPOCHS} | "
                    f"train={train_loss:.5f} | val={val_loss:.5f} | "
                    f"lr={lr_now:.2e} | best_ep={self.best_epoch_}"
                )

            if no_improve >= EARLY_STOP_PAT:
                logger.info(
                    f"[{self.name}] Early stopping at epoch {epoch} "
                    f"(no improvement for {EARLY_STOP_PAT} epochs). "
                    f"Best epoch: {self.best_epoch_}, val_loss: {self.best_val_loss_:.5f}"
                )
                break

        # Restore best weights
        if best_state is not None:
            self.net_.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        return self

    def _train_epoch(
        self,
        loader:    DataLoader,
        optimiser: torch.optim.Optimizer,
    ) -> float:
        self.net_.train()
        total_loss = 0.0
        n_batches  = 0
        for x_batch, y_batch, w_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            w_batch = w_batch.to(self.device)
            optimiser.zero_grad()
            pred  = self.net_(x_batch)
            loss  = self.loss_fn_(pred, y_batch, w_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net_.parameters(), GRAD_CLIP)
            optimiser.step()
            total_loss += loss.item()
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.net_.eval()
        total_loss = 0.0
        n_batches  = 0
        for x_batch, y_batch, w_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            w_batch = w_batch.to(self.device)
            pred  = self.net_(x_batch)
            loss  = self.loss_fn_(pred, y_batch, w_batch)
            total_loss += loss.item()
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_log(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Predict log(1+rainfall) from pre-built sequence array.

        Parameters
        ----------
        X_seq : np.ndarray of shape (N, lookback, n_features).

        Returns
        -------
        np.ndarray of shape (N,) — predicted log(1+rainfall).
        """
        self.net_.eval()
        dataset = torch.tensor(X_seq, dtype=torch.float32)
        preds   = []
        batch_size = 512
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size].to(self.device)
            preds.append(self.net_(batch).cpu().numpy())
        return np.concatenate(preds)

    def predict_mm(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict in mm/day (back-transformed from log scale), clipped ≥ 0."""
        log_preds = self.predict_log(X_seq)
        return np.clip(np.expm1(log_preds), 0.0, None)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        out = path or abs_path(f"outputs/models/{self.name.lower()}_model.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":     self.net_.state_dict(),
            "n_features":     self.n_features,
            "lookback":       self.lookback,
            "history":        self.history_,
            "best_epoch":     self.best_epoch_,
            "best_val_loss":  self.best_val_loss_,
            "name":           self.name,
        }, out)
        logger.info(f"[{self.name}] Model saved → {out}")
        return out

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj  = cls(
            n_features=data["n_features"],
            lookback=data["lookback"],
            name=data.get("name", "LSTM"),
        )
        obj.net_.load_state_dict(data["state_dict"])
        obj.history_       = data["history"]
        obj.best_epoch_    = data["best_epoch"]
        obj.best_val_loss_ = data["best_val_loss"]
        logger.info(f"[{obj.name}] Model loaded from {path}")
        return obj
