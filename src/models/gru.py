"""
gru.py
======
Multi-layer GRU (Gated Recurrent Unit) for rainfall forecasting.

Architecture mirrors lstm.py exactly, with GRU replacing LSTM cells.
This is intentional: an identical architecture enables a fair empirical
comparison between LSTM and GRU on the Lucknow dataset.

GRU vs LSTM trade-off
---------------------
GRU uses two gates (update, reset) vs LSTM's three (input, forget, output).
- Fewer parameters → less overfitting risk on the ~6,600-row training set
- Comparable representational power for sequences with moderate memory depth
- Empirically, GRU often matches LSTM on meteorological time-series
  while training ~20-30% faster

The diagnostic pass showed lag-1 XGBoost residual ACF = 0.019, confirming
that the primary signal is in the raw rainfall sequence rather than in
long-range residual patterns.  GRU's lighter memory structure may be
better suited to this regime.

All hyperparameters, loss function, and training protocol are shared
with lstm.py via the WeightedMSELoss and the same training loop structure.
The only difference is the recurrent cell type.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path
from src.models.lstm import (
    WeightedMSELoss,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT, FC_HIDDEN_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, EARLY_STOP_PAT,
    LR_PATIENCE, LR_FACTOR, GRAD_CLIP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GRU network definition
# ---------------------------------------------------------------------------

class GRUNet(nn.Module):
    """
    Multi-layer GRU with input batch normalisation and fully-connected head.
    Architecture mirrors LSTMNet; only the recurrent cell differs.
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

        self.input_bn = nn.BatchNorm1d(n_features)

        # GRU: no cell state, two gates instead of three
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, lookback, n_features)

        Returns
        -------
        (batch,) — predicted log(1+rainfall)
        """
        batch, seq_len, n_feat = x.shape
        x_flat = x.reshape(batch * seq_len, n_feat)
        x_bn   = self.input_bn(x_flat)
        x      = x_bn.reshape(batch, seq_len, n_feat)

        # GRU returns (output, h_n); h_n shape: (num_layers, batch, hidden)
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]               # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# GRU trainer / wrapper  (mirrors LSTMModel interface exactly)
# ---------------------------------------------------------------------------

class GRUModel:
    """
    Training and inference wrapper for GRUNet.
    Interface is identical to LSTMModel for pipeline compatibility.
    """

    def __init__(
        self,
        n_features: int,
        lookback:   int,
        name:       str = "GRU",
    ) -> None:
        self.name       = name
        self.lookback   = lookback
        self.n_features = n_features
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net_           = GRUNet(n_features=n_features).to(self.device)
        self.loss_fn_       = WeightedMSELoss()
        self.history_:      Dict[str, List[float]] = {"train": [], "val": []}
        self.best_epoch_:   int   = 0
        self.best_val_loss_: float = np.inf

        logger.info(
            f"[{self.name}] Initialised on {self.device} | "
            f"n_features={n_features} | lookback={lookback}"
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> "GRUModel":
        optimiser = torch.optim.AdamW(
            self.net_.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
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
                best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
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
                    f"[{self.name}] Early stopping at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch_}"
                )
                break

        if best_state is not None:
            self.net_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self

    def _train_epoch(self, loader: DataLoader, optimiser) -> float:
        self.net_.train()
        total, n = 0.0, 0
        for x_b, y_b, w_b in loader:
            x_b, y_b, w_b = x_b.to(self.device), y_b.to(self.device), w_b.to(self.device)
            optimiser.zero_grad()
            pred = self.net_(x_b)
            loss = self.loss_fn_(pred, y_b, w_b)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net_.parameters(), GRAD_CLIP)
            optimiser.step()
            total += loss.item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.net_.eval()
        total, n = 0.0, 0
        for x_b, y_b, w_b in loader:
            x_b, y_b, w_b = x_b.to(self.device), y_b.to(self.device), w_b.to(self.device)
            pred  = self.net_(x_b)
            loss  = self.loss_fn_(pred, y_b, w_b)
            total += loss.item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def predict_log(self, X_seq: np.ndarray) -> np.ndarray:
        self.net_.eval()
        dataset = torch.tensor(X_seq, dtype=torch.float32)
        preds   = []
        for i in range(0, len(dataset), 512):
            batch = dataset[i: i + 512].to(self.device)
            preds.append(self.net_(batch).cpu().numpy())
        return np.concatenate(preds)

    def predict_mm(self, X_seq: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(self.predict_log(X_seq)), 0.0, None)

    def save(self, path: Optional[Path] = None) -> Path:
        out = path or abs_path(f"outputs/models/{self.name.lower()}_model.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":    self.net_.state_dict(),
            "n_features":    self.n_features,
            "lookback":      self.lookback,
            "history":       self.history_,
            "best_epoch":    self.best_epoch_,
            "best_val_loss": self.best_val_loss_,
            "name":          self.name,
        }, out)
        logger.info(f"[{self.name}] Model saved → {out}")
        return out

    @classmethod
    def load(cls, path: Path) -> "GRUModel":
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj  = cls(n_features=data["n_features"], lookback=data["lookback"],
                   name=data.get("name", "GRU"))
        obj.net_.load_state_dict(data["state_dict"])
        obj.history_       = data["history"]
        obj.best_epoch_    = data["best_epoch"]
        obj.best_val_loss_ = data["best_val_loss"]
        return obj
