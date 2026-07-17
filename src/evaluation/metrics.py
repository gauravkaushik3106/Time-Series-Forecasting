"""
metrics.py
==========
Evaluation metrics for rainfall forecasting models.

Metric catalogue
----------------
RMSE  — Root Mean Squared Error.  Penalises large errors quadratically;
        sensitive to extreme rainfall events, which is appropriate for
        hydrological applications where under-prediction of floods matters.

MAE   — Mean Absolute Error.  Robust to outliers; interpretable in mm/day.

R²    — Coefficient of determination.  Fraction of variance explained.
        Note: for a highly zero-inflated target, R² can be misleadingly high
        even for naive models; always read alongside RMSE and NSE.

MAPE  — Mean Absolute Percentage Error.  Computed only on WET days (rainfall
        > threshold) because MAPE is undefined/infinite when actuals are zero.

NSE   — Nash–Sutcliffe Efficiency.  Standard benchmark metric in hydrology.
        NSE = 1 − (MSE_model / MSE_climatology).
        NSE = 1.0 → perfect; NSE = 0.0 → model = mean; NSE < 0 → worse than mean.

All metrics are computed on the ORIGINAL rainfall scale (mm/day), never on
log-transformed values, so results are physically interpretable.

Breakdowns
----------
- Overall: full test period
- Monsoon (JJAS): Jun–Sep — the high-signal, high-stakes season
- Non-monsoon: Oct–May — the dry baseline season
- Extreme events: days where observed rainfall ≥ 50 mm
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DRY_THRESHOLD: float    = 0.1    # mm — consistent with config.rainfall.dry_day_threshold
EXTREME_THRESHOLD: float = 50.0  # mm — heavy+ rain per IMD
MONSOON_MONTHS: tuple   = (6, 7, 8, 9)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mape_wet(actual: np.ndarray, predicted: np.ndarray,
             threshold: float = DRY_THRESHOLD) -> float:
    """MAPE computed only on wet days to avoid division-by-zero."""
    wet = actual > threshold
    if wet.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[wet] - predicted[wet]) / actual[wet])) * 100)


def nse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Nash–Sutcliffe Efficiency."""
    mean_actual = actual.mean()
    numerator   = np.sum((actual - predicted) ** 2)
    denominator = np.sum((actual - mean_actual) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean bias (predicted − actual). Positive = over-prediction."""
    return float(np.mean(predicted - actual))


def hit_rate(actual: np.ndarray, predicted: np.ndarray,
             threshold: float = DRY_THRESHOLD) -> float:
    """Proportion of wet days correctly predicted as wet."""
    actual_wet    = actual > threshold
    predicted_wet = predicted > threshold
    if actual_wet.sum() == 0:
        return float("nan")
    return float((actual_wet & predicted_wet).sum() / actual_wet.sum())


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def evaluate(
    actual: pd.Series,
    predicted: pd.Series,
    model_name: str = "model",
    index: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float | str]:
    """
    Compute the complete metric suite, with seasonal and extreme breakdowns.

    Parameters
    ----------
    actual      : Observed rainfall (mm/day), original scale.
    predicted   : Model predictions (mm/day), original scale.
    model_name  : Name embedded in the result dict for table construction.
    index       : DatetimeIndex for seasonal slicing.  If None, uses
                  actual.index if it is a DatetimeIndex.

    Returns
    -------
    dict with keys: model, RMSE, MAE, R2, MAPE_wet, NSE, Bias,
    HitRate, and seasonal/extreme variants.
    """
    # Clip negative predictions to zero (physically meaningful floor)
    act  = np.array(actual, dtype=float)
    pred = np.clip(np.array(predicted, dtype=float), 0.0, None)

    idx = index if index is not None else (
        actual.index if isinstance(actual.index, pd.DatetimeIndex) else None
    )

    result: Dict[str, float | str] = {
        "Model":    model_name,
        "RMSE":     rmse(act, pred),
        "MAE":      mae(act, pred),
        "R2":       r_squared(act, pred),
        "MAPE_wet": mape_wet(act, pred),
        "NSE":      nse(act, pred),
        "Bias":     bias(act, pred),
        "HitRate":  hit_rate(act, pred),
    }

    # --- Seasonal breakdown ---
    if idx is not None:
        months = pd.DatetimeIndex(idx).month

        for season_label, mask_fn in [
            ("Monsoon",     lambda m: np.isin(m, list(MONSOON_MONTHS))),
            ("NonMonsoon",  lambda m: ~np.isin(m, list(MONSOON_MONTHS))),
        ]:
            mask = mask_fn(months)
            if mask.sum() == 0:
                continue
            a_s, p_s = act[mask], pred[mask]
            result[f"RMSE_{season_label}"]    = rmse(a_s, p_s)
            result[f"MAE_{season_label}"]     = mae(a_s, p_s)
            result[f"NSE_{season_label}"]     = nse(a_s, p_s)
            result[f"HitRate_{season_label}"] = hit_rate(a_s, p_s)

        # --- Extreme events breakdown (≥ 50 mm observed) ---
        extreme_mask = act >= EXTREME_THRESHOLD
        n_extreme = extreme_mask.sum()
        result["N_extreme"] = int(n_extreme)
        if n_extreme > 0:
            a_e, p_e = act[extreme_mask], pred[extreme_mask]
            result["RMSE_Extreme"] = rmse(a_e, p_e)
            result["MAE_Extreme"]  = mae(a_e, p_e)
            result["Bias_Extreme"] = bias(a_e, p_e)
        else:
            result["RMSE_Extreme"] = float("nan")
            result["MAE_Extreme"]  = float("nan")
            result["Bias_Extreme"] = float("nan")

    _log_metrics(result)
    return result


def _log_metrics(m: Dict) -> None:
    logger.info(
        f"[{m['Model']}] RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f} "
        f"R2={m['R2']:.4f} NSE={m['NSE']:.4f} Bias={m['Bias']:+.3f}"
    )


# ---------------------------------------------------------------------------
# Summary table construction
# ---------------------------------------------------------------------------

def build_comparison_table(results: list[Dict]) -> pd.DataFrame:
    """
    Convert a list of evaluate() result dicts into a sorted comparison DataFrame.
    Sorted by RMSE ascending (best model first).
    """
    df = pd.DataFrame(results)
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE").reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def format_comparison_table(df: pd.DataFrame) -> str:
    """Return a markdown-formatted comparison table string."""
    key_cols = [c for c in [
        "Rank", "Model", "RMSE", "MAE", "R2", "NSE",
        "MAPE_wet", "Bias", "HitRate",
        "RMSE_Monsoon", "RMSE_NonMonsoon", "RMSE_Extreme",
    ] if c in df.columns]
    sub = df[key_cols].copy()
    float_cols = [c for c in sub.columns if c not in ("Rank", "Model")]
    for c in float_cols:
        sub[c] = sub[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    lines = ["| " + " | ".join(sub.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(sub.columns)) + "|")
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)
