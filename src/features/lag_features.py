"""
lag_features.py
===============
Lagged feature construction for the Lucknow rainfall dataset.

Temporal integrity guarantee
-----------------------------
Every lag operation uses pandas .shift(n), which shifts values FORWARD
by n positions — meaning the value at row t becomes available at row t+n.
This is equivalent to: feature[t] = original[t - n], i.e., the feature
observed at time t carries only information from n days in the past.

No future information is ever visible to the model at any time step.
This guarantee holds because:
  1.  We never use shift(-n) (negative shifts look ahead).
  2.  We never compute lags on the full dataframe after train/test merging.
  3.  The pipeline computes lags once on the full ordered time series,
      then applies the chronological split.  The first `max_lag` rows of the
      training set will contain NaN for long-lag features — these are dropped
      by the pipeline before model fitting to prevent false look-ahead from
      imputed values.

Feature naming convention
--------------------------
All lag features follow the pattern:  <SOURCE_COLUMN>_lag<N>
Examples:  RAINFALL_lag1, RH_lag7, CLOUD_lag3
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lag specification — single source of truth.
# Changing these dicts propagates automatically to the full pipeline.
# ---------------------------------------------------------------------------

# Rainfall lags: captures short (1–3 day), weekly (7), biweekly (14),
# and monthly-scale (30) persistence.  Justified by ACF analysis (Phase 2):
# lag-1 r=0.492, lag-7 r=0.122, lag-14 r=0.104.
RAINFALL_LAG_DAYS: List[int] = [1, 2, 3, 7, 14, 30]

# Key atmospheric driver lags: shorter memory appropriate for variables
# that respond more rapidly to synoptic forcing.
ATMOSPHERIC_LAG_CONFIG: Dict[str, List[int]] = {
    "RH":           [1, 3, 7],
    "PRESSURE":     [1, 3, 7],
    "CLOUD":        [1, 3, 7],
    "SOIL_WET_SURF":[1, 3, 7],
}


def create_lag_features(
    df: pd.DataFrame,
    rainfall_lags: List[int] = RAINFALL_LAG_DAYS,
    atmospheric_lags: Dict[str, List[int]] = ATMOSPHERIC_LAG_CONFIG,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct all lag features and append them to the dataframe.

    Parameters
    ----------
    df              : Cleaned dataframe with DatetimeIndex, sorted ascending.
    rainfall_lags   : List of lag days to apply to the RAINFALL column.
    atmospheric_lags: Mapping of column name → list of lag days.

    Returns
    -------
    df_out      : Original dataframe with lag columns appended.
    lag_col_names: List of all newly created column names.
    """
    _validate_sorted_index(df)
    df_out = df.copy()
    created: List[str] = []

    # --- Rainfall lags ---
    for lag in rainfall_lags:
        col_name = f"RAINFALL_lag{lag}"
        # shift(lag) moves row t's value to row t+lag, so at row t we see
        # the value from lag steps ago — strictly past information.
        df_out[col_name] = df_out["RAINFALL"].shift(lag)
        created.append(col_name)

    # --- Atmospheric driver lags ---
    for source_col, lags in atmospheric_lags.items():
        if source_col not in df_out.columns:
            logger.warning(
                f"Column '{source_col}' not found — skipping its lag features"
            )
            continue
        for lag in lags:
            col_name = f"{source_col}_lag{lag}"
            df_out[col_name] = df_out[source_col].shift(lag)
            created.append(col_name)

    max_lag = max(
        max(rainfall_lags) if rainfall_lags else 0,
        max(
            (max(v) for v in atmospheric_lags.values() if v),
            default=0,
        ),
    )

    n_nan_rows = df_out[created].isna().any(axis=1).sum()
    logger.info(
        f"Lag features created: {len(created)} columns | "
        f"max lag: {max_lag} days | "
        f"rows with ≥1 NaN lag: {n_nan_rows} "
        f"(will be dropped before model fitting)"
    )

    return df_out, created


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_sorted_index(df: pd.DataFrame) -> None:
    """Assert the index is a monotonically increasing DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected DatetimeIndex, got {type(df.index).__name__}. "
            "Lags are only meaningful on a time-ordered index."
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index is not sorted in ascending order. "
            "Sort before calling create_lag_features()."
        )


def get_max_lag() -> int:
    """Return the largest lag applied anywhere — used as the warm-up period."""
    return max(
        max(RAINFALL_LAG_DAYS),
        max(max(v) for v in ATMOSPHERIC_LAG_CONFIG.values()),
    )
