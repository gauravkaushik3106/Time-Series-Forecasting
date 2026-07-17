"""
temporal_features.py
====================
Calendar and cyclical temporal features for the Lucknow rainfall dataset.

Why cyclical encoding?
-----------------------
Raw calendar integers (month=1..12, day_of_year=1..365) are ordinal but
NOT cyclical — a standard linear model would see month 12 and month 1 as
maximally different when they are actually adjacent (December → January).

Cyclical encoding maps each period to a point on the unit circle:
  sin_enc = sin(2π × value / period)
  cos_enc = cos(2π × value / period)

This preserves the continuity of the calendar: December 31 and January 1
are close in the encoded space.  Both the sin and cos components are required
together — sin alone is symmetric around the mid-cycle, cos alone is symmetric
around the start.  Together they uniquely encode every point in the cycle.

Features produced
-----------------
  Calendar integers:  MONTH, QUARTER, DAY_OF_YEAR, WEEK_OF_YEAR
  Season label:       already present from loader — refreshed here cleanly
  Binary flags:       IS_MONSOON (already present), IS_WEEKEND
  Cyclical pairs:     MONTH_SIN, MONTH_COS, DOY_SIN, DOY_COS
  Year position:      YEAR_FRAC  — continuous 0..1 position within year
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG

logger = logging.getLogger(__name__)


def create_temporal_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate calendar, cyclical, and seasonal temporal features.

    The loader already adds YEAR, MONTH, DAY_OF_YEAR, DAY_OF_WEEK,
    SEASON, and IS_MONSOON.  This module adds the cyclical encodings
    and additional calendar granularity that are absent from the loader.

    Parameters
    ----------
    df : Cleaned dataframe with sorted DatetimeIndex.

    Returns
    -------
    df_out       : Input dataframe with temporal columns appended.
    temp_col_names: List of newly created column names only.
    """
    df_out = df.copy()
    created: List[str] = []

    idx = df_out.index  # DatetimeIndex

    # --- Calendar granularity ---
    # QUARTER: maps months → 1-4, useful for capturing pre/peak/post-monsoon
    if "QUARTER" not in df_out.columns:
        df_out["QUARTER"] = idx.quarter
        created.append("QUARTER")

    # WEEK_OF_YEAR: ISO week (1–53), fine-grained seasonal position
    if "WEEK_OF_YEAR" not in df_out.columns:
        df_out["WEEK_OF_YEAR"] = idx.isocalendar().week.astype(int)
        created.append("WEEK_OF_YEAR")

    # IS_WEEKEND: rainfall distributions differ slightly on weekends in
    # urban-influenced stations due to measurement/reporting patterns
    if "IS_WEEKEND" not in df_out.columns:
        df_out["IS_WEEKEND"] = (idx.dayofweek >= 5).astype(int)
        created.append("IS_WEEKEND")

    # YEAR_FRAC: continuous position within year [0, 1)
    # Useful for smooth annual trend modelling
    if "YEAR_FRAC" not in df_out.columns:
        # Use day-of-year / 365.25 for leap-year robustness
        df_out["YEAR_FRAC"] = idx.dayofyear / 365.25
        created.append("YEAR_FRAC")

    # --- Cyclical encoding: MONTH ---
    # Period = 12 months
    if "MONTH_SIN" not in df_out.columns:
        month_rad = 2.0 * np.pi * df_out["MONTH"] / 12.0
        df_out["MONTH_SIN"] = np.sin(month_rad)
        df_out["MONTH_COS"] = np.cos(month_rad)
        created += ["MONTH_SIN", "MONTH_COS"]

    # --- Cyclical encoding: DAY_OF_YEAR ---
    # Period = 365.25 for leap-year robustness
    if "DOY_SIN" not in df_out.columns:
        doy_rad = 2.0 * np.pi * df_out["DAY_OF_YEAR"] / 365.25
        df_out["DOY_SIN"] = np.sin(doy_rad)
        df_out["DOY_COS"] = np.cos(doy_rad)
        created += ["DOY_SIN", "DOY_COS"]

    # --- Season integer encoding (ordered by rainfall intensity) ---
    # Provides an ordinal scale that tree-based models can use directly.
    #   0 = Winter (lowest rainfall)
    #   1 = Pre-Monsoon
    #   2 = Post-Monsoon
    #   3 = Monsoon (highest rainfall)
    season_order = {
        "Winter":       0,
        "Pre-Monsoon":  1,
        "Post-Monsoon": 2,
        "Monsoon":      3,
    }
    if "SEASON_CODE" not in df_out.columns:
        df_out["SEASON_CODE"] = df_out["SEASON"].map(season_order).fillna(0).astype(int)
        created.append("SEASON_CODE")

    # --- PRE/POST-MONSOON proximity features ---
    # Distance from monsoon onset (June 1 = DOY 152) and withdrawal (Sep 30 = DOY 273)
    # These capture the ramp-up and ramp-down of the monsoon season.
    if "DAYS_FROM_MONSOON_ONSET" not in df_out.columns:
        onset_doy = 152   # June 1
        withdraw_doy = 273  # Sep 30
        doy = df_out["DAY_OF_YEAR"]
        df_out["DAYS_FROM_MONSOON_ONSET"]      = doy - onset_doy
        df_out["DAYS_FROM_MONSOON_WITHDRAWAL"] = doy - withdraw_doy
        created += ["DAYS_FROM_MONSOON_ONSET", "DAYS_FROM_MONSOON_WITHDRAWAL"]

    logger.info(f"Temporal features created: {len(created)} columns")

    return df_out, created
