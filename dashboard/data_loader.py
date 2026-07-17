"""
data_loader.py
==============
Cached data loading for all precomputed dashboard artifacts.
ALL computations happened in Phases 1-6. This module only reads files.
No model inference, no training, no SHAP recomputation happens here.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import abs_path

PREDICTION_FILES: Dict[str, str] = {
    "Persistence":          "outputs/predictions/persistence_test.parquet",
    "Climatology":          "outputs/predictions/climatology_test.parquet",
    "SARIMAX":              "outputs/predictions/sarimax_test.parquet",
    "XGBoost":              "outputs/predictions/xgboost_test.parquet",
    "LSTM":                 "outputs/predictions/lstm_test.parquet",
    "GRU":                  "outputs/predictions/gru_test.parquet",
    "Hybrid SARIMAX+LSTM":  "outputs/predictions/hybrid_test.parquet",
}

MODEL_COLORS: Dict[str, str] = {
    "Persistence":         "#94A3B8",
    "Climatology":         "#F4A261",
    "SARIMAX":             "#2E86AB",
    "XGBoost":             "#3BB273",
    "LSTM":                "#5C6BC0",
    "GRU":                 "#8B5E83",
    "Hybrid SARIMAX+LSTM": "#E84855",
}

MODEL_RANK_COLORS = ["#F59E0B", "#94A3B8", "#CD7F32", "#CBD5E1",
                     "#CBD5E1", "#CBD5E1", "#CBD5E1"]


@st.cache_data
def load_full_features() -> pd.DataFrame:
    p = abs_path("outputs/features/full_features.parquet")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.DatetimeIndex(df.index)
    return df


@st.cache_data
def load_predictions() -> Dict[str, pd.DataFrame]:
    preds: Dict[str, pd.DataFrame] = {}
    for name, rel_path in PREDICTION_FILES.items():
        p = abs_path(rel_path)
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.DatetimeIndex(df.index)
            if "actual" in df.columns and "predicted" in df.columns:
                preds[name] = df[["actual", "predicted"]].copy()
    return preds


@st.cache_data
def load_comparison_table() -> pd.DataFrame:
    p4 = abs_path("outputs/predictions/model_comparison_table_test.csv")
    p5 = abs_path("outputs/predictions/model_comparison_table_phase5.csv")
    frames = []
    if p4.exists():
        frames.append(pd.read_csv(p4))
    if p5.exists():
        df5 = pd.read_csv(p5)
        if frames:
            existing = set(frames[0]["Model"])
            df5 = df5[~df5["Model"].isin(existing)]
        frames.append(df5)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "RMSE" in combined.columns:
        combined = combined.sort_values("RMSE").reset_index(drop=True)
        if "Rank" not in combined.columns:
            combined.insert(0, "Rank", range(1, len(combined) + 1))
    return combined


@st.cache_data
def load_shap_summary() -> pd.DataFrame:
    p = abs_path("outputs/explainability/shap_summary.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_calibration_metrics() -> pd.DataFrame:
    p = abs_path("outputs/uncertainty/calibration_metrics.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_mc_dropout_predictions() -> pd.DataFrame:
    p = abs_path("outputs/uncertainty/mc_dropout_predictions.parquet")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.DatetimeIndex(df.index)
    return df


@st.cache_data
def load_figure_paths() -> Dict[str, Dict[str, Path]]:
    categories = {
        "explainability": abs_path("outputs/figures/explainability"),
        "uncertainty":    abs_path("outputs/figures/uncertainty"),
        "models":         abs_path("outputs/figures/models"),
        "eda":            abs_path("outputs/figures/eda"),
    }
    result: Dict[str, Dict[str, Path]] = {}
    for cat, dir_path in categories.items():
        if dir_path.exists():
            result[cat] = {p.stem: p for p in sorted(dir_path.glob("*.png"))}
        else:
            result[cat] = {}
    return result


@st.cache_data
def load_dataset_stats() -> Dict:
    df = load_full_features()
    if df.empty:
        return {}
    rain = df["RAINFALL"] if "RAINFALL" in df.columns else pd.Series(dtype=float)
    return {
        "n_records":        len(df),
        "date_start":       str(df.index.min().date()),
        "date_end":         str(df.index.max().date()),
        "n_years":          df.index.year.nunique(),
        "n_features":       len([c for c in df.columns
                                  if c not in ("RAINFALL","SPLIT","SEASON")]),
        "pct_dry":          float((rain < 0.1).mean() * 100),
        "max_rainfall":     float(rain.max()),
        "annual_mean_mm":   float(rain.groupby(df.index.year).sum().mean()),
        "monsoon_frac_pct": float(
            rain[df.index.month.isin([6,7,8,9])].sum() / rain.sum() * 100
        ),
    }