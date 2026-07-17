"""
loader.py
=========
Production-grade data loading pipeline for the Lucknow Rainfall dataset.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    passed: bool
    n_rows: int
    n_columns: int
    date_start: str
    date_end: str
    missing_values: Dict[str, int]
    duplicate_dates: int
    temporal_gaps: List[str]
    bound_violations: Dict[str, int]
    dtype_issues: Dict[str, str]
    warnings: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Status         : {'PASSED' if self.passed else 'FAILED'}",
            f"Rows           : {self.n_rows:,}",
            f"Columns        : {self.n_columns}",
            f"Date range     : {self.date_start} to {self.date_end}",
            f"Missing values : {sum(self.missing_values.values())} total",
            f"Duplicate dates: {self.duplicate_dates}",
            f"Temporal gaps  : {len(self.temporal_gaps)}",
            f"Bound violations: {sum(self.bound_violations.values())} total",
        ]
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


class RainfallDataLoader:
    """
    Loads, validates, cleans, and splits the Lucknow meteorological dataset.

    Parameters
    ----------
    raw_path : Path or str, optional
        Override the default path from config.
    """

    def __init__(self, raw_path: Path | str | None = None) -> None:
        self.raw_path: Path = (
            Path(raw_path) if raw_path else abs_path(CFG.paths.data_raw)
        )
        self.processed_path: Path = abs_path(CFG.paths.data_processed)
        self.df_: pd.DataFrame | None = None
        self.train_: pd.DataFrame | None = None
        self.val_: pd.DataFrame | None = None
        self.test_: pd.DataFrame | None = None
        self.validation_report_: ValidationReport | None = None

    def run(self) -> "RainfallDataLoader":
        logger.info("Starting data loading pipeline")
        logger.info(f"Source: {self.raw_path}")

        df_raw = self._load_raw()
        df_clean, report = self._validate_and_clean(df_raw)
        self.df_ = df_clean
        self.validation_report_ = report

        logger.info(report.summary())

        if not report.passed:
            raise RuntimeError(
                "Data validation failed. Inspect the ValidationReport for details."
            )

        self._split()
        self._save_processed()
        logger.info("Data loading pipeline complete")
        return self

    @property
    def splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if any(s is None for s in [self.train_, self.val_, self.test_]):
            raise RuntimeError("Call run() before accessing splits.")
        return self.train_, self.val_, self.test_

    def _load_raw(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")
        logger.info(f"Loading raw data from {self.raw_path}")
        with open(self.raw_path, "rb") as fh:
            header_bytes = fh.read(8)
        is_csv = header_bytes[:4] not in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")
        if is_csv:
            logger.info("Detected CSV format")
            df = pd.read_csv(self.raw_path, low_memory=False)
        else:
            logger.info("Detected binary Excel format")
            try:
                df = pd.read_excel(self.raw_path, engine="xlrd")
            except Exception:
                df = pd.read_excel(self.raw_path, engine="openpyxl")
        logger.info(f"Raw data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def _validate_and_clean(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        warnings: List[str] = []
        passed = True

        date_col = CFG.schema.date_column
        target_col = CFG.schema.target_column
        feature_cols: List[str] = list(CFG.schema.feature_columns)
        expected_cols = [date_col, target_col] + feature_cols

        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info("Column presence check passed")

        df = df[expected_cols].copy()

        # Date parsing
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        n_unparseable = df[date_col].isna().sum()
        if n_unparseable > 0:
            passed = False
            warnings.append(f"{n_unparseable} DATE values could not be parsed")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        actual_start = df[date_col].min()
        actual_end   = df[date_col].max()
        logger.info(f"Date range: {actual_start.date()} to {actual_end.date()}")

        # Duplicate detection
        n_duplicates = df.duplicated(subset=[date_col]).sum()
        if n_duplicates > 0:
            passed = False
            warnings.append(f"{n_duplicates} duplicate DATE entries found")
            df = df.drop_duplicates(subset=[date_col], keep="first")
        else:
            logger.info("No duplicate dates")

        # Temporal continuity
        full_range   = pd.date_range(start=actual_start, end=actual_end, freq="D")
        present      = set(df[date_col].dt.normalize())
        missing_dates = sorted(set(full_range) - present)
        temporal_gaps: List[str] = []
        if missing_dates:
            gaps = _group_consecutive_dates(missing_dates)
            temporal_gaps = [
                f"{g[0].date()} to {g[-1].date()} ({len(g)} days)" for g in gaps
            ]
            warnings.append(f"{len(missing_dates)} missing dates: {temporal_gaps}")
        else:
            logger.info("Temporal continuity verified — no gaps")

        # Missing values
        missing_values: Dict[str, int] = df.isnull().sum().to_dict()
        if sum(missing_values.values()) == 0:
            logger.info("No missing values in any column")
        else:
            warnings.append(f"Missing values: {missing_values}")

        # Dtype coercion
        numeric_cols = [target_col] + feature_cols
        dtype_issues: Dict[str, str] = {}
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                dtype_issues[col] = str(df[col].dtype)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Physical bounds
        bounds_cfg = vars(CFG.schema.bounds)
        bound_violations: Dict[str, int] = {}
        for col, bounds in bounds_cfg.items():
            if col not in df.columns:
                continue
            lo, hi = bounds[0], bounds[1]
            n_viol = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_viol > 0:
                bound_violations[col] = n_viol
                warnings.append(
                    f"Column '{col}': {n_viol} values outside [{lo}, {hi}]"
                )
        if not bound_violations:
            logger.info("All values within physical plausibility bounds")

        # Set DatetimeIndex
        df = df.set_index(date_col)
        df.index.name = "DATE"
        df.index = pd.DatetimeIndex(df.index, freq="D")

        # Derived temporal columns
        df["YEAR"]       = df.index.year
        df["MONTH"]      = df.index.month
        df["DAY_OF_YEAR"] = df.index.dayofyear
        df["DAY_OF_WEEK"] = df.index.dayofweek
        df["SEASON"]     = df["MONTH"].map(_month_to_season)
        df["IS_MONSOON"] = df["MONTH"].isin(list(CFG.monsoon_months)).astype(int)

        report = ValidationReport(
            passed=passed,
            n_rows=len(df),
            n_columns=len(df.columns),
            date_start=str(actual_start.date()),
            date_end=str(actual_end.date()),
            missing_values=missing_values,
            duplicate_dates=int(n_duplicates),
            temporal_gaps=temporal_gaps,
            bound_violations=bound_violations,
            dtype_issues=dtype_issues,
            warnings=warnings,
        )
        logger.info(f"Cleaned dataframe: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df, report

    def _split(self) -> None:
        """
        Produce strictly chronological train / validation / test splits.

        WHY CHRONOLOGICAL SPLITTING IS MANDATORY
        -----------------------------------------
        Random k-fold cross-validation is invalid for time series because:
        1. Temporal dependence: consecutive observations are correlated.
           Random splitting leaks future observations into the training set
           (look-ahead bias), producing metrics that are too optimistic.
        2. Distribution shift: rainfall behaviour changes across seasons and
           years. Evaluation must reflect the model predicting the future
           from the past, exactly as it must do in production.

        Correct structure:
           [========== TRAIN ==========][=== VAL ===][=== TEST ===]
           sorted chronologically, no overlap, no shuffling.
        """
        df = self.df_
        n = len(df)
        n_train = int(n * CFG.split.train_frac)
        n_val   = int(n * CFG.split.val_frac)

        self.train_ = df.iloc[:n_train].copy()
        self.val_   = df.iloc[n_train: n_train + n_val].copy()
        self.test_  = df.iloc[n_train + n_val:].copy()

        logger.info(
            f"Chronological split:\n"
            f"  Train : {len(self.train_):,} rows "
            f"({self.train_.index[0].date()} to {self.train_.index[-1].date()})\n"
            f"  Val   : {len(self.val_):,} rows "
            f"({self.val_.index[0].date()} to {self.val_.index[-1].date()})\n"
            f"  Test  : {len(self.test_):,} rows "
            f"({self.test_.index[0].date()} to {self.test_.index[-1].date()})"
        )
        assert self.train_.index[-1] < self.val_.index[0], "Train/val overlap!"
        assert self.val_.index[-1]   < self.test_.index[0], "Val/test overlap!"
        logger.info("No temporal overlap between splits confirmed")

    def _save_processed(self) -> None:
        out = self.processed_path
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df_.to_parquet(out, engine="pyarrow", compression="snappy")
        logger.info(f"Processed data saved: {out}")


def _group_consecutive_dates(dates: List[pd.Timestamp]) -> List[List[pd.Timestamp]]:
    if not dates:
        return []
    groups: List[List[pd.Timestamp]] = [[dates[0]]]
    for d in dates[1:]:
        if (d - groups[-1][-1]).days == 1:
            groups[-1].append(d)
        else:
            groups.append([d])
    return groups


def _month_to_season(month: int) -> str:
    if month in list(CFG.monsoon_months):
        return "Monsoon"
    if month in list(CFG.premonsoon_months):
        return "Pre-Monsoon"
    if month in list(CFG.postmonsoon_months):
        return "Post-Monsoon"
    if month in list(CFG.winter_months):
        return "Winter"
    return "Unknown"


def load_data(
    raw_path: Path | str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience one-liner for notebooks: returns (df_full, train, val, test)."""
    loader = RainfallDataLoader(raw_path=raw_path)
    loader.run()
    return loader.df_, loader.train_, loader.val_, loader.test_
