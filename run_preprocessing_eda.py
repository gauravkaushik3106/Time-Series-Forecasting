"""
run_preprocessing_eda.py
========================
Master runner for Phase 2: Preprocessing + EDA.

Execution order
---------------
1.  Load and validate raw data                     (src/preprocessing/loader.py)
2.  Rainfall distribution analysis                 (src/visualization/eda_rainfall_distribution.py)
3.  Seasonal and temporal analysis                 (src/visualization/eda_temporal.py)
4.  Correlation and multicollinearity analysis     (src/visualization/eda_correlation.py)
5.  Time-series structure analysis                 (src/visualization/eda_timeseries.py)
6.  Generate preprocessing report                 (src/preprocessing/report_generator.py)
7.  Generate EDA report                            (src/preprocessing/report_generator.py)

Usage
-----
    # From the project root:
    python run_preprocessing_eda.py

    # With explicit data path override:
    python run_preprocessing_eda.py --data path/to/file.csv

    # Skip figure saving (fast run for CI / testing):
    python run_preprocessing_eda.py --no-save
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable from any working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path
from src.preprocessing.loader import RainfallDataLoader
from src.preprocessing.report_generator import (
    generate_preprocessing_report,
    generate_eda_report,
)
from src.visualization.eda_rainfall_distribution import analyse_rainfall_distribution
from src.visualization.eda_temporal import analyse_temporal_patterns
from src.visualization.eda_correlation import analyse_correlations
from src.visualization.eda_timeseries import analyse_timeseries_structure

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: Path | None = None,
    save_figures: bool = True,
) -> None:
    """
    Execute the complete preprocessing and EDA pipeline.

    Parameters
    ----------
    data_path    : Override default raw data path.
    save_figures : Set False to skip disk I/O for rapid iteration.
    """
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("LUCKNOW RAINFALL FRAMEWORK — PREPROCESSING & EDA PIPELINE")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Data loading and validation
    # ------------------------------------------------------------------
    logger.info("\n[STEP 1/6] Data Loading & Validation")
    t0 = time.time()

    loader = RainfallDataLoader(raw_path=data_path)
    loader.run()

    df    = loader.df_
    train = loader.train_
    val   = loader.val_
    test  = loader.test_
    val_report = loader.validation_report_

    logger.info(f"  ✓ Completed in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # STEP 2: Rainfall distribution analysis
    # ------------------------------------------------------------------
    logger.info("\n[STEP 2/6] Rainfall Distribution Analysis")
    t0 = time.time()

    dist_stats = analyse_rainfall_distribution(df, save=save_figures)

    logger.info(f"  ✓ Completed in {time.time() - t0:.1f}s")
    logger.info(
        f"     Dry days: {dist_stats['pct_dry']:.1f}% | "
        f"Skewness: {dist_stats['skewness']:.2f} | "
        f"Max: {dist_stats['max']:.1f} mm"
    )

    # ------------------------------------------------------------------
    # STEP 3: Seasonal and temporal analysis
    # ------------------------------------------------------------------
    logger.info("\n[STEP 3/6] Seasonal & Temporal Analysis")
    t0 = time.time()

    temporal_stats = analyse_temporal_patterns(df, save=save_figures)

    logger.info(f"  ✓ Completed in {time.time() - t0:.1f}s")
    logger.info(
        f"     Monsoon fraction: {temporal_stats['monsoon_fraction_pct']:.1f}% | "
        f"Max dry spell: {temporal_stats['max_dry_spell_days']} days | "
        f"Annual CV: {temporal_stats['annual_cv_pct']:.1f}%"
    )

    # ------------------------------------------------------------------
    # STEP 4: Correlation and multicollinearity
    # ------------------------------------------------------------------
    logger.info("\n[STEP 4/6] Correlation & Multicollinearity Analysis")
    t0 = time.time()

    corr_stats = analyse_correlations(df, save=save_figures)

    logger.info(f"  ✓ Completed in {time.time() - t0:.1f}s")

    # Log VIF summary
    vif_df = corr_stats["vif_scores"]
    severe = vif_df[vif_df["Severity"] == "Severe"]["Feature"].tolist()
    if severe:
        logger.info(f"     ⚠ Severe multicollinearity in: {severe}")
    else:
        logger.info("     ✓ No severe multicollinearity detected")

    # ------------------------------------------------------------------
    # STEP 5: Time-series structure
    # ------------------------------------------------------------------
    logger.info("\n[STEP 5/6] Time-Series Structure Analysis")
    t0 = time.time()

    ts_stats = analyse_timeseries_structure(df, save=save_figures)

    logger.info(f"  ✓ Completed in {time.time() - t0:.1f}s")
    adf_daily = ts_stats["adf"].get("log_daily", {})
    logger.info(
        f"     ADF (log daily): p={adf_daily.get('p_value', 'N/A'):.4f} "
        f"({'stationary' if adf_daily.get('is_stationary') else 'non-stationary'}) | "
        f"STL seasonal strength: {ts_stats['stl']['seasonal_strength']:.3f}"
    )

    # ------------------------------------------------------------------
    # STEP 6: Report generation
    # ------------------------------------------------------------------
    logger.info("\n[STEP 6/6] Report Generation")
    t0 = time.time()

    generate_preprocessing_report(
        validation_report=val_report,
        df=df, train=train, val=val, test=test,
        save=True,
    )

    generate_eda_report(
        dist_stats=dist_stats,
        temporal_stats=temporal_stats,
        corr_stats=corr_stats,
        ts_stats=ts_stats,
        save=True,
    )

    logger.info(f"  ✓ Completed in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    n_figures = len(list(abs_path("outputs/figures/eda").glob("*.png")))

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total runtime   : {total_time:.1f}s")
    logger.info(f"  Figures saved   : {n_figures} PNG files")
    logger.info(f"  Reports saved   : outputs/reports/")
    logger.info(f"  Processed data  : {abs_path(CFG.paths.data_processed)}")
    logger.info("")
    logger.info("Key findings:")
    logger.info(f"  • Zero-inflation : {dist_stats['pct_dry']:.1f}% dry days → two-stage model mandatory")
    logger.info(f"  • Skewness       : {dist_stats['skewness']:.2f} → log transform required")
    logger.info(f"  • Monsoon frac   : {temporal_stats['monsoon_fraction_pct']:.1f}% → regime-aware training")
    logger.info(f"  • Lag-1 ACF      : {ts_stats['acf_pacf']['lag1_acf_raw']:.3f} → LSTM lookback ≥ 14 days")
    logger.info(f"  • Severe VIF     : {severe if severe else 'None detected'}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lucknow Rainfall Framework — Preprocessing & EDA Pipeline"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to raw data file (overrides config)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving figures to disk (faster run for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        data_path=args.data,
        save_figures=not args.no_save,
    )
