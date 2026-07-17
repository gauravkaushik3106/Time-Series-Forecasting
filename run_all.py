"""
run_all.py
==========
Complete Lucknow Rainfall Framework pipeline runner.

Executes all phases in strict dependency order:

  Phase 2 — Preprocessing + EDA
  Phase 3 — Feature Engineering
  Phase 4 — Baseline + Classical + ML Models
  Phase 5 — Deep Learning + Hybrid Model
  Phase 6 — Explainability + Uncertainty

Usage
-----
    # Full pipeline (all phases):
    python run_all.py

    # Skip deep learning (faster, runs Phases 2–4 + 6 only):
    python run_all.py --skip-dl

    # Skip SARIMAX auto-search (use default orders, much faster):
    python run_all.py --no-sarimax-search

    # Dry run — check all imports and paths without executing:
    python run_all.py --dry-run

    # Resume from a specific phase (skip earlier completed phases):
    python run_all.py --start-phase 4

Estimated runtimes (CPU, single core)
--------------------------------------
  Phase 2 (Preprocessing + EDA)    :  ~40s
  Phase 3 (Feature Engineering)    :  ~10s
  Phase 4 (Models, no SARIMA search):  ~3 min
  Phase 4 (Models, with SARIMA search): ~8 min
  Phase 5 (LSTM + GRU + Hybrid)    :  ~5 min (20 epochs, CPU)
  Phase 6 (Explainability + UQ)    :  ~90s
  Total (no SARIMA search, no DL)  :  ~5 min
  Total (full pipeline)            :  ~15 min

Data requirement
----------------
Place the raw data file at:
    data/raw/Lucknow_rainfall_cleaned.xls

before running.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path — support running from any directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_all")


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase2(save_figures: bool = True) -> None:
    """Preprocessing + EDA."""
    from run_preprocessing_eda import run_pipeline
    run_pipeline(save_figures=save_figures)


def run_phase3(save_figures: bool = True) -> None:
    """Feature Engineering."""
    from run_feature_engineering import run
    run(save_figures=save_figures)


def run_phase4(sarimax_search: bool = True, save_figures: bool = True) -> None:
    """Baseline + Classical + ML Models."""
    from src.models.model_pipeline import run_model_pipeline
    run_model_pipeline(
        run_sarimax_search=sarimax_search,
        save_figures=save_figures,
    )


def run_phase5(save_figures: bool = True) -> None:
    """Deep Learning + Hybrid Model."""
    # Apply reduced epoch cap for CPU environments
    import src.models.lstm as lstm_mod
    import src.models.gru  as gru_mod
    import src.models.sequence_generator as sg_mod

    if not _has_gpu():
        logger.warning(
            "No GPU detected — reducing LSTM/GRU to 20 epochs for CPU feasibility. "
            "For full quality, run on a CUDA-capable GPU with max_epochs=100."
        )
        lstm_mod.MAX_EPOCHS     = 20
        lstm_mod.EARLY_STOP_PAT = 8
        gru_mod.MAX_EPOCHS      = 20
        gru_mod.EARLY_STOP_PAT  = 8
        sg_mod.BATCH_SIZE       = 256

    from run_deep_learning import run_dl_pipeline
    run_dl_pipeline(save_figures=save_figures)


def run_phase6(save: bool = True) -> None:
    """Explainability + Uncertainty."""
    from run_explainability_uncertainty import run_phase6 as _run
    _run(save=save)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _banner(title: str, phase: int, total: int) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  PHASE {phase}/{total}: {title}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lucknow Rainfall Framework — Full Pipeline Runner"
    )
    parser.add_argument(
        "--skip-dl", action="store_true",
        help="Skip Phase 5 (LSTM/GRU/Hybrid) — run faster on CPU-only machines",
    )
    parser.add_argument(
        "--no-sarimax-search", action="store_true",
        help="Skip SARIMAX auto_arima order search; use default (1,0,1)×(1,1,1,12)",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip saving figures to disk (useful for CI or testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate imports and paths without executing anything",
    )
    parser.add_argument(
        "--start-phase", type=int, default=2, choices=[2, 3, 4, 5, 6],
        help="Resume from this phase (skip earlier phases already completed)",
    )
    args = parser.parse_args()

    save_figures = not args.no_figures

    # Dry run: test all imports
    if args.dry_run:
        _dry_run()
        return

    # Verify raw data exists before starting
    raw_data = _PROJECT_ROOT / "data" / "raw" / "Lucknow_rainfall_cleaned.xls"
    if not raw_data.exists():
        logger.error(
            f"Raw data file not found: {raw_data}\n"
            "Place 'Lucknow_rainfall_cleaned.xls' in data/raw/ and retry."
        )
        sys.exit(1)

    phases_to_run = [2, 3, 4, 5, 6]
    if args.skip_dl:
        phases_to_run = [p for p in phases_to_run if p != 5]

    phases_to_run = [p for p in phases_to_run if p >= args.start_phase]

    total       = len(phases_to_run)
    t_pipeline  = time.time()
    phase_times = {}

    logger.info("=" * 70)
    logger.info("  LUCKNOW RAINFALL FRAMEWORK — FULL PIPELINE")
    logger.info(f"  Phases to run : {phases_to_run}")
    logger.info(f"  GPU available : {_has_gpu()}")
    logger.info(f"  Save figures  : {save_figures}")
    logger.info("=" * 70)

    for seq_idx, phase_num in enumerate(phases_to_run, 1):

        if phase_num == 2:
            _banner("Preprocessing + EDA", seq_idx, total)
            t0 = time.time()
            run_phase2(save_figures=save_figures)
            phase_times[2] = time.time() - t0

        elif phase_num == 3:
            _banner("Feature Engineering", seq_idx, total)
            t0 = time.time()
            run_phase3(save_figures=save_figures)
            phase_times[3] = time.time() - t0

        elif phase_num == 4:
            _banner("Baseline + Classical + ML Models", seq_idx, total)
            t0 = time.time()
            run_phase4(
                sarimax_search=not args.no_sarimax_search,
                save_figures=save_figures,
            )
            phase_times[4] = time.time() - t0

        elif phase_num == 5:
            _banner("Deep Learning + Hybrid Model", seq_idx, total)
            t0 = time.time()
            run_phase5(save_figures=save_figures)
            phase_times[5] = time.time() - t0

        elif phase_num == 6:
            _banner("Explainability + Uncertainty", seq_idx, total)
            t0 = time.time()
            run_phase6(save=save_figures)
            phase_times[6] = time.time() - t0

    # ── Final summary ─────────────────────────────────────────────────────
    total_time = time.time() - t_pipeline
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total runtime : {total_time/60:.1f} minutes")
    logger.info("")
    logger.info("  Per-phase timings:")
    phase_labels = {
        2: "Preprocessing + EDA",
        3: "Feature Engineering",
        4: "Models",
        5: "Deep Learning",
        6: "Explainability + UQ",
    }
    for phase_num, elapsed in phase_times.items():
        logger.info(f"    Phase {phase_num} ({phase_labels[phase_num]}): "
                    f"{elapsed:.1f}s")
    logger.info("")
    logger.info("  Outputs:")
    logger.info("    outputs/features/    — scaled feature datasets")
    logger.info("    outputs/predictions/ — all model predictions")
    logger.info("    outputs/models/      — serialised model weights")
    logger.info("    outputs/figures/     — all plots (EDA, models, explainability)")
    logger.info("    outputs/reports/     — Markdown performance reports")
    logger.info("    outputs/explainability/ — SHAP summary table")
    logger.info("    outputs/uncertainty/    — MC Dropout predictions + calibration")
    logger.info("")
    logger.info("  Launch dashboard:")
    logger.info("    streamlit run dashboard/app.py")
    logger.info("=" * 70)


def _dry_run() -> None:
    """Validate all imports and critical paths without running anything."""
    logger.info("DRY RUN — validating imports and paths only")
    errors = []

    # Check data file
    raw = _PROJECT_ROOT / "data" / "raw" / "Lucknow_rainfall_cleaned.xls"
    if raw.exists():
        logger.info(f"  ✓ Raw data file found: {raw}")
    else:
        errors.append(f"  ✗ Raw data file missing: {raw}")

    # Check Python packages
    required = [
        ("pandas", "3.0"),
        ("numpy", "2.0"),
        ("scipy", "1.10"),
        ("matplotlib", "3.7"),
        ("seaborn", "0.12"),
        ("statsmodels", "0.14"),
        ("sklearn", "1.3"),
        ("xgboost", "2.0"),
        ("shap", "0.40"),
        ("torch", "2.0"),
        ("pmdarima", "2.0"),
        ("streamlit", "1.20"),
        ("plotly", "5.0"),
        ("yaml", None),
    ]
    for pkg, min_ver in required:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            logger.info(f"  ✓ {pkg} {ver}")
        except ImportError:
            errors.append(f"  ✗ {pkg} — NOT INSTALLED")

    # Check runner scripts
    for script in ["run_preprocessing_eda.py", "run_feature_engineering.py",
                   "run_models.py", "run_deep_learning.py",
                   "run_explainability_uncertainty.py", "dashboard/app.py"]:
        p = _PROJECT_ROOT / script
        if p.exists():
            logger.info(f"  ✓ {script}")
        else:
            errors.append(f"  ✗ {script} — MISSING")

    logger.info("")
    if errors:
        logger.error("DRY RUN FAILED — issues found:")
        for e in errors:
            logger.error(e)
        sys.exit(1)
    else:
        logger.info("DRY RUN PASSED — all imports and paths OK")


if __name__ == "__main__":
    main()
