"""
run_models.py
=============
Master runner for Phase 4: Baselines + Classical + ML Models.

Usage
-----
    python run_models.py                    # full run with auto_arima search
    python run_models.py --no-sarimax-search  # use default SARIMAX orders (faster)
    python run_models.py --no-save          # skip figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.model_pipeline import run_model_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lucknow Rainfall Framework — Phase 4: Model Training"
    )
    parser.add_argument(
        "--no-sarimax-search", action="store_true",
        help="Skip auto_arima search; use default SARIMAX orders"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving figures to disk"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_model_pipeline(
        run_sarimax_search=not args.no_sarimax_search,
        save_figures=not args.no_save,
    )
