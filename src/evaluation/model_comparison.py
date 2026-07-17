"""
model_comparison.py
===================
Aggregates evaluation results from all models into comparison tables,
rankings, and a structured performance summary report.

Outputs
-------
outputs/reports/04_model_performance_report.md
outputs/predictions/model_comparison_table.csv
outputs/predictions/model_ranking_table.csv
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import abs_path
from src.evaluation.metrics import (
    build_comparison_table,
    format_comparison_table,
    evaluate,
)

logger = logging.getLogger(__name__)


def build_all_comparisons(
    pred_dfs: Dict[str, pd.DataFrame],
    split: str = "test",
) -> pd.DataFrame:
    """
    Evaluate all models and return a sorted comparison DataFrame.

    Parameters
    ----------
    pred_dfs : dict mapping model_name → DataFrame with [actual, predicted].
    split    : Label for logging (e.g. 'test', 'val').

    Returns
    -------
    Sorted DataFrame (best RMSE first) with all metric columns.
    """
    results = []
    for model_name, pred_df in pred_dfs.items():
        metrics = evaluate(
            actual=pred_df["actual"],
            predicted=pred_df["predicted"],
            model_name=model_name,
            index=pred_df.index,
        )
        results.append(metrics)

    comparison_df = build_comparison_table(results)

    out_dir = abs_path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_dir / f"model_comparison_table_{split}.csv", index=False)
    logger.info(f"Comparison table saved for {split} split")

    return comparison_df


def generate_performance_report(
    comparison_df: pd.DataFrame,
    pred_dfs: Dict[str, pd.DataFrame],
    split: str = "test",
) -> str:
    """
    Generate the full Markdown performance report.

    Parameters
    ----------
    comparison_df : Output of build_all_comparisons().
    pred_dfs      : Raw prediction DataFrames for narrative statistics.
    split         : 'test' or 'val'.

    Returns
    -------
    Markdown string (also saved to disk).
    """
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best = comparison_df.iloc[0]

    lines = [
        "# Lucknow Rainfall Framework — Model Performance Report",
        f"Generated: {now}  |  Evaluation split: **{split}**",
        "",
        "---",
        "",
        "## 1. Model Ranking (by RMSE, ascending)",
        "",
        format_comparison_table(comparison_df),
        "",
        "---",
        "",
        "## 2. Best Model Summary",
        "",
        f"**Best model:** {best['Model']}",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| RMSE | {best['RMSE']:.4f} mm/day |",
        f"| MAE  | {best['MAE']:.4f} mm/day |",
        f"| R²   | {best['R2']:.4f} |",
        f"| NSE  | {best['NSE']:.4f} |",
        f"| MAPE (wet days) | {best.get('MAPE_wet', float('nan')):.2f}% |",
        f"| Bias | {best['Bias']:+.4f} mm/day |",
        f"| Hit Rate | {best.get('HitRate', float('nan')):.4f} |",
        "",
        "---",
        "",
        "## 3. Seasonal Performance",
        "",
        "### 3a. Monsoon (Jun–Sep) vs Non-Monsoon",
        "",
    ]

    # Seasonal table
    season_cols = [c for c in comparison_df.columns
                   if "Monsoon" in c or c == "Model"]
    if len(season_cols) > 1:
        season_df = comparison_df[season_cols].copy()
        lines.append(_df_to_markdown(season_df))
    else:
        lines.append("_(Seasonal breakdown not available)_")

    lines += [
        "",
        "### 3b. Interpretation",
        "",
    ]

    for _, row in comparison_df.iterrows():
        rmse_m  = row.get("RMSE_Monsoon",    float("nan"))
        rmse_nm = row.get("RMSE_NonMonsoon", float("nan"))
        if pd.notna(rmse_m) and pd.notna(rmse_nm):
            harder = "monsoon" if rmse_m > rmse_nm else "non-monsoon"
            lines.append(
                f"- **{row['Model']}**: Monsoon RMSE={rmse_m:.3f} mm, "
                f"Non-Monsoon RMSE={rmse_nm:.3f} mm. "
                f"Harder season: **{harder}**."
            )

    lines += [
        "",
        "---",
        "",
        "## 4. Extreme Rainfall Performance (≥ 50 mm/day)",
        "",
    ]

    extreme_cols = [c for c in comparison_df.columns
                    if "Extreme" in c or c in ("Model", "N_extreme")]
    if len(extreme_cols) > 2:
        ext_df = comparison_df[extreme_cols].copy()
        lines.append(_df_to_markdown(ext_df))
        lines.append("")
        lines.append(
            f"> Test period contained **{int(comparison_df['N_extreme'].iloc[0])}** "
            "extreme rainfall days (≥ 50 mm). All models are expected to "
            "under-predict extremes due to the right-skewed target distribution. "
            "Negative bias (under-prediction) is the dominant failure mode."
        )
    else:
        lines.append("_(Extreme breakdown not available)_")

    lines += [
        "",
        "---",
        "",
        "## 5. Model-by-Model Notes",
        "",
    ]

    model_notes = {
        "Persistence": (
            "Simplest possible model: predict tomorrow = today. Strong during "
            "monsoon onset (high autocorrelation) but fails badly on the "
            "transition from wet to dry days. Sets the minimum useful skill bar."
        ),
        "Climatology": (
            "Predicts the smoothed historical mean for each day-of-year. "
            "Captures the seasonal cycle perfectly but has zero day-to-day skill. "
            "Systematically over-predicts dry months and under-predicts heavy rain. "
            "NSE close to 0 is expected."
        ),
        "SARIMAX": (
            "Captures linear autoregressive structure and seasonal trends with "
            "exogenous meteorological predictors. Performance is constrained by "
            "the log-Gaussian distributional assumption and the inability to model "
            "nonlinear feature interactions. Residuals are saved for the hybrid model."
        ),
        "XGBoost": (
            "Two-stage model: rain/no-rain classifier followed by wet-day amount "
            "regressor. Handles nonlinear feature interactions and zero-inflation "
            "natively. Expected to be the strongest classical model, particularly "
            "on extreme events where SARIMAX is constrained by its linear structure."
        ),
    }

    for model_name, note in model_notes.items():
        row = comparison_df[comparison_df["Model"] == model_name]
        if len(row) == 0:
            continue
        rank = int(row["Rank"].values[0]) if "Rank" in row.columns else "—"
        lines += [
            f"### {model_name} (Rank {rank})",
            "",
            note,
            "",
        ]

    lines += [
        "---",
        "",
        "## 6. Key Findings",
        "",
        _generate_key_findings(comparison_df),
        "",
        "---",
        "",
        "## 7. Next Steps",
        "",
        "- Phase 5: LSTM and GRU deep learning models",
        "- Phase 6: Hybrid SARIMAX + LSTM residual model",
        "- Phase 7: Uncertainty quantification (MC Dropout)",
        "- Phase 8: SHAP explainability on the best model",
        "- Phase 9: Interactive Streamlit dashboard",
        "",
        "---",
        f"*Report generated by `src/evaluation/model_comparison.py`*",
    ]

    report_text = "\n".join(lines)
    out_path = abs_path("outputs/reports/04_model_performance_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    logger.info(f"Performance report saved → {out_path}")

    return report_text


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table string."""
    float_cols = df.select_dtypes(include=float).columns
    df = df.copy()
    for c in float_cols:
        df[c] = df[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    header = "| " + " | ".join(df.columns) + " |"
    sep    = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows   = ["| " + " | ".join(str(v) for v in row) + " |"
              for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def _generate_key_findings(comparison_df: pd.DataFrame) -> str:
    """Auto-generate narrative key findings from comparison metrics."""
    findings = []
    best  = comparison_df.iloc[0]
    worst = comparison_df.iloc[-1]

    # Best vs worst RMSE improvement
    rmse_improvement = (
        (worst["RMSE"] - best["RMSE"]) / worst["RMSE"] * 100
        if worst["RMSE"] > 0 else 0
    )
    findings.append(
        f"- **Best model** ({best['Model']}) achieves RMSE={best['RMSE']:.3f} mm/day, "
        f"a **{rmse_improvement:.1f}%** improvement over the weakest model "
        f"({worst['Model']}, RMSE={worst['RMSE']:.3f} mm/day)."
    )

    # NSE interpretation
    for _, row in comparison_df.iterrows():
        nse_val = row.get("NSE", float("nan"))
        if pd.notna(nse_val):
            if nse_val > 0.5:
                verdict = "good skill (NSE > 0.5)"
            elif nse_val > 0.0:
                verdict = "modest skill (NSE > 0)"
            else:
                verdict = "no skill over climatology mean (NSE ≤ 0)"
            findings.append(f"- **{row['Model']}**: NSE={nse_val:.4f} — {verdict}.")

    # Bias direction
    for _, row in comparison_df.iterrows():
        bias_val = row.get("Bias", float("nan"))
        if pd.notna(bias_val):
            direction = "over-predicts" if bias_val > 0 else "under-predicts"
            findings.append(
                f"- **{row['Model']}** systematically {direction} "
                f"by {abs(bias_val):.3f} mm/day on average."
            )

    return "\n".join(findings)
