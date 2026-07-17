"""
run_deep_learning.py
====================
Phase 5 runner: LSTM, GRU, and Hybrid SARIMAX+LSTM.

Execution order
---------------
1.  Load feature datasets and SARIMAX outputs from Phase 4
2.  Build sliding-window sequences (lookback=60)
3.  Train standalone LSTM (LOG_RAINFALL target, weighted loss)
4.  Train standalone GRU  (same configuration)
5.  Train Hybrid SARIMAX+LSTM (residual target)
6.  Evaluate all DL models + XGBoost on test set
7.  Generate per-model prediction figures
8.  Generate extended comparison plots (training curves, extreme events)
9.  Write Phase 5 performance report

Usage
-----
    python run_deep_learning.py
    python run_deep_learning.py --no-save    # skip figures
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path
from src.models.sequence_generator import SequenceGenerator, LOOKBACK_LONG
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.hybrid_model import run_hybrid
from src.evaluation.metrics import evaluate, build_comparison_table, format_comparison_table
from src.visualization.plot_utils import (
    apply_style, save_figure, add_figure_title,
    BLUE, RED, GREEN, ORANGE, GRAY, CATEGORICAL_PALETTE,
    annotate_monsoon_bands,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phase5")

MODEL_COLORS = {
    "XGBoost":           GREEN,
    "LSTM":              "#5C6BC0",
    "GRU":               "#8B5E83",
    "Hybrid_SARIMAX_LSTM": RED,
    "SARIMAX":           BLUE,
    "Persistence":       GRAY,
    "Climatology":       ORANGE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data():
    fd = abs_path("outputs/features")
    train_ml = pd.read_parquet(fd / "train_features_ml.parquet")
    val_ml   = pd.read_parquet(fd / "val_features_ml.parquet")
    test_ml  = pd.read_parquet(fd / "test_features_ml.parquet")

    pd_dir = abs_path("outputs/predictions")
    sarimax_resid = pd.read_parquet(pd_dir / "sarimax_train_residuals.parquet")[
        "SARIMAX_train_residuals"
    ]
    sarimax_val   = pd.read_parquet(pd_dir / "sarimax_val.parquet")
    sarimax_test  = pd.read_parquet(pd_dir / "sarimax_test.parquet")
    xgb_test      = pd.read_parquet(pd_dir / "xgboost_test.parquet")

    logger.info(
        f"Data loaded: train={len(train_ml):,} val={len(val_ml):,} "
        f"test={len(test_ml):,}"
    )
    return train_ml, val_ml, test_ml, sarimax_resid, sarimax_val, sarimax_test, xgb_test


def _build_sequences(train_ml, val_ml, test_ml, target_col="LOG_RAINFALL"):
    seq_gen = SequenceGenerator(lookback=LOOKBACK_LONG, target_col=target_col)
    loaders = seq_gen.build(train_ml, val_ml, test_ml)
    return seq_gen, loaders


def _get_test_sequences(seq_gen, test_ml):
    X_seq, y_seq, _ = seq_gen.get_arrays(test_ml, split_name="test")
    return X_seq, y_seq


def _pred_to_df(model, X_seq, test_ml, lookback):
    preds_mm = model.predict_mm(X_seq)
    # Align index: first `lookback` rows have no sequence
    valid_idx = test_ml.index[lookback:]
    actual_mm = test_ml.loc[valid_idx, "RAINFALL"].values
    return pd.DataFrame(
        {"actual": actual_mm, "predicted": preds_mm},
        index=valid_idx,
    )


# ---------------------------------------------------------------------------
# Training curve figure
# ---------------------------------------------------------------------------

def _plot_training_curves(histories: Dict, save: bool) -> None:
    apply_style()
    n   = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, hist) in zip(axes, histories.items()):
        color = MODEL_COLORS.get(name, BLUE)
        ax.plot(hist["train"], lw=1.2, color=color, label="Train loss")
        ax.plot(hist["val"],   lw=1.2, color=RED, linestyle="--", label="Val loss")
        best_ep = int(np.argmin(hist["val"])) + 1
        ax.axvline(best_ep, color=GRAY, lw=0.8, linestyle=":", label=f"Best epoch {best_ep}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted MSE Loss")
        ax.set_title(f"{name} — Training Curve", fontweight="bold")
        ax.legend(fontsize=8)
    add_figure_title(fig, "Deep Learning Training Curves — Phase 5")
    if save:
        save_figure(fig, "dl_training_curves", subdir="models")
    plt.close(fig)


# ---------------------------------------------------------------------------
# All-model comparison figure (Phase 5 extended)
# ---------------------------------------------------------------------------

def _plot_full_comparison(all_preds: Dict[str, pd.DataFrame], save: bool) -> None:
    apply_style()
    models_sorted = list(all_preds.keys())
    results = [
        evaluate(
            pd.Series(df["actual"]), pd.Series(df["predicted"]),
            model_name=m, index=df.index,
        )
        for m, df in all_preds.items()
    ]
    comp = build_comparison_table(results)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    models = comp["Model"].tolist()
    colors = [MODEL_COLORS.get(m, BLUE) for m in models]

    for ax, (metric, ylabel, note) in zip(axes, [
        ("RMSE", "RMSE (mm/day)", "lower is better"),
        ("NSE",  "NSE",           "higher is better"),
        ("MAE",  "MAE (mm/day)",  "lower is better"),
    ]):
        vals = comp[metric].values
        bars = ax.bar(models, vals, color=colors, alpha=0.82, edgecolor="none")
        ax.set_title(f"{ylabel}\n({note})", fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.015,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        best = int(np.nanargmin(vals)) if "lower" in note else int(np.nanargmax(vals))
        bars[best].set_edgecolor("#FF0000"); bars[best].set_linewidth(2)

    add_figure_title(fig, "Full Model Comparison (Phases 4+5) — Test Set",
                     "Red border = best model per metric")
    if save:
        save_figure(fig, "dl_full_model_comparison", subdir="models")
    plt.close(fig)
    return comp


# ---------------------------------------------------------------------------
# Prediction overlay — all DL models
# ---------------------------------------------------------------------------

def _plot_dl_overlay(all_preds: Dict[str, pd.DataFrame], save: bool) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=False)

    dl_models  = {k: v for k, v in all_preds.items()
                  if k in ("LSTM", "GRU", "Hybrid_SARIMAX_LSTM", "XGBoost")}
    first_df   = list(dl_models.values())[0]

    for ax_idx, (ax, title) in enumerate(zip(axes, [
        "Full test period — All Models",
        "Monsoon season zoom",
    ])):
        if ax_idx == 1:
            zoom_start = None
            for ts in first_df.index:
                if ts.month == 6:
                    zoom_start = ts; break
            if zoom_start is None:
                ax.set_visible(False); continue
            zoom_end  = zoom_start + pd.DateOffset(months=5)
            mask_fn   = lambda idx: (idx >= zoom_start) & (idx <= zoom_end)
        else:
            mask_fn = lambda idx: np.ones(len(idx), dtype=bool)

        mask = mask_fn(first_df.index)
        ax.fill_between(first_df.index[mask], first_df["actual"].values[mask],
                        alpha=0.18, color=BLUE, label="Actual")
        ax.plot(first_df.index[mask], first_df["actual"].values[mask],
                lw=0.7, color=BLUE, alpha=0.8)

        for mname, df in dl_models.items():
            m = mask_fn(df.index)
            ax.plot(df.index[m], df["predicted"].values[m],
                    lw=1.1, color=MODEL_COLORS.get(mname, GRAY),
                    alpha=0.85, label=mname)

        if ax_idx == 0:
            annotate_monsoon_bands(ax, first_df, alpha=0.06)
        ax.set_ylabel("Rainfall (mm/day)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9, ncol=4)
        ax.set_ylim(bottom=0)

    add_figure_title(fig, "Deep Learning Models — Prediction Overlay (Test Set)")
    if save:
        save_figure(fig, "dl_prediction_overlay", subdir="models")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Extreme event comparison
# ---------------------------------------------------------------------------

def _plot_extreme_comparison(all_preds: Dict[str, pd.DataFrame], save: bool) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel a: >50 mm MAE for all models
    extreme_maes = {}
    for mname, df in all_preds.items():
        mask = df["actual"] >= 50
        if mask.sum() > 0:
            extreme_maes[mname] = float(np.mean(np.abs(
                df.loc[mask, "actual"].values - df.loc[mask, "predicted"].values
            )))

    sorted_models = sorted(extreme_maes, key=extreme_maes.get)
    colors = [MODEL_COLORS.get(m, BLUE) for m in sorted_models]
    axes[0].bar(sorted_models, [extreme_maes[m] for m in sorted_models],
                color=colors, alpha=0.82, edgecolor="none")
    for i, (m, v) in enumerate([(m, extreme_maes[m]) for m in sorted_models]):
        axes[0].text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("MAE (mm/day)")
    axes[0].set_title(">50 mm Events — MAE by Model\n(lower is better)", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)

    # Panel b: Actual vs Predicted scatter for best model (Hybrid or XGBoost)
    best_model = sorted_models[0]
    df_best    = all_preds[best_model]
    ext_mask   = df_best["actual"] >= 50

    axes[1].scatter(df_best.loc[~ext_mask, "actual"],
                    df_best.loc[~ext_mask, "predicted"],
                    s=4, alpha=0.15, color=GRAY, label="<50 mm")
    if ext_mask.sum() > 0:
        axes[1].scatter(df_best.loc[ext_mask, "actual"],
                        df_best.loc[ext_mask, "predicted"],
                        s=60, alpha=0.85, color=RED, marker="*", label="≥50 mm")
    lim = max(df_best["actual"].max(), df_best["predicted"].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], color=GRAY, lw=1.0, linestyle="--")
    axes[1].set_xlabel("Actual (mm/day)")
    axes[1].set_ylabel("Predicted (mm/day)")
    axes[1].set_title(f"{best_model} — Scatter (★ = extreme events ≥50 mm)", fontweight="bold")
    axes[1].legend(fontsize=9)

    add_figure_title(fig, "Extreme Rainfall Event Performance — Phase 5 Models")
    if save:
        save_figure(fig, "dl_extreme_event_comparison", subdir="models")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 5 report
# ---------------------------------------------------------------------------

def _write_report(comp: pd.DataFrame, histories: Dict, save: bool) -> None:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Lucknow Rainfall Framework — Phase 5 Report",
        f"Generated: {now}",
        "",
        "---",
        "",
        "## 1. Model Ranking — Full Test Set (all phases)",
        "",
        format_comparison_table(comp),
        "",
        "---",
        "",
        "## 2. Training Summary",
        "",
        "| Model | Best Epoch | Best Val Loss |",
        "|---|---|---|",
    ]
    for mname, hist in histories.items():
        best_ep  = int(np.argmin(hist["val"])) + 1
        best_val = float(np.min(hist["val"]))
        lines.append(f"| {mname} | {best_ep} | {best_val:.5f} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Seasonal Performance",
        "",
        "| Model | RMSE Monsoon | RMSE Non-Monsoon | NSE Monsoon | NSE Non-Monsoon |",
        "|---|---|---|---|---|",
    ]
    for _, row in comp.iterrows():
        lines.append(
            f"| {row['Model']} "
            f"| {row.get('RMSE_Monsoon', float('nan')):.4f} "
            f"| {row.get('RMSE_NonMonsoon', float('nan')):.4f} "
            f"| {row.get('NSE_Monsoon', float('nan')):.4f} "
            f"| {row.get('NSE_NonMonsoon', float('nan')):.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Extreme Rainfall (>50 mm) Performance",
        "",
        "| Model | RMSE_Extreme | MAE_Extreme | Bias_Extreme | N_extreme |",
        "|---|---|---|---|---|",
    ]
    for _, row in comp.iterrows():
        lines.append(
            f"| {row['Model']} "
            f"| {row.get('RMSE_Extreme', float('nan')):.4f} "
            f"| {row.get('MAE_Extreme', float('nan')):.4f} "
            f"| {row.get('Bias_Extreme', float('nan')):.4f} "
            f"| {int(row.get('N_extreme', 0))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Key Findings",
        "",
        "- **Hybrid model** adds LSTM-learned corrections on top of SARIMAX, ",
        "  targeting the residual ACF (0.117) that SARIMAX leaves unexploited.",
        "- **Weighted loss** (5× for >20 mm) biases all DL models toward ",
        "  heavy-rain events, the dominant failure mode identified in diagnostics.",
        "- **60-day lookback** extends beyond the XGBoost 30-day lag ceiling,",
        "  potentially capturing monsoon onset dynamics.",
        "- **GRU vs LSTM**: empirical comparison on this dataset answers whether",
        "  fewer gated parameters generalise better on the ~6,600 row training set.",
        "",
        "---",
        f"*Generated by `run_deep_learning.py`*",
    ]

    report = "\n".join(lines)
    if save:
        out = abs_path("outputs/reports/05_phase5_report.md")
        out.write_text(report, encoding="utf-8")
        logger.info(f"Phase 5 report saved → {out}")

    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_dl_pipeline(save_figures: bool = True) -> None:
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 5: DEEP LEARNING + HYBRID MODEL")
    logger.info("=" * 70)

    # 1. Load
    (train_ml, val_ml, test_ml,
     sarimax_resid, sarimax_val, sarimax_test, xgb_test) = _load_data()

    out_pred = abs_path("outputs/predictions")
    out_pred.mkdir(parents=True, exist_ok=True)
    histories: Dict = {}

    # 2. Build sequences for standalone models
    logger.info("\n[1/6] Building sequences (lookback=60)")
    seq_gen, loaders = _build_sequences(train_ml, val_ml, test_ml)
    X_test_seq, y_test_seq = _get_test_sequences(seq_gen, test_ml)
    logger.info(f"  Sequences built: {X_test_seq.shape}")

    # 3. Train LSTM
    logger.info("\n[2/6] Training LSTM")
    lstm = LSTMModel(n_features=seq_gen.n_features_, lookback=LOOKBACK_LONG, name="LSTM")
    lstm.fit(loaders["train"], loaders["val"])
    lstm.save()
    histories["LSTM"] = lstm.history_

    lstm_df = _pred_to_df(lstm, X_test_seq, test_ml, LOOKBACK_LONG)
    lstm_df.to_parquet(out_pred / "lstm_test.parquet")

    # 4. Train GRU
    logger.info("\n[3/6] Training GRU")
    gru = GRUModel(n_features=seq_gen.n_features_, lookback=LOOKBACK_LONG, name="GRU")
    gru.fit(loaders["train"], loaders["val"])
    gru.save()
    histories["GRU"] = gru.history_

    gru_df = _pred_to_df(gru, X_test_seq, test_ml, LOOKBACK_LONG)
    gru_df.to_parquet(out_pred / "gru_test.parquet")

    # 5. Train Hybrid
    logger.info("\n[4/6] Training Hybrid SARIMAX+LSTM")
    _, hybrid_preds = run_hybrid(
        train_ml=train_ml, val_ml=val_ml, test_ml=test_ml,
        sarimax_train_residuals=sarimax_resid,
        sarimax_val=sarimax_val, sarimax_test=sarimax_test,
        lookback=LOOKBACK_LONG,
    )
    hybrid_test = hybrid_preds["test"][["actual","predicted"]]

    # Load Hybrid LSTM history from saved model
    from src.models.lstm import LSTMModel as LM
    hybrid_lstm = LM.load(abs_path("outputs/models/hybrid_lstm_residual.pt"))
    histories["Hybrid_SARIMAX_LSTM"] = hybrid_lstm.history_

    # 6. Evaluate all models
    logger.info("\n[5/6] Evaluating all models")

    # Align XGBoost to same index as DL models (DL loses first `lookback` rows)
    dl_idx   = lstm_df.index
    xgb_aligned = xgb_test.loc[xgb_test.index.isin(dl_idx)].copy()

    all_preds: Dict[str, pd.DataFrame] = {
        "XGBoost":           xgb_aligned,
        "LSTM":              lstm_df,
        "GRU":               gru_df,
        "Hybrid_SARIMAX_LSTM": hybrid_test.loc[hybrid_test.index.isin(dl_idx)],
    }

    results = []
    for mname, df in all_preds.items():
        m = evaluate(
            pd.Series(df["actual"].values, index=df.index),
            pd.Series(df["predicted"].values, index=df.index),
            model_name=mname, index=df.index,
        )
        results.append(m)
    comp = build_comparison_table(results)
    comp.to_csv(out_pred / "model_comparison_table_phase5.csv", index=False)

    logger.info("\n--- PHASE 5 TEST RANKING ---")
    for _, row in comp.iterrows():
        logger.info(
            f"  #{int(row.get('Rank',0))} {row['Model']:25s}  "
            f"RMSE={row['RMSE']:.4f}  NSE={row['NSE']:.4f}  "
            f"RMSE_Extreme={row.get('RMSE_Extreme', float('nan')):.4f}"
        )

    # 7. Figures
    if save_figures:
        logger.info("\n[6/6] Generating figures")
        _plot_training_curves(histories, save=True)
        _plot_full_comparison(all_preds, save=True)
        _plot_dl_overlay(all_preds, save=True)
        _plot_extreme_comparison(all_preds, save=True)

    # 8. Report
    _write_report(comp, histories, save=True)

    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Runtime     : {elapsed:.1f}s")
    logger.info(f"  Best model  : {comp.iloc[0]['Model']} (RMSE={comp.iloc[0]['RMSE']:.4f})")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 5: Deep Learning + Hybrid")
    p.add_argument("--no-save", action="store_true", help="Skip figures")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dl_pipeline(save_figures=not args.no_save)
