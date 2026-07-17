"""
home.py — Redesigned Home Page
================================
Layer 1 (5s): Hero banner + best model result + flagship chart
Layer 2 (60s): Pipeline overview + leaderboard + 3 finding cards
Layer 3 (deep): Expanded methodology inside expanders
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.data_loader import (
    load_dataset_stats, load_comparison_table,
    load_predictions, MODEL_COLORS,
)

MONSOON = {6, 7, 8, 9}


def render() -> None:
    stats = load_dataset_stats()
    comp  = load_comparison_table()
    preds = load_predictions()

    best_rmse  = float(comp.iloc[0]["RMSE"])  if not comp.empty else 3.66
    best_nse   = float(comp.iloc[0]["NSE"])   if not comp.empty else 0.935
    best_model = comp.iloc[0]["Model"]         if not comp.empty else "XGBoost"

    # ── HERO ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-banner">
        <h1>☁️ Lucknow Rainfall Forecasting</h1>
        <p class="hero-sub">
            Hybrid Stochastic Hydrology &nbsp;·&nbsp; Deep Learning &nbsp;·&nbsp;
            SHAP Explainability &nbsp;·&nbsp; Uncertainty Quantification
        </p>
        <span class="hero-badge">Best model: {best_model}</span>
        <span class="hero-badge">NSE = {best_nse:.3f}</span>
        <span class="hero-badge">RMSE = {best_rmse:.2f} mm/day</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, f"{stats.get('n_records', 9467):,}", "Daily Records",     "2000–2025", ""),
        (c2, f"{stats.get('n_years', 26)}",        "Years",             "of data",   ""),
        (c3, f"{stats.get('monsoon_frac_pct', 85.2):.1f}%",
                                                    "Monsoon Rain",      "Jun–Sep",   "orange"),
        (c4, f"{best_nse:.3f}",                    "Best NSE",          best_model,  "green"),
        (c5, f"{best_rmse:.2f} mm",                "Best RMSE",         "test set",  ""),
    ]
    for col, val, label, sub, cls in kpi_data:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value {cls}">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── FLAGSHIP CHART ────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Actual vs Predicted Rainfall — Best Model</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">XGBoost on the held-out test set (2022–2025). '
                'Green shading = monsoon (Jun–Sep). ★ = extreme events ≥50 mm. '
                'Use the range slider below the chart to zoom.</div>',
                unsafe_allow_html=True)

    _render_flagship_chart(preds)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── PIPELINE OVERVIEW ─────────────────────────────────────────────────
    st.markdown('<div class="section-heading">System Architecture</div>',
                unsafe_allow_html=True)
    steps = [
        ("📂 Raw Data", False), ("→", True),
        ("🔍 EDA", False), ("→", True),
        ("⚙️ Features", False), ("→", True),
        ("📈 SARIMAX", False), ("→", True),
        ("🌲 XGBoost", False), ("→", True),
        ("🧠 LSTM/GRU", False), ("→", True),
        ("🔀 Hybrid", False), ("→", True),
        ("💡 SHAP", False), ("→", True),
        ("📊 Uncertainty", False),
    ]
    html = ['<div class="pipeline-row">']
    for label, is_arrow in steps:
        if is_arrow:
            html.append(f'<span class="pipeline-arrow">{label}</span>')
        else:
            hl = "highlight" if label in ("🌲 XGBoost", "💡 SHAP") else ""
            html.append(f'<span class="pipeline-node {hl}">{label}</span>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── LEADERBOARD + FINDING CARDS ───────────────────────────────────────
    col_lb, col_cards = st.columns([1.3, 1])

    with col_lb:
        st.markdown('<div class="section-heading">Model Leaderboard</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-sub">All models on the 1,426-day test set, '
                    'ranked by RMSE. Bar shows improvement vs climatology baseline.</div>',
                    unsafe_allow_html=True)
        _render_leaderboard(comp)

    with col_cards:
        st.markdown('<div class="section-heading">Top Discoveries</div>',
                    unsafe_allow_html=True)
        for accent, eyebrow, headline, stat, body in [
            ("green", "FINDING 1", "🌱 One engineered feature dominated",
             "36.7%",
             "SOIL_MOISTURE_GRADIENT explained more than every atmospheric "
             "measurement combined — validating domain-driven feature engineering."),
            ("orange", "FINDING 2", "⚡ Extremes remain unsolved",
             "≥50 mm",
             "All 7 models severely underpredict the largest events. "
             "8 extreme days in the test set remain the open research problem."),
            ("", "FINDING 3", "☁️ Cloud × Humidity superadditive",
             "2D PDP",
             "High cloud AND high humidity produce more rainfall than their "
             "individual effects summed — a signal linear models cannot capture."),
        ]:
            st.markdown(f"""
            <div class="insight-card {accent}">
                <div class="ic-eyebrow">{eyebrow}</div>
                <div class="ic-headline">{headline}</div>
                <div class="ic-stat">{stat}</div>
                <div class="ic-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    # ── DEEP DIVE EXPANDER ────────────────────────────────────────────────
    with st.expander("📋 Project methodology & dataset details"):
        col_m, col_d = st.columns(2)
        with col_m:
            st.markdown("**Modelling strategy**")
            st.markdown("""
- **Zero-inflation handled explicitly:** 55.2% of days record no rain.
  XGBoost uses a two-stage classifier + wet-day regressor.
- **Target transformation:** log(1+rainfall) reduces skewness from 6.8 → 1.8.
- **Temporal integrity:** Strictly chronological 70/15/15 split.
  No random shuffling anywhere in the pipeline.
- **Feature engineering:** 58 features — lag, rolling, cyclical calendar,
  and domain-physics variables (soil moisture gradient, dewpoint).
""")
        with col_d:
            st.markdown("**Dataset**")
            st.markdown(f"""
- **Source:** Lucknow, Uttar Pradesh, India — daily station data
- **Period:** 2000-01-01 to 2025-12-31 ({stats.get('n_records', 9467):,} days)
- **Missing values:** Zero — no imputation required
- **Variables:** RAINFALL, TMAX, TMIN, TAVG, RH, WIND, PRESSURE,
  CLOUD, SOLAR_RAD, SOIL_WET_SURF, SOIL_WET_ROOT
- **Max single-day rainfall:** {stats.get('max_rainfall', 171.9):.1f} mm
- **Monsoon fraction:** {stats.get('monsoon_frac_pct', 85.2):.1f}% of annual total
""")


# ---------------------------------------------------------------------------
# Flagship chart
# ---------------------------------------------------------------------------

def _render_flagship_chart(preds: dict) -> None:
    if "XGBoost" not in preds:
        st.info("XGBoost predictions not found.")
        return

    df = preds["XGBoost"].copy()
    extreme_mask = df["actual"] >= 50

    fig = go.Figure()

    # Monsoon shading
    for yr in range(df.index.year.min(), df.index.year.max() + 2):
        fig.add_vrect(
            x0=f"{yr}-06-01", x1=f"{yr}-09-30",
            fillcolor="rgba(59,178,115,0.08)", layer="below", line_width=0,
        )

    # Actual rainfall — filled area
    fig.add_trace(go.Scatter(
        x=df.index, y=df["actual"],
        fill="tozeroy",
        fillcolor="rgba(46,134,171,0.18)",
        line=dict(color="rgba(46,134,171,0.55)", width=0.8),
        name="Actual rainfall",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.1f} mm<extra></extra>",
    ))

    # XGBoost prediction
    fig.add_trace(go.Scatter(
        x=df.index, y=df["predicted"],
        line=dict(color="#3BB273", width=1.8),
        name="XGBoost predicted",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Predicted: %{y:.1f} mm<extra></extra>",
    ))

    # Extreme events
    if extreme_mask.any():
        ext = df[extreme_mask]
        fig.add_trace(go.Scatter(
            x=ext.index, y=ext["actual"],
            mode="markers+text",
            marker=dict(symbol="star", size=14, color="#E84855",
                        line=dict(color="white", width=0.5)),
            text=[f"{v:.0f}mm" for v in ext["actual"]],
            textposition="top center",
            textfont=dict(size=9, color="#E84855"),
            name="Extreme (≥50 mm)",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.1f} mm"
                          "<br><b>★ Extreme event</b><extra></extra>",
        ))

    # Worst-miss annotation
    worst_idx  = (df["actual"] - df["predicted"]).abs().idxmax()
    worst_act  = df.loc[worst_idx, "actual"]
    worst_pred = df.loc[worst_idx, "predicted"]
    fig.add_annotation(
        x=worst_idx, y=worst_act,
        text=f"Largest miss<br>{worst_act:.0f} mm actual<br>{worst_pred:.0f} mm predicted",
        showarrow=True, arrowhead=2, arrowcolor="#E84855", arrowwidth=1.5,
        ax=70, ay=-55,
        font=dict(size=10, color="#E84855"),
        bgcolor="white", bordercolor="#E84855", borderwidth=1, borderpad=4,
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=50, l=65, r=20),
        legend=dict(orientation="h", y=1.02, x=0,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)"),
        xaxis=dict(
            showgrid=False, tickfont=dict(size=11),
            rangeslider=dict(visible=True, thickness=0.06),
        ),
        yaxis=dict(
            title="Rainfall (mm/day)",
            showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            tickfont=dict(size=11),
        ),
        font=dict(family="system-ui, -apple-system, sans-serif"),
    )
    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text="🟢 Green = Monsoon season (Jun–Sep)",
        showarrow=False, font=dict(size=10, color="#2E7D32"),
        bgcolor="rgba(240,253,244,0.9)",
        bordercolor="#3BB273", borderwidth=1, xanchor="left",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Inline metric chips
    a, p = df["actual"].values, df["predicted"].values
    rmse = float(np.sqrt(np.mean((a - p)**2)))
    mae  = float(np.mean(np.abs(a - p)))
    nse  = float(1 - np.sum((a-p)**2) / np.sum((a-a.mean())**2))
    hit  = float(((a>=0.1)&(p>=0.1)).sum() / max((a>=0.1).sum(),1))

    st.markdown(
        f'<span class="metric-chip">RMSE <span class="chip-val">{rmse:.3f} mm</span></span>'
        f'<span class="metric-chip">MAE <span class="chip-val">{mae:.3f} mm</span></span>'
        f'<span class="metric-chip">NSE <span class="chip-val">{nse:.4f}</span></span>'
        f'<span class="metric-chip">Rain hit-rate '
        f'<span class="chip-val">{hit*100:.1f}%</span></span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _render_leaderboard(comp: pd.DataFrame) -> None:
    if comp.empty:
        st.info("Comparison data not found.")
        return

    clim_row = comp[comp["Model"] == "Climatology"]
    baseline_rmse = float(clim_row["RMSE"].values[0]) if len(clim_row) else float(comp["RMSE"].max())

    rank_icons = ["🥇", "🥈", "🥉"]
    bar_colors = ["#3BB273","#5C6BC0","#8B5E83","#2E86AB","#F4A261","#94A3B8","#CBD5E1"]

    for _, row in comp.head(7).iterrows():
        rank   = int(row.get("Rank", 1))
        model  = row["Model"]
        rmse   = float(row["RMSE"])
        nse    = float(row.get("NSE", 0))
        icon   = rank_icons[rank-1] if rank <= 3 else str(rank)
        winner = "winner" if rank == 1 else ""
        improvement = max(0, (baseline_rmse - rmse) / baseline_rmse * 100)
        bar_pct  = max(4, min(100, int(improvement)))
        bar_col  = bar_colors[min(rank-1, len(bar_colors)-1)]
        rank_cls = {1:"gold",2:"silver",3:"bronze"}.get(rank,"")
        impr_str = f"−{improvement:.0f}% vs baseline" if improvement > 0 else "baseline"

        st.markdown(f"""
        <div class="leaderboard-row {winner}">
            <div class="lb-rank {rank_cls}">{icon}</div>
            <div class="lb-model">{model}</div>
            <div class="lb-metric">RMSE <strong>{rmse:.3f}</strong></div>
            <div class="lb-metric">NSE <strong>{nse:.3f}</strong></div>
            <div class="lb-bar-wrap">
                <div class="lb-bar" style="width:{bar_pct}%;background:{bar_col}"></div>
            </div>
            <div class="lb-pct">{impr_str}</div>
        </div>""", unsafe_allow_html=True)