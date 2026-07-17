"""
model_comparison.py (dashboard component)
==========================================
Model comparison panel: ranking table, multi-metric bars, Taylor diagram,
seasonal performance, and extreme-event breakdown.
Loads precomputed comparison CSVs and prediction Parquets only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.data_loader import (
    load_comparison_table, load_predictions, MODEL_COLORS
)

MONSOON_MONTHS    = {6, 7, 8, 9}
EXTREME_THRESHOLD = 50.0


def render() -> None:
    st.markdown('<div class="section-heading">🏆 Model Comparison</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subheading">'
        'All metrics computed on the held-out test set. '
        'Rankings are by RMSE (lower = better). '
        'Seasonal and extreme-event breakdowns reveal where each model excels or fails.'
        '</div>',
        unsafe_allow_html=True,
    )

    comp_df  = load_comparison_table()
    all_preds = load_predictions()

    if comp_df.empty:
        st.error("Comparison table not found.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Ranking Table", "📊 Metric Bars",
        "🌍 Taylor Diagram", "🌧️ Seasonal & Extreme"
    ])

    # ── Tab 1: Ranking table ──────────────────────────────────────────────
    with tab1:
        display_cols = [c for c in [
            "Rank", "Model", "RMSE", "MAE", "R2", "NSE",
            "MAPE_wet", "Bias", "HitRate",
        ] if c in comp_df.columns]

        styled = comp_df[display_cols].style.format({
            c: "{:.4f}" for c in display_cols
            if c not in ("Rank", "Model")
        }).background_gradient(
            subset=["RMSE", "MAE"] if "RMSE" in display_cols else [],
            cmap="RdYlGn_r",
        ).background_gradient(
            subset=["NSE", "R2"] if "NSE" in display_cols else [],
            cmap="RdYlGn",
        )
        st.dataframe(styled, use_container_width=True, height=320)

        if "RMSE" in comp_df.columns:
            best  = comp_df.iloc[0]
            worst = comp_df.iloc[-1]
            improvement = (worst["RMSE"] - best["RMSE"]) / worst["RMSE"] * 100

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best['Model']}</div>
                <div class="metric-label">Best Model</div>
                <div class="metric-sub">RMSE = {best['RMSE']:.4f} mm/day</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{improvement:.1f}%</div>
                <div class="metric-label">Best vs Worst RMSE improvement</div>
                <div class="metric-sub">{worst['Model']} → {best['Model']}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best.get('NSE', float('nan')):.4f}</div>
                <div class="metric-label">Best NSE</div>
                <div class="metric-sub">Nash–Sutcliffe Efficiency</div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 2: Metric bar charts ──────────────────────────────────────────
    with tab2:
        metric_pairs = [
            ("RMSE", "RMSE (mm/day)", True),
            ("MAE",  "MAE (mm/day)",  True),
            ("NSE",  "NSE",           False),
            ("R2",   "R²",            False),
        ]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[p[1] for p in metric_pairs],
            vertical_spacing=0.16, horizontal_spacing=0.12,
        )

        for idx, (metric, ylabel, lower_better) in enumerate(metric_pairs):
            if metric not in comp_df.columns:
                continue
            row = idx // 2 + 1
            col = idx % 2 + 1
            df_sub = comp_df.dropna(subset=[metric]).copy()
            colors = [MODEL_COLORS.get(m, "#333333") for m in df_sub["Model"]]

            fig.add_trace(
                go.Bar(
                    x=df_sub["Model"],
                    y=df_sub[metric],
                    marker_color=colors,
                    marker_opacity=0.82,
                    text=[f"{v:.3f}" for v in df_sub[metric]],
                    textposition="outside",
                    textfont=dict(size=9),
                    showlegend=False,
                    name=metric,
                ),
                row=row, col=col,
            )
            # Highlight best bar
            if lower_better:
                best_idx = int(df_sub[metric].idxmin())
            else:
                best_idx = int(df_sub[metric].idxmax())
            best_model = df_sub.loc[best_idx, "Model"]
            fig.add_annotation(
                x=best_model, y=df_sub.loc[best_idx, metric],
                text="★", showarrow=False,
                font=dict(size=14, color="red"),
                row=row, col=col,
            )

        fig.update_layout(
            height=560,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=60, b=50, l=60, r=20),
        )
        fig.update_xaxes(tickangle=25, tickfont=dict(size=9))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("★ = best model per metric")

    # ── Tab 3: Taylor diagram ─────────────────────────────────────────────
    with tab3:
        if not all_preds:
            st.info("Prediction files required for Taylor diagram.")
        else:
            st.markdown(
                '<div class="info-box">📐 <strong>Taylor Diagram</strong> — '
                'Angle = arccos(correlation coefficient) | '
                'Radius = normalised standard deviation (σ_model / σ_obs). '
                'Ideal model sits at (r=1, σ_norm=1) — the black star.</div>',
                unsafe_allow_html=True,
            )
            _render_taylor_plotly(all_preds)

    # ── Tab 4: Seasonal & extreme breakdown ───────────────────────────────
    with tab4:
        season_cols = [c for c in comp_df.columns if "Monsoon" in c or "Extreme" in c]
        if not season_cols:
            st.info("Seasonal metrics not available in comparison table.")
        else:
            col_s, col_e = st.columns(2)

            # Seasonal RMSE grouped bar
            with col_s:
                if "RMSE_Monsoon" in comp_df.columns and "RMSE_NonMonsoon" in comp_df.columns:
                    df_s = comp_df.dropna(subset=["RMSE_Monsoon"]).copy()
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Bar(
                        x=df_s["Model"], y=df_s["RMSE_Monsoon"],
                        name="Monsoon RMSE",
                        marker_color="#3BB273", opacity=0.82,
                    ))
                    fig_s.add_trace(go.Bar(
                        x=df_s["Model"], y=df_s["RMSE_NonMonsoon"],
                        name="Non-Monsoon RMSE",
                        marker_color="#F4A261", opacity=0.82,
                    ))
                    fig_s.update_layout(
                        title="RMSE by Season",
                        barmode="group",
                        xaxis_tickangle=20,
                        yaxis_title="RMSE (mm/day)",
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=360,
                        margin=dict(t=50, b=60, l=60, r=20),
                        legend=dict(orientation="h", y=1.08),
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

            # Extreme event RMSE
            with col_e:
                if "RMSE_Extreme" in comp_df.columns:
                    df_e = comp_df.dropna(subset=["RMSE_Extreme"]).copy()
                    fig_e = go.Figure(go.Bar(
                        x=df_e["Model"], y=df_e["RMSE_Extreme"],
                        marker_color=[MODEL_COLORS.get(m, "#333") for m in df_e["Model"]],
                        marker_opacity=0.82,
                        text=[f"{v:.1f}" for v in df_e["RMSE_Extreme"]],
                        textposition="outside",
                    ))
                    fig_e.update_layout(
                        title=f"RMSE on Extreme Events (≥{EXTREME_THRESHOLD:.0f} mm)",
                        xaxis_tickangle=20,
                        yaxis_title="RMSE (mm/day)",
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=360,
                        margin=dict(t=50, b=60, l=60, r=20),
                    )
                    st.plotly_chart(fig_e, use_container_width=True)

            # Compact extreme event table
            if any(c in comp_df.columns for c in ["RMSE_Extreme", "MAE_Extreme"]):
                ext_cols = ["Model"] + [c for c in ["RMSE_Extreme","MAE_Extreme","Bias_Extreme","N_extreme"]
                                         if c in comp_df.columns]
                st.dataframe(
                    comp_df[ext_cols].dropna().style.format({
                        c: "{:.4f}" for c in ext_cols
                        if c not in ("Model", "N_extreme")
                    }),
                    use_container_width=True,
                )


def _render_taylor_plotly(all_preds: dict) -> None:
    """Interactive polar Taylor diagram using Plotly."""
    # Use a common reference (first model's actual series)
    first_df   = list(all_preds.values())[0]
    obs        = first_df["actual"].values
    obs_std    = obs.std()

    theta_vals, r_vals, labels, colors = [], [], [], []
    for model_name, df in all_preds.items():
        pred = df["predicted"].values
        # Align lengths
        n = min(len(obs), len(pred))
        if n == 0:
            continue
        r  = float(np.corrcoef(obs[:n], pred[:n])[0, 1])
        rn = pred[:n].std() / (obs_std + 1e-10)
        theta_vals.append(float(np.degrees(np.arccos(np.clip(r, -1, 1)))))
        r_vals.append(rn)
        labels.append(f"{model_name}<br>r={r:.3f}")
        colors.append(MODEL_COLORS.get(model_name, "#333333"))

    fig = go.Figure()
    # Reference point (observations)
    fig.add_trace(go.Scatterpolar(
        r=[1.0], theta=[0],
        mode="markers+text",
        marker=dict(symbol="star", size=18, color="black"),
        text=["Obs"], textposition="top right",
        name="Observations",
        showlegend=True,
    ))
    # Model points
    for i, (th, rv, lab, col) in enumerate(
            zip(theta_vals, r_vals, labels, colors)):
        fig.add_trace(go.Scatterpolar(
            r=[rv], theta=[th],
            mode="markers+text",
            marker=dict(size=13, color=col),
            text=[lab.split("<br>")[0]],
            textposition="top right",
            textfont=dict(size=9),
            name=lab.split("<br>")[0],
            showlegend=True,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(title="Normalised Std Dev", range=[0, 2]),
            angularaxis=dict(
                direction="counterclockwise",
                tickmode="array",
                tickvals=list(range(0, 91, 10)),
                ticktext=[f"r={np.cos(np.radians(v)):.2f}" for v in range(0, 91, 10)],
                thetaunit="degrees",
            ),
        ),
        title="Taylor Diagram — Test Set<br>"
              "<sup>Angle = 1−r | Radius = σ_model/σ_obs | Star = observations</sup>",
        height=500,
        margin=dict(t=80, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)