"""
forecast_panel.py — Redesigned Forecast Panel
===============================================
Layer 1: Large actual-vs-predicted chart with inline regime toggles
Layer 2: Scatter plot (model switcher) + per-season metrics
Layer 3: Error analysis expander
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.data_loader import load_predictions, MODEL_COLORS

MONSOON = {6, 7, 8, 9}
EXTREME_T = 50.0


def render() -> None:
    st.markdown('<div class="section-heading">🔭 Forecast Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Compare model predictions against observed rainfall '
                'on the held-out test set (2022–2025). All predictions are precomputed.</div>',
                unsafe_allow_html=True)

    all_preds = load_predictions()
    if not all_preds:
        st.error("Prediction files not found.")
        return

    # ── Controls ──────────────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        available = list(all_preds.keys())
        selected  = st.multiselect(
            "Models",
            options=available,
            default=[m for m in ["XGBoost","SARIMAX","Hybrid SARIMAX+LSTM"]
                     if m in available],
            label_visibility="collapsed",
        )
    with ctrl2:
        regime = st.radio(
            "Regime",
            ["All days", "Monsoon (JJAS)", "Non-Monsoon", "Extremes ≥50 mm"],
            horizontal=True,
            label_visibility="collapsed",
        )

    if not selected:
        st.info("Select at least one model above.")
        return

    # Build common index
    common_idx = None
    for m in selected:
        idx = all_preds[m].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    # Apply regime filter
    if regime == "Monsoon (JJAS)":
        common_idx = common_idx[common_idx.month.isin(MONSOON)]
    elif regime == "Non-Monsoon":
        common_idx = common_idx[~common_idx.month.isin(MONSOON)]
    elif regime == "Extremes ≥50 mm":
        ref = all_preds[selected[0]].loc[all_preds[selected[0]].index.isin(common_idx), "actual"]
        common_idx = common_idx[ref.values >= EXTREME_T]

    if len(common_idx) == 0:
        st.warning("No data points match the current filter.")
        return

    actual = all_preds[selected[0]].loc[common_idx, "actual"]

    # ── CHART 1: Time-series overlay ──────────────────────────────────────
    st.markdown(f'<div class="section-heading" style="margin-top:16px">'
                f'Actual vs Predicted — {regime} ({len(common_idx):,} days)</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    # Monsoon shading
    if regime == "All days":
        for yr in range(common_idx.year.min(), common_idx.year.max() + 2):
            fig.add_vrect(
                x0=f"{yr}-06-01", x1=f"{yr}-09-30",
                fillcolor="rgba(59,178,115,0.07)", layer="below", line_width=0,
            )

    # Actual
    fig.add_trace(go.Scatter(
        x=common_idx, y=actual.values,
        fill="tozeroy", fillcolor="rgba(46,134,171,0.15)",
        line=dict(color="rgba(46,134,171,0.5)", width=0.8),
        name="Actual",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.1f} mm<extra></extra>",
    ))

    # Each selected model
    for m in selected:
        pred = all_preds[m].loc[common_idx, "predicted"]
        fig.add_trace(go.Scatter(
            x=common_idx, y=pred.values,
            line=dict(color=MODEL_COLORS.get(m, "#333"), width=1.6),
            name=m,
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{m}: %{{y:.1f}} mm<extra></extra>",
        ))

    # Extreme event markers
    extreme_mask = actual >= EXTREME_T
    if extreme_mask.any():
        fig.add_trace(go.Scatter(
            x=common_idx[extreme_mask], y=actual[extreme_mask].values,
            mode="markers",
            marker=dict(symbol="star", size=12, color="#E84855",
                        line=dict(color="white", width=0.5)),
            name="≥50 mm event",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.1f} mm"
                          "<br><b>★ Extreme</b><extra></extra>",
        ))

    fig.update_layout(
        height=520,
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=15, b=55, l=65, r=20),
        legend=dict(orientation="h", y=1.02, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=11),
                   rangeslider=dict(visible=True, thickness=0.05)),
        yaxis=dict(title="Rainfall (mm/day)", showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11)),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Inline metric chips ───────────────────────────────────────────────
    chips = ""
    for m in selected:
        a = actual.values
        p = all_preds[m].loc[common_idx, "predicted"].values
        rmse = float(np.sqrt(np.mean((a-p)**2)))
        nse_val = float(1 - np.sum((a-p)**2) / max(np.sum((a-a.mean())**2), 1e-9))
        col = MODEL_COLORS.get(m, "#333")
        chips += (f'<span class="metric-chip" style="border-color:{col}">'
                  f'<span style="color:{col};font-weight:700">{m}</span>'
                  f' &nbsp;RMSE <span class="chip-val">{rmse:.3f}</span>'
                  f' &nbsp;NSE <span class="chip-val">{nse_val:.3f}</span></span> ')
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── CHART 2: Scatter (model selector) ────────────────────────────────
    col_sc, col_info = st.columns([2, 1])

    with col_sc:
        scatter_model = st.selectbox("Scatter model", selected, label_visibility="collapsed")
        _render_scatter(all_preds, scatter_model, common_idx, actual)

    with col_info:
        st.markdown('<div class="section-heading" style="font-size:1.05rem;margin-top:8px">'
                    'Performance Summary</div>', unsafe_allow_html=True)
        m = scatter_model
        a = actual.values
        p = all_preds[m].loc[common_idx, "predicted"].values
        metrics = [
            ("RMSE", f"{float(np.sqrt(np.mean((a-p)**2))):.4f} mm"),
            ("MAE",  f"{float(np.mean(np.abs(a-p))):.4f} mm"),
            ("NSE",  f"{float(1-np.sum((a-p)**2)/max(np.sum((a-a.mean())**2),1e-9)):.4f}"),
            ("Bias", f"{float(np.mean(p-a)):+.4f} mm"),
            ("N days", f"{len(a):,}"),
        ]
        for label, val in metrics:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;
                padding:8px 12px;background:white;border-radius:8px;margin-bottom:6px;
                border:1px solid #E8EDF5;font-size:0.88rem">
                <span style="color:#64748B">{label}</span>
                <strong style="color:#0F172A">{val}</strong>
            </div>""", unsafe_allow_html=True)

        # Seasonal split
        st.markdown('<div style="margin-top:16px;font-size:0.85rem;font-weight:600;'
                    'color:#64748B;text-transform:uppercase;letter-spacing:0.5px">'
                    'By Season</div>', unsafe_allow_html=True)
        df_eval = pd.DataFrame({"actual": a, "predicted": p}, index=common_idx)
        for label, mon_mask in [("Monsoon", df_eval.index.month.isin(MONSOON)),
                                  ("Non-Monsoon", ~df_eval.index.month.isin(MONSOON))]:
            if mon_mask.sum() == 0:
                continue
            ae = df_eval.loc[mon_mask, "actual"].values
            pe = df_eval.loc[mon_mask, "predicted"].values
            r  = float(np.sqrt(np.mean((ae-pe)**2)))
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:6px 12px;'
                f'background:#F8FAFC;border-radius:6px;margin-bottom:5px;font-size:0.83rem">'
                f'<span>{label} (N={mon_mask.sum():,})</span>'
                f'<strong>RMSE {r:.3f}</strong></div>',
                unsafe_allow_html=True,
            )

    # ── Error analysis expander ───────────────────────────────────────────
    with st.expander("🔬 Error analysis — residuals & extreme events"):
        col_r, col_e = st.columns(2)

        with col_r:
            st.markdown("**Residual distribution**")
            fig_r = go.Figure()
            for m in selected:
                a = actual.values
                p = all_preds[m].loc[common_idx, "predicted"].values
                resid = p - a
                fig_r.add_trace(go.Histogram(
                    x=resid, name=m, nbinsx=50, opacity=0.65,
                    marker_color=MODEL_COLORS.get(m, "#333"),
                    hovertemplate=f"{m}: %{{x:.1f}} mm<extra></extra>",
                ))
            fig_r.add_vline(x=0, line_dash="dash", line_color="#64748B")
            fig_r.update_layout(
                barmode="overlay", height=320,
                xaxis_title="Residual (predicted − actual, mm)",
                yaxis_title="Count",
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10, b=50, l=55, r=10),
                legend=dict(font=dict(size=10)),
                font=dict(family="system-ui, sans-serif"),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with col_e:
            st.markdown("**Extreme event performance (≥50 mm)**")
            extreme_idx = common_idx[actual >= EXTREME_T]
            if len(extreme_idx) == 0:
                st.info("No extreme events in current filter.")
            else:
                fig_e = go.Figure()
                lim = float(actual[actual >= EXTREME_T].max()) * 1.1
                fig_e.add_trace(go.Scatter(
                    x=[0, lim], y=[0, lim], mode="lines",
                    line=dict(color="#94A3B8", dash="dash", width=1),
                    showlegend=False,
                ))
                for m in selected:
                    ae = actual[actual >= EXTREME_T].values
                    pe = all_preds[m].loc[actual[actual >= EXTREME_T].index, "predicted"].values
                    fig_e.add_trace(go.Scatter(
                        x=ae, y=pe, mode="markers",
                        name=m,
                        marker=dict(size=10, color=MODEL_COLORS.get(m, "#333"),
                                    opacity=0.8, line=dict(color="white", width=0.5)),
                        hovertemplate=f"{m}<br>Actual: %{{x:.1f}}<br>Predicted: %{{y:.1f}}<extra></extra>",
                    ))
                fig_e.update_layout(
                    height=320,
                    xaxis=dict(title="Actual (mm)", range=[0, lim]),
                    yaxis=dict(title="Predicted (mm)", range=[0, lim]),
                    plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=10, b=50, l=55, r=10),
                    legend=dict(font=dict(size=10)),
                    font=dict(family="system-ui, sans-serif"),
                )
                st.plotly_chart(fig_e, use_container_width=True)


def _render_scatter(all_preds, model_name, common_idx, actual):
    df_sc = pd.DataFrame({
        "actual":    actual.values,
        "predicted": all_preds[model_name].loc[common_idx, "predicted"].values,
        "month":     common_idx.month,
    })
    df_sc["season"] = df_sc["month"].map(
        lambda m: "Monsoon" if m in MONSOON else "Non-Monsoon"
    )
    color   = MODEL_COLORS.get(model_name, "#333")
    a, p    = df_sc["actual"].values, df_sc["predicted"].values
    r_val   = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 else 0.0
    lim     = max(float(a.max()), float(p.max())) * 1.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim], mode="lines",
        line=dict(color="#CBD5E1", dash="dash", width=1.2),
        showlegend=False,
    ))
    for season, s_color in [("Non-Monsoon","#F4A261"),("Monsoon","#3BB273")]:
        mask = df_sc["season"] == season
        fig.add_trace(go.Scatter(
            x=df_sc.loc[mask,"actual"], y=df_sc.loc[mask,"predicted"],
            mode="markers",
            marker=dict(size=4, color=s_color, opacity=0.35,
                        line=dict(width=0)),
            name=season,
            hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
        ))
    fig.add_annotation(
        x=0.05, y=0.93, xref="paper", yref="paper",
        text=f"<b>r = {r_val:.3f}</b>",
        showarrow=False, font=dict(size=13, color=color),
        bgcolor="white", bordercolor=color, borderwidth=1, borderpad=5,
    )
    fig.update_layout(
        height=400, title=f"{model_name} — Actual vs Predicted",
        title_font=dict(size=13),
        xaxis=dict(title="Actual (mm/day)", range=[0,lim], tickfont=dict(size=10)),
        yaxis=dict(title="Predicted (mm/day)", range=[0,lim], tickfont=dict(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=45, b=50, l=60, r=15),
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)