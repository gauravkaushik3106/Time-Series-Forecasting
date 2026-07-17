"""
uncertainty_panel.py — Redesigned Uncertainty Panel
=====================================================
Layer 1: Interval fan chart + 3 KPI chips
Layer 2: Reliability diagram + seasonal uncertainty profile
Layer 3: Calibration diagnosis + conformal prediction guide in expanders
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

from dashboard.data_loader import (
    load_mc_dropout_predictions, load_calibration_metrics, load_figure_paths,
)

MONSOON = {6, 7, 8, 9}
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def render() -> None:
    st.markdown('<div class="section-heading">📉 Uncertainty Quantification</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Monte Carlo Dropout applied to the GRU model '
                '(T = 100 stochastic forward passes). Prediction intervals at 70%, 80%, and 90% '
                'nominal coverage levels on the 1,366-day test set.</div>',
                unsafe_allow_html=True)

    pred_df  = load_mc_dropout_predictions()
    calib_df = load_calibration_metrics()
    fig_paths = load_figure_paths()
    unc_figs  = fig_paths.get("uncertainty", {})

    # Compute summary stats
    ece     = float(calib_df["calibration_error"].mean()) if not calib_df.empty else None
    cov_90  = None
    if not calib_df.empty and "empirical_coverage" in calib_df.columns:
        r90 = calib_df[calib_df["nominal_coverage"].round(2) == 0.90]
        if len(r90):
            cov_90 = float(r90["empirical_coverage"].values[0])

    mon_std  = float(pred_df.loc[pred_df.index.month.isin(MONSOON), "std_mm"].mean()) \
               if not pred_df.empty else None
    nmon_std = float(pred_df.loc[~pred_df.index.month.isin(MONSOON), "std_mm"].mean()) \
               if not pred_df.empty else None
    ratio    = (mon_std / nmon_std) if (mon_std and nmon_std and nmon_std > 0) else None

    # ── KPI chips ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    chip_data = [
        (c1, f"{cov_90*100:.1f}%" if cov_90 else "N/A",
             "90% PI Coverage", "nominal = 90%", "orange" if cov_90 and cov_90 < 0.5 else ""),
        (c2, f"{ece:.3f}" if ece else "N/A",
             "ECE", "calibration error", "orange"),
        (c3, f"{mon_std:.3f} mm" if mon_std else "N/A",
             "Monsoon Std Dev", "predictive uncertainty", ""),
        (c4, f"{ratio:.1f}×" if ratio else "N/A",
             "Monsoon/Non-Monsoon", "relative uncertainty", ""),
    ]
    for col, val, label, sub, cls in chip_data:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-value {cls}" style="font-size:1.9rem">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── CHART 1: Interactive interval fan chart ───────────────────────────
    st.markdown('<div class="section-heading">GRU Prediction Intervals</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Nested fan chart: 70% / 80% / 90% nominal coverage bands. '
                'Use the slider to zoom. ★ = actual values that exceeded the 90% upper bound.</div>',
                unsafe_allow_html=True)

    if not pred_df.empty:
        zoom = st.slider("Months to display from test start",
                         min_value=3, max_value=47, value=18, step=3)
        t_end = pred_df.index[0] + pd.DateOffset(months=zoom)
        view  = pred_df.loc[pred_df.index <= t_end].copy()
        _render_interval_chart(view)
    else:
        st.info("MC Dropout predictions not found. Run Phase 6.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── CHART 2+3: Reliability diagram + seasonal profile ─────────────────
    col_rel, col_sea = st.columns(2)

    with col_rel:
        st.markdown('<div class="section-heading" style="font-size:1.1rem">'
                    'Reliability Diagram</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Perfect calibration = all points on the diagonal. '
                    'Points below = over-confident (intervals too narrow).</div>',
                    unsafe_allow_html=True)
        if not calib_df.empty:
            _render_reliability_plotly(calib_df, ece)
        elif "calibration_reliability_diagram" in unc_figs:
            st.image(str(unc_figs["calibration_reliability_diagram"]),
                     use_container_width=True)

    with col_sea:
        st.markdown('<div class="section-heading" style="font-size:1.1rem">'
                    'Seasonal Uncertainty</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Monthly mean predictive std dev. '
                    'Monsoon months highlighted in green.</div>',
                    unsafe_allow_html=True)
        if not pred_df.empty:
            _render_seasonal_uncertainty(pred_df)
        elif "uncertainty_seasonal_profile" in unc_figs:
            st.image(str(unc_figs["uncertainty_seasonal_profile"]),
                     use_container_width=True)

    # ── Calibration diagnosis expander ────────────────────────────────────
    with st.expander("🔬 Why are the intervals too narrow? — Technical diagnosis"):
        st.markdown(f"""
        <div class="warn-inline">
        <strong>ECE = {ece:.3f}</strong> &nbsp;|&nbsp;
        90% PI empirical coverage = {f'{cov_90*100:.1f}%' if cov_90 else 'N/A'} (nominal = 90%)<br>
        The model is severely over-confident. This is a known, documented failure mode —
        not a code bug. Root causes are explained below.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Root cause 1 — Single-layer GRU architecture**

        PyTorch sets inter-layer dropout to `0.0` when `num_layers=1`. Only the
        fully-connected head dropout fires during MC inference. The FC head perturbation
        is small relative to the recurrent computation, producing log-scale std ≈ 0.036.
        After `expm1()` back-transformation, most intervals are sub-millimetre on a target
        ranging 0–154 mm.

        **Root cause 2 — 20-epoch training budget (CPU constraint)**

        MC Dropout interval width reflects the learned weight uncertainty. An undertrained
        model's weights have not converged to a regime where dropout stochasticity spans
        the true predictive range. The intended training budget was 100 epochs on GPU.

        **What the results still confirm (relative ordering is correct)**

        Despite absolute miscalibration, the *relative* uncertainty ordering is correct:
        - Monsoon uncertainty is **{f'{ratio:.1f}×' if ratio else 'N/A'} higher** than non-monsoon
        - Heteroscedasticity is directionally correct (uncertainty grows with rainfall intensity)
        - The model correctly identifies the monsoon season as more uncertain
        """)

    # ── Conformal prediction guide ─────────────────────────────────────────
    with st.expander("✅ Recommended fix: Conformal Prediction"):
        st.markdown("""
        Conformal prediction produces **theoretically guaranteed coverage** with any model,
        requiring no architectural changes and no retraining:

        ```python
        # Step 1: Compute nonconformity scores on calibration set
        scores = np.abs(y_cal - model.predict(X_cal))

        # Step 2: Find the quantile at the desired coverage level
        alpha = 0.10   # for 90% coverage
        q = np.quantile(scores, (1 - alpha) * (1 + 1/len(scores)))

        # Step 3: Apply to test set
        y_hat = model.predict(X_test)
        lower = y_hat - q
        upper = y_hat + q
        # Guaranteed: P(y_test in [lower, upper]) >= 1 - alpha
        ```

        **Why this works:** The coverage guarantee holds under exchangeability (approximately
        satisfied by meteorological data within a season), regardless of the underlying model
        family. It is distribution-free and computationally trivial.

        **Alternative: XGBoost quantile regression**
        ```python
        # Train two additional XGBoost models at the 5th and 95th percentiles
        model_low  = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.05)
        model_high = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.95)
        ```
        This produces direct 90% prediction intervals calibrated to the training distribution.
        """)

    # ── CRPS figure ────────────────────────────────────────────────────────
    with st.expander("📊 CRPS breakdown by season and intensity"):
        st.markdown("""
        <div class="info-inline">
        <strong>CRPS (Continuous Ranked Probability Score)</strong> rewards both calibration
        and sharpness simultaneously. Lower = better probabilistic forecast.
        CRPS = E[|X−y|] − 0.5 × E[|X−X′|] where X, X′ are independent draws from the
        predictive distribution.
        </div>
        """, unsafe_allow_html=True)
        if "calibration_crps" in unc_figs:
            st.image(str(unc_figs["calibration_crps"]), use_container_width=True)
        else:
            st.info("CRPS figure not found.")


# ---------------------------------------------------------------------------
# Helper charts
# ---------------------------------------------------------------------------

def _render_interval_chart(view: pd.DataFrame) -> None:
    fig = go.Figure()

    # Monsoon shading
    for yr in range(view.index.year.min(), view.index.year.max() + 2):
        fig.add_vrect(
            x0=f"{yr}-06-01", x1=f"{yr}-09-30",
            fillcolor="rgba(59,178,115,0.07)", layer="below", line_width=0,
        )

    # Nested bands — widest first (90%), then 80%, then 70%
    band_configs = [
        ("lower_90", "upper_90", "90% PI", "rgba(46,134,171,0.10)"),
        ("lower_80", "upper_80", "80% PI", "rgba(46,134,171,0.16)"),
        ("lower_70", "upper_70", "70% PI", "rgba(46,134,171,0.24)"),
    ]
    for lo_col, hi_col, label, fill_col in band_configs:
        if lo_col in view.columns and hi_col in view.columns:
            fig.add_trace(go.Scatter(
                x=list(view.index) + list(view.index[::-1]),
                y=list(view[hi_col]) + list(view[lo_col][::-1]),
                fill="toself", fillcolor=fill_col,
                line=dict(width=0), name=label,
                hoverinfo="skip",
            ))

    # Mean prediction
    fig.add_trace(go.Scatter(
        x=view.index, y=view["mean_mm"],
        line=dict(color="#2E86AB", width=1.8),
        name="GRU mean",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>GRU mean: %{y:.2f} mm<extra></extra>",
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=view.index, y=view["actual"],
        line=dict(color="#E84855", width=1.0),
        name="Actual",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.1f} mm<extra></extra>",
    ))

    # Points outside 90% upper bound
    if "upper_90" in view.columns:
        exceed = view["actual"] > view["upper_90"]
        if exceed.any():
            fig.add_trace(go.Scatter(
                x=view.index[exceed], y=view.loc[exceed, "actual"],
                mode="markers",
                marker=dict(symbol="star", size=10, color="#E84855",
                            line=dict(color="white", width=0.5)),
                name=f"Outside 90% PI ({exceed.sum()} days)",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.1f} mm"
                              "<br>★ Outside 90% bound<extra></extra>",
            ))

    fig.update_layout(
        height=500,
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=15, b=55, l=65, r=20),
        xaxis=dict(showgrid=False, tickfont=dict(size=11),
                   rangeslider=dict(visible=True, thickness=0.05)),
        yaxis=dict(title="Rainfall (mm/day)", showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11)),
        legend=dict(orientation="h", y=1.02, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)"),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_reliability_plotly(calib_df: pd.DataFrame, ece: float) -> None:
    nominal   = calib_df["nominal_coverage"].values
    empirical = calib_df["empirical_coverage"].values

    fig = go.Figure()
    # Perfect calibration reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#CBD5E1", dash="dash", width=1.5),
        name="Perfect calibration",
        hoverinfo="skip",
    ))
    # Over/under-confident shading
    fig.add_trace(go.Scatter(
        x=list(nominal) + list(nominal[::-1]),
        y=list(nominal) + list(empirical[::-1]),
        fill="toself", fillcolor="rgba(248,113,113,0.12)",
        line=dict(width=0), name="Over-confident zone",
        hoverinfo="skip",
    ))
    # Actual calibration curve
    fig.add_trace(go.Scatter(
        x=nominal, y=empirical,
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2.2),
        marker=dict(size=8, color="#2E86AB",
                    line=dict(color="white", width=1.5)),
        name="GRU MC Dropout",
        hovertemplate="Nominal: %{x:.0%}<br>Empirical: %{y:.1%}<extra></extra>",
    ))
    fig.add_annotation(
        x=0.5, y=0.92, xref="paper", yref="paper",
        text=f"ECE = {ece:.3f}",
        showarrow=False,
        font=dict(size=13, color="#E84855", family="system-ui"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E84855", borderwidth=1.5, borderpad=6,
    )
    fig.update_layout(
        height=360,
        xaxis=dict(title="Nominal Coverage", tickformat=".0%",
                   range=[0,1], tickfont=dict(size=11)),
        yaxis=dict(title="Empirical Coverage", tickformat=".0%",
                   range=[0,1], tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=15, b=55, l=65, r=15),
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_seasonal_uncertainty(pred_df: pd.DataFrame) -> None:
    monthly_std = pred_df["std_mm"].groupby(pred_df.index.month).mean()

    bar_colors = ["#3BB273" if m in MONSOON else "#2E86AB" for m in range(1, 13)]
    vals       = [float(monthly_std.get(m, 0)) for m in range(1, 13)]

    fig = go.Figure(go.Bar(
        x=MONTH_ABBR, y=vals,
        marker_color=bar_colors, marker_opacity=0.85,
        marker_line_width=0,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside", textfont=dict(size=9),
        hovertemplate="%{x}: std = %{y:.3f} mm<extra></extra>",
    ))
    fig.add_vrect(x0=4.5, x1=8.5, fillcolor="rgba(59,178,115,0.07)",
                  layer="below", line_width=0)
    fig.update_layout(
        height=360,
        yaxis=dict(title="Mean predictive std (mm)", tickfont=dict(size=10),
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        xaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=15, b=50, l=65, r=20),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)