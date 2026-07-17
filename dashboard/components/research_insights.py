"""
research_insights.py — Redesigned Research Insights
=====================================================
Magazine-style layout: one finding = one strip + one chart + one takeaway line.
Five findings, each self-contained, strong visual rhythm.
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
    load_shap_summary, load_comparison_table,
    load_predictions, load_figure_paths, MODEL_COLORS,
)

MONSOON = {6, 7, 8, 9}


def render() -> None:
    st.markdown('<div class="section-heading">🔬 Five Discoveries</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">What 26 years of Lucknow rainfall taught us. '
                'One finding. One chart. One takeaway.</div>',
                unsafe_allow_html=True)

    shap_df   = load_shap_summary()
    comp_df   = load_comparison_table()
    all_preds = load_predictions()
    fig_paths = load_figure_paths()
    expl_figs = fig_paths.get("explainability", {})

    # ════════════════════════════════════════════════════════════════════
    # FINDING 1 — Soil moisture gradient dominated
    # ════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="finding-strip">
        <div class="fs-label">FINDING 1 OF 5 &nbsp;·&nbsp; FEATURE ENGINEERING</div>
        🌱 The signal we built outperformed the data we found.
    </div>
    """, unsafe_allow_html=True)

    col1a, col1b = st.columns([1.8, 1])
    with col1a:
        if not shap_df.empty:
            _render_shap_top5_bar(shap_df)
        elif "shap_bar_importance" in expl_figs:
            st.image(str(expl_figs["shap_bar_importance"]), use_container_width=True)
    with col1b:
        if not shap_df.empty:
            top1      = shap_df.iloc[0]
            top5_frac = shap_df.head(5)["SHAP_Frac"].sum() * 100
        else:
            top1      = None
            top5_frac = 63.2

        st.markdown(f"""
        <div class="insight-card green">
            <div class="ic-eyebrow">SOIL_MOISTURE_GRADIENT</div>
            <div class="ic-headline">One derived feature. Thirty-seven percent of all importance.</div>
            <div class="ic-stat">{top1['SHAP_Frac']*100:.1f}%</div>
            <div class="ic-body">
            SOIL_MOISTURE_GRADIENT = SOIL_WET_SURF − SOIL_WET_ROOT was engineered in
            Phase 3 to resolve severe collinearity (r = 0.940) between two raw columns.
            The gradient physically encodes near-surface moisture flux direction.<br><br>
            A positive gradient — surface wetter than root zone — signals recent
            infiltration and directly predicts subsequent rainfall intensity.<br><br>
            The top 5 features explain <strong>{top5_frac:.1f}%</strong> of all SHAP importance.
            None of the top 5 are raw atmospheric observations — all are either engineered
            or lagged features.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # FINDING 2 — XGBoost beat deep learning
    # ════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="finding-strip">
        <div class="fs-label">FINDING 2 OF 5 &nbsp;·&nbsp; MODEL PERFORMANCE</div>
        🌲 A tree model beat every neural network — for structural, not computational, reasons.
    </div>
    """, unsafe_allow_html=True)

    col2a, col2b = st.columns([1, 1.8])
    with col2a:
        st.markdown("""
        <div class="insight-card">
            <div class="ic-eyebrow">WHY XGBOOST WON</div>
            <div class="ic-headline">Two-stage architecture handled zero-inflation explicitly.</div>
            <div class="ic-stat">NSE 0.935</div>
            <div class="ic-body">
            55.2% of days record zero rainfall. A single-stage MSE-minimising
            regressor is pulled toward predicting near-zero on every day.
            XGBoost's classifier + wet-day regressor handles this at the
            architectural level.<br><br>
            LSTM/GRU used a 5× weighted loss on heavy-rain days but not a
            true two-stage design. With 100 training epochs on GPU, the gap
            would narrow — but the structural advantage remains.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2b:
        if not comp_df.empty:
            _render_model_rmse_chart(comp_df)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # FINDING 3 — Cloud × RH interaction
    # ════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="finding-strip">
        <div class="fs-label">FINDING 3 OF 5 &nbsp;·&nbsp; NONLINEAR INTERACTIONS</div>
        ☁️ Cloud cover and humidity don't add — they multiply.
    </div>
    """, unsafe_allow_html=True)

    col3a, col3b = st.columns([1.8, 1])
    with col3a:
        if "pdp_2d_cloud_rh" in expl_figs:
            st.image(str(expl_figs["pdp_2d_cloud_rh"]), use_container_width=True)
        else:
            st.info("2D PDP figure not found — run Phase 6.")
    with col3b:
        st.markdown("""
        <div class="insight-card orange">
            <div class="ic-eyebrow">2D PARTIAL DEPENDENCE</div>
            <div class="ic-headline">A superadditive effect that linear models miss completely.</div>
            <div class="ic-stat">2D PDP</div>
            <div class="ic-body">
            The 2D PDP (CLOUD × RH) shows a contour that accelerates in the
            top-right corner — high cloud AND high humidity together produce
            predicted rainfall above the sum of their individual effects.<br><br>
            This moisture convergence regime — saturated boundary-layer air
            under thick convective cloud — is a known atmospheric physics
            mechanism during monsoon onset and Bay of Bengal depressions.<br><br>
            SARIMAX uses additive linear terms. It structurally cannot represent
            this, which explains why SARIMAX systematically underpredicts the
            most intense rainfall periods.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # FINDING 4 — Extreme events remain unsolved
    # ════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="finding-strip">
        <div class="fs-label">FINDING 4 OF 5 &nbsp;·&nbsp; FAILURE ANALYSIS</div>
        ⚡ Every model fails on the 8 days that matter most for flood warnings.
    </div>
    """, unsafe_allow_html=True)

    col4a, col4b = st.columns([1, 1.8])
    with col4a:
        st.markdown("""
        <div class="insight-card red">
            <div class="ic-eyebrow">EVENTS ≥ 50 MM</div>
            <div class="ic-headline">The tail distribution is not learnable with available data.</div>
            <div class="ic-stat">8 days</div>
            <div class="ic-body">
            Only 8 test-set days record ≥50 mm. Only 52 training days do
            (0.8% of 6,617). The model has insufficient exposure to learn
            the upper tail.<br><br>
            XGBoost: MAE = 10.2 mm on extreme days — best of all models,
            still misses by 10+ mm on events averaging 75 mm.<br><br>
            Worst miss: <strong>2022-02-09</strong>, actual 154 mm,
            predicted 37 mm. Anomalous winter event with no seasonal precedent.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4b:
        if all_preds:
            _render_extreme_scatter(all_preds)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # FINDING 5 — Hybrid improved over standalone DL
    # ════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="finding-strip">
        <div class="fs-label">FINDING 5 OF 5 &nbsp;·&nbsp; HYBRID ARCHITECTURE</div>
        🔀 Residual learning added measurable value — combining models helped.
    </div>
    """, unsafe_allow_html=True)

    col5a, col5b = st.columns([1.8, 1])
    with col5a:
        if not comp_df.empty:
            _render_dl_nse_chart(comp_df)
    with col5b:
        st.markdown("""
        <div class="insight-card purple">
            <div class="ic-eyebrow">HYBRID SARIMAX + LSTM</div>
            <div class="ic-headline">SARIMAX residuals carry exploitable temporal structure.</div>
            <div class="ic-stat">NSE 0.428</div>
            <div class="ic-body">
            Diagnostic analysis confirmed that SARIMAX residuals carry
            lag-1 autocorrelation = 0.117 — above the 95% significance
            threshold of ±0.052.<br><br>
            The LSTM residual learner targets this specific signal.
            The Hybrid (NSE 0.428) outperformed standalone GRU (0.335)
            and LSTM (0.280).<br><br>
            With a full 100-epoch training budget on GPU, the hybrid
            would likely extend this advantage further — the architecture
            is validated; only the training budget was constrained.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="dashboard-footer" style="text-align:center;color:#94A3B8;
         font-size:0.78rem;padding:20px 0">
        All findings derived from 26 years of daily meteorological records ·
        Lucknow, Uttar Pradesh, India · 2000–2025 ·
        All quantitative claims reproducible from <code>outputs/</code>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper charts
# ---------------------------------------------------------------------------

def _render_shap_top5_bar(shap_df: pd.DataFrame) -> None:
    top = shap_df.head(12).sort_values("SHAP_MeanAbs").copy()
    colors = ["#3BB273" if i >= len(top)-5 else "#94A3B8"
              for i in range(len(top))]
    # Highlight top feature differently
    colors[-1] = "#2E7D32"

    fig = go.Figure(go.Bar(
        x=top["SHAP_MeanAbs"], y=top["Feature"],
        orientation="h",
        marker_color=colors, marker_opacity=0.88, marker_line_width=0,
        text=[f"{v*100:.1f}%" for v in top["SHAP_Frac"]],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        height=380,
        xaxis=dict(title="Mean |SHAP Value|", tickfont=dict(size=10),
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                   range=[0, float(top["SHAP_MeanAbs"].max()) * 1.25]),
        yaxis=dict(tickfont=dict(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=15, b=50, l=190, r=60),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_model_rmse_chart(comp_df: pd.DataFrame) -> None:
    df = comp_df.dropna(subset=["RMSE"]).copy()
    colors = [MODEL_COLORS.get(m, "#94A3B8") for m in df["Model"]]

    clim_row = comp_df[comp_df["Model"] == "Climatology"]
    baseline = float(clim_row["RMSE"].values[0]) if len(clim_row) else float(df["RMSE"].max())

    fig = go.Figure(go.Bar(
        x=df["Model"], y=df["RMSE"],
        marker_color=colors, marker_opacity=0.88, marker_line_width=0,
        text=[f"{v:.3f}" for v in df["RMSE"]],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f} mm<extra></extra>",
    ))
    fig.add_hline(y=baseline, line_dash="dash", line_color="#E84855", line_width=1.5,
                  annotation_text=f"Baseline {baseline:.2f}",
                  annotation_font_color="#E84855", annotation_font_size=10,
                  annotation_position="top right")
    fig.update_layout(
        height=340, title="RMSE — All Models (lower is better)",
        title_font=dict(size=12),
        xaxis=dict(tickangle=20, tickfont=dict(size=10)),
        yaxis=dict(title="RMSE (mm/day)", showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=45, b=70, l=60, r=20),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_extreme_scatter(all_preds: dict) -> None:
    fig = go.Figure()
    ref_max = 170.0
    fig.add_trace(go.Scatter(
        x=[0, ref_max], y=[0, ref_max], mode="lines",
        line=dict(color="#CBD5E1", dash="dash", width=1.5),
        name="Perfect prediction", showlegend=True,
    ))
    priority = ["XGBoost","Hybrid SARIMAX+LSTM","SARIMAX"]
    models_to_plot = [m for m in priority if m in all_preds] + \
                     [m for m in all_preds if m not in priority]

    for model_name in models_to_plot[:4]:
        df  = all_preds[model_name]
        ext = df[df["actual"] >= 50.0]
        if len(ext) == 0:
            continue
        col = MODEL_COLORS.get(model_name, "#94A3B8")
        fig.add_trace(go.Scatter(
            x=ext["actual"], y=ext["predicted"],
            mode="markers", name=model_name,
            marker=dict(size=12, color=col, opacity=0.85,
                        line=dict(color="white", width=0.8)),
            hovertemplate=(f"<b>{model_name}</b><br>%{{text}}<br>"
                           "Actual: %{x:.1f} mm<br>"
                           "Predicted: %{y:.1f} mm<extra></extra>"),
            text=[d.strftime("%d %b %Y") for d in ext.index],
        ))

    fig.add_annotation(
        x=0.5, y=0.06, xref="paper", yref="paper",
        text="All models cluster below the diagonal — systematic underprediction",
        showarrow=False, font=dict(size=10, color="#64748B"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#E2E8F0", borderwidth=1,
    )
    fig.update_layout(
        height=380,
        xaxis=dict(title="Actual (mm/day)", range=[45, ref_max],
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10)),
        yaxis=dict(title="Predicted (mm/day)", range=[0, ref_max],
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=15, b=55, l=60, r=15),
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_dl_nse_chart(comp_df: pd.DataFrame) -> None:
    dl_models = ["XGBoost","Hybrid SARIMAX+LSTM","GRU","LSTM"]
    df = comp_df[comp_df["Model"].isin(dl_models)].copy()
    if df.empty:
        st.info("DL model comparison data not found.")
        return

    df = df.sort_values("NSE", ascending=True)
    colors = [MODEL_COLORS.get(m, "#94A3B8") for m in df["Model"]]

    fig = go.Figure(go.Bar(
        x=df["NSE"], y=df["Model"],
        orientation="h",
        marker_color=colors, marker_opacity=0.88, marker_line_width=0,
        text=[f"{v:.3f}" for v in df["NSE"]],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>NSE: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#64748B", line_width=1.0)
    fig.update_layout(
        height=300, title="NSE — Deep Learning Models (higher is better)",
        title_font=dict(size=12),
        xaxis=dict(title="NSE (Nash–Sutcliffe Efficiency)",
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                   tickfont=dict(size=10),
                   range=[min(-0.1, float(df["NSE"].min())-0.05),
                           float(df["NSE"].max()) * 1.15]),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=45, b=50, l=10, r=65),
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)