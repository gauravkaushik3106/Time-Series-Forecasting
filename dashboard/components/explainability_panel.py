"""
explainability_panel.py — Redesigned Explainability Panel
===========================================================
Layer 1: Hero finding card + interactive SHAP importance bar
Layer 2: SHAP beeswarm (enlarged)
Layer 3: Dependence plots + waterfall in expanders
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

from dashboard.data_loader import load_shap_summary, load_figure_paths

FEATURE_GROUPS = {
    "SOIL_MOISTURE_GRADIENT":    ("Soil Physics",    "#3BB273"),
    "SOIL_WET_SURF":             ("Soil Physics",    "#3BB273"),
    "SOIL_WET_ROOT":             ("Soil Physics",    "#3BB273"),
    "SOIL_WET_SURF_lag1":        ("Soil Lag",        "#5CB85C"),
    "SOIL_WET_SURF_lag3":        ("Soil Lag",        "#5CB85C"),
    "SOIL_WET_SURF_roll_mean_7": ("Soil Rolling",    "#8BC34A"),
    "RAINFALL_roll_std_7":       ("Rainfall Hist.",  "#2E86AB"),
    "RAINFALL_roll_mean_7":      ("Rainfall Hist.",  "#2E86AB"),
    "RAINFALL_roll_mean_14":     ("Rainfall Hist.",  "#2E86AB"),
    "RAINFALL_lag1":             ("Rainfall Lag",    "#64B5F6"),
    "CLOUD":                     ("Atmospheric",     "#F4A261"),
    "RH":                        ("Atmospheric",     "#F4A261"),
    "RH_lag1":                   ("Atmospheric",     "#F4A261"),
    "DEWPOINT_APPROX":           ("Atmospheric",     "#F4A261"),
    "CLOUD_lag1":                ("Atmospheric",     "#F4A261"),
    "PRESSURE_RH_INTERACTION":   ("Atmospheric",     "#F4A261"),
    "MONTH_COS":                 ("Temporal",        "#8B5E83"),
    "MONTH_SIN":                 ("Temporal",        "#8B5E83"),
    "DOY_SIN":                   ("Temporal",        "#8B5E83"),
    "DOY_COS":                   ("Temporal",        "#8B5E83"),
}


def render() -> None:
    st.markdown('<div class="section-heading">💡 Model Explainability</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">SHAP (SHapley Additive exPlanations) decomposes each '
                'XGBoost prediction into contributions from individual features. '
                'Applied to 491 wet test days using TreeExplainer (exact, no approximation).'
                '</div>', unsafe_allow_html=True)

    shap_df   = load_shap_summary()
    fig_paths = load_figure_paths()
    expl_figs = fig_paths.get("explainability", {})

    if shap_df.empty:
        st.error("SHAP summary not found. Run Phase 6 first.")
        return

    top1       = shap_df.iloc[0]
    top5_frac  = shap_df.head(5)["SHAP_Frac"].sum() * 100

    # ── Hero finding ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="finding-strip">
        <div class="fs-label">KEY FINDING — SHAP GLOBAL IMPORTANCE</div>
        🌱 One engineered feature — <strong>SOIL_MOISTURE_GRADIENT</strong> — explains
        <strong>{top1['SHAP_Frac']*100:.1f}%</strong> of all wet-day prediction variance.
        More than cloud cover, humidity, and all rainfall lag features combined.
        The top 5 features account for <strong>{top5_frac:.1f}%</strong> of total SHAP importance.
    </div>
    """, unsafe_allow_html=True)

    # ── Primary: Interactive SHAP importance bar ──────────────────────────
    st.markdown('<div class="section-heading">Global Feature Importance</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Mean |SHAP value| across all 491 wet test days. '
                'Longer bar = more influence on the prediction. '
                'Colour = feature group. Hover for exact values.</div>',
                unsafe_allow_html=True)

    _render_shap_interactive_bar(shap_df)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Secondary: SHAP beeswarm (enlarged) ──────────────────────────────
    st.markdown('<div class="section-heading">SHAP Beeswarm</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Each dot = one test-day prediction. '
                'Position on x-axis = SHAP contribution to log-rainfall. '
                'Colour = feature value (red = high, blue = low). '
                'Vertical spread = density of predictions at that SHAP value.</div>',
                unsafe_allow_html=True)

    if "shap_beeswarm" in expl_figs:
        st.image(str(expl_figs["shap_beeswarm"]),
                 use_container_width=True)
    else:
        st.info("Beeswarm figure not found — run Phase 6.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Insight: Soil Moisture Gradient ───────────────────────────────────
    st.markdown('<div class="section-heading">Why this matters</div>',
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card" style="margin-top:12px">
        <div class="ic-eyebrow">SOIL MOISTURE GRADIENT</div>
        <div class="ic-headline">Engineering beats observation</div>
        <div class="ic-body">
        SOIL_MOISTURE_GRADIENT = SOIL_WET_SURF − SOIL_WET_ROOT was created
        to resolve collinearity (r=0.94) between the two raw soil columns.
        The gradient physically encodes moisture flux direction — recent
        infiltration predicts subsequent rainfall intensity.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PDP grid expander ─────────────────────────────────────────────────
    with st.expander("📈 Partial Dependence Plots — individual features"):
        st.markdown("""
        <div class="info-inline">
        <strong>How to read PDPs:</strong> The blue line shows the average model prediction as one
        feature varies while all others are held at their observed values. Grey lines are
        individual ICE curves — they show whether the relationship is the same for all days
        (clustered) or varies by context (spread). The blue band is the 5th–95th percentile
        of ICE curves.
        </div>
        """, unsafe_allow_html=True)

        if "pdp_grid" in expl_figs:
            st.image(str(expl_figs["pdp_grid"]), use_container_width=True)
        else:
            st.info("PDP grid figure not found.")

        st.markdown("""
        | Feature | Key PDP finding |
        |---|---|
        | **CLOUD** | Strong positive trend; steepens above ~0.5 (scaled), suggesting a threshold |
        | **SOIL_MOISTURE_GRADIENT** | Near-linear positive; saturates above ~0.15 |
        | **RAINFALL_lag1** | Steep initial slope, then plateau — autocorrelation saturates past ~5mm previous day |
        | **RAINFALL_roll_mean_7** | Broad positive — current wet-spell regime |
        """)

    # ── Local explanations expander ───────────────────────────────────────
    with st.expander("🔍 Day-level explanations — waterfall plots"):
        st.markdown("""
        <div class="info-inline">
        <strong>How to read waterfall plots:</strong> Starting from the base value (average prediction
        = 1.459 on the log scale, ≈ 3.3 mm/day), each bar shows one feature's contribution.
        Green bars push the prediction higher; red bars pull it lower.
        The final value back-transforms to mm/day via <code>expm1()</code>.
        </div>
        """, unsafe_allow_html=True)

        if "shap_local_waterfall" in expl_figs:
            st.image(str(expl_figs["shap_local_waterfall"]), use_container_width=True)
        else:
            st.info("Local waterfall figure not found.")

    # ── Dependence plots expander ──────────────────────────────────────────
    with st.expander("🔬 SHAP dependence plots"):
        st.markdown("""
        <div class="info-inline">
        Each dot = one wet test-day prediction. x-axis = feature value (scaled).
        y-axis = that feature's SHAP contribution. Points are coloured by lag-1 rainfall,
        showing how the relationship changes in wet vs dry contexts.
        The red LOWESS smoother shows the average trend.
        </div>
        """, unsafe_allow_html=True)

        if "shap_dependence_grid" in expl_figs:
            st.image(str(expl_figs["shap_dependence_grid"]), use_container_width=True)
        else:
            st.info("Dependence grid figure not found.")


# ---------------------------------------------------------------------------
# Interactive SHAP importance bar
# ---------------------------------------------------------------------------

def _render_shap_interactive_bar(shap_df: pd.DataFrame) -> None:
    top = shap_df.head(20).sort_values("SHAP_MeanAbs").copy()

    bar_colors = []
    group_labels = []
    for feat in top["Feature"]:
        grp, col = FEATURE_GROUPS.get(feat, ("Other", "#94A3B8"))
        bar_colors.append(col)
        group_labels.append(grp)

    top["group"]   = group_labels
    top["color"]   = bar_colors
    top["pct_str"] = (top["SHAP_Frac"] * 100).round(1).astype(str) + "%"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["SHAP_MeanAbs"],
        y=top["Feature"],
        orientation="h",
        marker_color=top["color"],
        marker_opacity=0.88,
        marker_line_width=0,
        text=top["pct_str"],
        textposition="outside",
        textfont=dict(size=11, family="system-ui"),
        customdata=np.stack([top["group"], top["SHAP_MeanAbs"],
                              (top["SHAP_Frac"]*100).round(1)], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Group: %{customdata[0]}<br>"
            "Mean |SHAP|: %{customdata[1]:.5f}<br>"
            "% of total: %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
    ))

    # Annotate top feature prominently
    fig.add_annotation(
        x=float(top["SHAP_MeanAbs"].max()),
        y=top["Feature"].iloc[-1],
        text=f" ← {float(top['SHAP_Frac'].iloc[-1]*100):.1f}% of all importance",
        showarrow=False, xanchor="left",
        font=dict(size=11, color="#2E7D32", family="system-ui"),
    )

    # Legend for feature groups
    unique_groups = top.drop_duplicates("group")[["group","color"]].values
    for grp, col in unique_groups:
        fig.add_trace(go.Bar(
            x=[None], y=[None], orientation="h",
            name=grp,
            marker_color=col, marker_opacity=0.88,
            showlegend=True,
        ))

    fig.update_layout(
        height=560,
        xaxis=dict(
            title="Mean |SHAP Value| — contribution to log(1+rainfall)",
            showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            tickfont=dict(size=11),
            range=[0, float(top["SHAP_MeanAbs"].max()) * 1.22],
        ),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=15, b=55, l=200, r=100),
        legend=dict(
            title=dict(text="Feature group", font=dict(size=11)),
            orientation="v", x=1.02, y=0.5,
            font=dict(size=10),
        ),
        font=dict(family="system-ui, -apple-system, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)