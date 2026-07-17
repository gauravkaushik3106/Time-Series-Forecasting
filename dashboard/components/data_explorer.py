"""
data_explorer.py — Redesigned Data Explorer
=============================================
Single-page layout, no tabs. Controls inline above charts.
Centrepiece: DOY x Year heatmap at full width 700px.
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

from dashboard.data_loader import load_full_features

MONSOON = {6, 7, 8, 9}
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def render() -> None:
    st.markdown('<div class="section-heading">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">26 years of daily meteorological records for Lucknow, '
                'India. Use the controls below to filter and explore.</div>', unsafe_allow_html=True)

    df = load_full_features()
    if df.empty:
        st.error("Feature dataset not found. Run the preprocessing pipeline first.")
        return

    # ── Inline controls ───────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    with ctrl1:
        date_range = st.date_input(
            "Date range",
            value=(df.index.min().date(), df.index.max().date()),
            min_value=df.index.min().date(),
            max_value=df.index.max().date(),
            label_visibility="collapsed",
        )
    with ctrl2:
        log_scale = st.checkbox("Log scale", value=False)
    with ctrl3:
        roll_win = st.selectbox("Rolling mean", [7, 14, 30, 90], index=2,
                                label_visibility="collapsed")
    with ctrl4:
        sec_var = st.selectbox(
            "Secondary variable",
            options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                     and c not in ("RAINFALL","SPLIT","YEAR","MONTH","DAY_OF_YEAR",
                                   "DAY_OF_WEEK","IS_MONSOON","MONSOON_FLAG",
                                   "RAIN_OCCURRENCE","SEASON_CODE","IS_WEEKEND")],
            index=0,
            label_visibility="collapsed",
        )

    # Apply date filter
    if len(date_range) == 2:
        start, end = date_range
        mask = (df.index.date >= start) & (df.index.date <= end)
        dv = df.loc[mask].copy()
    else:
        dv = df.copy()

    rain = dv["RAINFALL"]
    rain_plot = np.log1p(rain) if log_scale else rain
    y_label   = "log(1 + Rainfall) mm" if log_scale else "Rainfall (mm/day)"
    roll       = rain_plot.rolling(roll_win, center=True, min_periods=roll_win//2).mean()

    # ── Quick stats row ───────────────────────────────────────────────────
    q1, q2, q3, q4, q5 = st.columns(5)
    stats_data = [
        (q1, f"{rain.mean():.2f} mm",  "Mean daily rainfall"),
        (q2, f"{rain.max():.1f} mm",   "Max single-day"),
        (q3, f"{(rain<0.1).mean()*100:.1f}%", "Dry days"),
        (q4, f"{rain.groupby(dv.index.year).sum().mean():.0f} mm", "Annual mean"),
        (q5, f"{len(dv):,}",           "Days shown"),
    ]
    for col, val, label in stats_data:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="font-size:1.6rem">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    # ── CHART 1: Daily rainfall time-series ───────────────────────────────
    st.markdown('<div class="section-heading" style="margin-top:24px">Daily Rainfall Time Series</div>',
                unsafe_allow_html=True)

    fig1 = go.Figure()
    for yr in range(dv.index.year.min(), dv.index.year.max() + 2):
        fig1.add_vrect(x0=f"{yr}-06-01", x1=f"{yr}-09-30",
                       fillcolor="rgba(59,178,115,0.07)", layer="below", line_width=0)

    fig1.add_trace(go.Bar(
        x=dv.index, y=rain_plot.values,
        marker_color="rgba(46,134,171,0.45)", marker_line_width=0,
        name="Daily rainfall",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:.2f}<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=dv.index, y=roll.values,
        line=dict(color="#E84855", width=2.0),
        name=f"{roll_win}-day mean",
        hovertemplate=f"{roll_win}-day mean: %{{y:.2f}}<extra></extra>",
    ))
    fig1.update_layout(
        height=420, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=15, b=50, l=65, r=20),
        yaxis=dict(title=y_label, showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11)),
        xaxis=dict(showgrid=False, tickfont=dict(size=11),
                   rangeslider=dict(visible=True, thickness=0.05)),
        legend=dict(orientation="h", y=1.02, font=dict(size=11)),
        font=dict(family="system-ui, sans-serif"),
        bargap=0,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── CHART 2: DOY × Year heatmap (centrepiece) ─────────────────────────
    st.markdown('<div class="section-heading">26-Year Rainfall Calendar</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Each row = one year. Each cell = one day. '
                'Colour = log(1+rainfall). White dashed lines mark monsoon boundaries '
                '(Jun 1, Sep 30).</div>', unsafe_allow_html=True)

    years = sorted(df.index.year.unique())
    grid  = np.full((len(years), 365), np.nan)
    for i, yr in enumerate(years):
        yr_data = df["RAINFALL"][df.index.year == yr]
        for dt, val in yr_data.items():
            doy = dt.dayofyear
            if 1 <= doy <= 365:
                grid[i, doy - 1] = val

    fig2 = px.imshow(
        np.log1p(grid),
        x=list(range(1, 366)),
        y=years,
        color_continuous_scale="YlOrRd",
        labels={"x": "Day of Year", "y": "Year", "color": "log(1+mm)"},
        aspect="auto",
    )
    fig2.add_vline(x=152, line_dash="dot", line_color="white", line_width=1.5,
                   annotation_text="Jun 1", annotation_font_color="white",
                   annotation_font_size=10)
    fig2.add_vline(x=273, line_dash="dot", line_color="white", line_width=1.5,
                   annotation_text="Sep 30", annotation_font_color="white",
                   annotation_font_size=10)
    # Month tick marks
    mid_doys = [15,46,74,105,135,166,196,227,258,288,319,349]
    fig2.update_xaxes(
        tickvals=mid_doys,
        ticktext=MONTH_ABBR,
        tickfont=dict(size=11),
    )
    fig2.update_yaxes(tickfont=dict(size=10))
    fig2.update_coloraxes(
        colorbar=dict(
            title="log(1+mm)", tickfont=dict(size=10),
            tickvals=[0, np.log1p(1), np.log1p(5), np.log1p(20),
                      np.log1p(50), np.log1p(100)],
            ticktext=["0","1","5","20","50","100+"],
        )
    )
    fig2.update_layout(
        height=680, margin=dict(t=15, b=50, l=55, r=20),
        paper_bgcolor="white",
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── CHART 3+4: Monthly climatology + Annual totals ────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-heading" style="font-size:1.1rem">Monthly Climatology</div>',
                    unsafe_allow_html=True)
        monthly_mean = df["RAINFALL"].groupby(df.index.month).mean()
        p_rain       = df["RAINFALL"].groupby(df.index.month).apply(lambda x: (x >= 0.1).mean() * 100)
        bar_cols     = ["#3BB273" if m in MONSOON else "#2E86AB" for m in range(1,13)]

        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(
            x=MONTH_ABBR,
            y=[monthly_mean.get(m, 0) for m in range(1,13)],
            marker_color=bar_cols, marker_line_width=0, name="Mean rainfall",
            hovertemplate="%{x}: %{y:.2f} mm/day<extra></extra>",
        ), secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=MONTH_ABBR,
            y=[p_rain.get(m, 0) for m in range(1,13)],
            line=dict(color="#E84855", width=2, dash="dot"),
            mode="lines+markers", marker_size=6,
            name="P(rain) %",
            hovertemplate="%{x}: %{y:.1f}% rainy days<extra></extra>",
        ), secondary_y=True)
        fig3.update_yaxes(title_text="Mean rainfall (mm/day)", secondary_y=False,
                          tickfont=dict(size=10))
        fig3.update_yaxes(title_text="P(rain) %", secondary_y=True,
                          tickfont=dict(size=10), range=[0,105])
        fig3.update_layout(
            height=340, hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=15, b=50, l=60, r=50),
            legend=dict(orientation="h", y=1.08, font=dict(size=10)),
            font=dict(family="system-ui, sans-serif"),
            bargap=0.15,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-heading" style="font-size:1.1rem">Annual Totals</div>',
                    unsafe_allow_html=True)
        annual = df["RAINFALL"].groupby(df.index.year).sum().reset_index()
        annual.columns = ["Year", "Total"]
        grand_mean = float(annual["Total"].mean())
        bar_cols2  = ["#3BB273" if v >= grand_mean else "#F4A261" for v in annual["Total"]]

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=annual["Year"], y=annual["Total"],
            marker_color=bar_cols2, marker_line_width=0,
            name="Annual total",
            hovertemplate="<b>%{x}</b><br>Total: %{y:.0f} mm<extra></extra>",
        ))
        fig4.add_hline(
            y=grand_mean, line_dash="dash", line_color="#E84855", line_width=1.5,
            annotation_text=f"26-yr mean: {grand_mean:.0f} mm",
            annotation_font_size=10, annotation_font_color="#E84855",
        )
        fig4.update_layout(
            height=340,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=15, b=50, l=60, r=20),
            yaxis=dict(title="Annual total (mm)", tickfont=dict(size=10),
                       showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            xaxis=dict(tickfont=dict(size=10)),
            font=dict(family="system-ui, sans-serif"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── DEEP DIVE: Secondary variable explorer ────────────────────────────
    with st.expander(f"🔬 Explore: {sec_var} vs Rainfall"):
        col_h, col_s = st.columns(2)
        with col_h:
            fig_h = px.histogram(
                dv, x=sec_var, nbins=50,
                color_discrete_sequence=["#2E86AB"],
                title=f"Distribution of {sec_var}",
            )
            fig_h.update_layout(
                height=320, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40,b=40,l=60,r=20),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
                font=dict(family="system-ui, sans-serif"),
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with col_s:
            dv_plot = dv.copy()
            dv_plot["Season"] = dv_plot.index.month.map(
                lambda m: "Monsoon" if m in MONSOON else "Non-Monsoon"
            )
            fig_s = px.scatter(
                dv_plot, x=sec_var, y="RAINFALL",
                color="Season",
                color_discrete_map={"Monsoon":"#3BB273","Non-Monsoon":"#F4A261"},
                opacity=0.35,
                title=f"{sec_var} vs Rainfall",
                trendline="lowess",
            )
            fig_s.update_traces(marker_size=3)
            fig_s.update_layout(
                height=320, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40,b=40,l=60,r=20),
                font=dict(family="system-ui, sans-serif"),
            )
            st.plotly_chart(fig_s, use_container_width=True)