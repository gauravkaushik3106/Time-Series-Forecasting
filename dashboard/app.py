"""
app.py — Redesigned Dashboard Entry Point
==========================================
Run: streamlit run dashboard/app.py  (from project root)

Architecture rule: presentation layer only.
Zero model inference, zero training, zero SHAP recomputation.
All data loaded from outputs/ via @st.cache_data loaders.
"""

from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

_DASHBOARD_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT  = _DASHBOARD_DIR.parent
for p in [str(_PROJECT_ROOT), str(_DASHBOARD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Page config — must be FIRST Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="Lucknow Rainfall Framework",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Lucknow Rainfall Forecasting Framework**\n\n"
            "Hybrid ML · Deep Learning · SHAP · MC Dropout\n"
            "Daily data: Lucknow, India, 2000–2025"
        ),
    },
)


# ── CSS injection ───────────────────────────────────────────────────────────
def _inject_css() -> None:
    css_path = _DASHBOARD_DIR / "assets" / "custom.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>",
                    unsafe_allow_html=True)
    else:
        st.error("Error: custom.css not found. Please ensure the assets folder exists.")


_inject_css()


# ── Sidebar navigation ──────────────────────────────────────────────────────
PAGES = {
    "🏠  Home":               "home",
    "📊  Data Explorer":      "data_explorer",
    "🔭  Forecast Panel":     "forecast_panel",
    "🏆  Model Comparison":   "model_comparison",
    "💡  Explainability":     "explainability",
    "📉  Uncertainty":        "uncertainty",
    "🔬  Research Insights":  "research_insights",
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 12px 0">
        <span style="font-size:3.5rem">☁️</span><br>
        <span style="font-size:1.1rem;font-weight:800;color:white;
                     letter-spacing:0.5px;line-height:1.3;display:block;
                     margin-top:12px">
            Lucknow Rainfall<br>Framework
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12);margin:16px 0'>",
                unsafe_allow_html=True)

    selected = st.radio(
        "Navigate",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12);margin:16px 0'>",
                unsafe_allow_html=True)
    
    # Ultra-minimalist recruiter stats
    st.markdown("""
    <div style="font-size:0.85rem;color:rgba(255,255,255,0.6);
                line-height:1.8;padding-bottom:16px;font-weight:500;">
        <strong>Dataset:</strong> 26 Years (2000–2025)<br>
        <strong>Models:</strong> XGBoost, LSTM, GRU, SARIMAX<br>
        <strong>Focus:</strong> Hydrology & Uncertainty
    </div>
    """, unsafe_allow_html=True)


# ── Page routing ─────────────────────────────────────────────────────────────
page_key = PAGES[selected]

if page_key == "home":
    from dashboard.components.home import render
    render()

elif page_key == "data_explorer":
    from dashboard.components.data_explorer import render
    render()

elif page_key == "forecast_panel":
    from dashboard.components.forecast_panel import render
    render()

elif page_key == "model_comparison":
    from dashboard.components.model_comparison import render
    render()

elif page_key == "explainability":
    from dashboard.components.explainability_panel import render
    render()

elif page_key == "uncertainty":
    from dashboard.components.uncertainty_panel import render
    render()

elif page_key == "research_insights":
    from dashboard.components.research_insights import render
    render()