import streamlit as st
from pathlib import Path

from ziko_st_toc import table_of_contents

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

with st.sidebar:
    table_of_contents()

st.set_page_config(page_title="Conclusions", layout="wide")

st.image(
    BASE_DIR / "src" / "streamlit" / "car.png",
    width="stretch",
)
# --- Native Streamlit Cards Grid ---
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 📊 Model Performance & Predictability")
        st.markdown(
            """
            * **High Predictability:** WLTP emissions are strongly predictable from vehicle specifications, but performance depends on powertrain regime and configuration novelty.
            * **Primary Model:** **ExtraTrees** is the recommended primary model. The **MLP** is a documented complementary benchmark, not the accuracy leader.
            """
        )

    with st.container(border=True):
        st.markdown("### ⚡ Powertrain & PHEV Insights")
        st.markdown(
            """
            * **Generalization:** Mono and Hybrid models generalised well.
            * **PHEV Weakness:** Unseen PHEV configurations remained the principal weakness.
            * **Data Needs:** PHEV errors indicate that battery capacity, drivetrain architecture, vehicle version, aerodynamics, and other homologation-relevant variables may be needed.
            """
        )

with col2:
    with st.container(border=True):
        st.markdown("### ⚙️ Feature Set Selection")
        st.markdown(
            """
            * **Extended Set:** Use the extended specification set when electric range and IT indicators are available.
            * **Design-Stage Set:** Retain the strict design-stage set for earlier estimates.
            """
        )

    with st.container(border=True):
        st.markdown("### 🛡️ Reporting & Safeguards")
        st.markdown(
            """
            * **Granular Metrics:** Always report results by configuration novelty, fuel type, fuel mode, and error tail rather than relying only on aggregate metrics.
            * **Recommended Safeguards:** Include prediction intervals, monitoring by powertrain and novelty status, periodic retraining with new model years, explicit feature-availability documentation, and a stricter future benchmark based on unseen vehicle families or manufacturers.
            """
        )
st.markdown("---")

# --- Scientific and Business Implications Section ---
st.header("Scientific and Business Implications")
st.caption("Expert insights in meteorology & climate research")

# Inject Custom CSS Stylesheet
st.markdown(
    """
    <style>
    .expert-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-top: 10px;
    }
    .insight-card {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .insight-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .insight-body {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .author-badge {
        text-align: center;
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 12px;
        margin-top: 20px;
        font-size: 1rem;
        color: #1e3a8a;
    }
    .author-link {
        color: #2563eb;
        font-weight: 700;
        text-decoration: none;
    }
    .author-link:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render HTML Content (Unindented to prevent Streamlit code block rendering)
html_content = """
<div class="expert-container">
<h3 style="margin-top:0; color:#0f172a;">🌤️ Expert Insights</h3>

<div class="insight-card" style="border-left-color: #ef4444;">
<div class="insight-title">⏳ Data Collection Delay</div>
<div class="insight-body">Data collection is a major pain point—data is recorded very late, which forces research to usually rely on projections.</div>
</div>

<div class="insight-card" style="border-left-color: #10b981;">
<div class="insight-title">💎 Real Data Value</div>
<div class="insight-body">A project grounded in recent real-world data rather than theoretical projections adds high scientific value.</div>
</div>

<div class="insight-card" style="border-left-color: #3b82f6;">
<div class="insight-title">🎯 Model Reliability</div>
<div class="insight-body">Machine learning models are significantly more reliable when built on direct observation rather than projected estimates.</div>
</div>

<div class="insight-card" style="border-left-color: #f59e0b;">
<div class="insight-title">⚠️ Potential Data Bias</div>
<div class="insight-body">Bias in datasets often stems from self-reporting by manufacturers and companies.</div>
</div>

<div class="insight-card" style="border-left-color: #8b5cf6;">
<div class="insight-title">🏙️ Collection Context</div>
<div class="insight-body">Academic institutions gather real-world data on the streets rather than solely in controlled test environments.</div>
</div>

<div class="author-badge">
    🎓 <b>Dr. Andrea Orfanoz</b>, Meteorology Specialist & Researcher at 
    <a href="https://mpimet.mpg.de/mitarbeiterinnen/andrea-orfanoz" target="_blank" class="author-link">
        Max Planck Institute for Meteorology, Hamburg, Germany.
    </a>
</div>
</div>
"""

st.markdown(html_content, unsafe_allow_html=True)