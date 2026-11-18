import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_sales_data, make_demo_data, prepare_daily_series
from forecasting import run_forecasts, evaluate_forecasts
from agents import run_agent_workflow

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Inventory Optimization System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# ENTERPRISE DESIGN SYSTEM (CUSTOM CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Import clean enterprise typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --primary-navy: #0F172A;
            --accent-blue: #1D4ED8;
            --accent-blue-hover: #1E40AF;
            --accent-teal: #0D9488;
            --surface-bg: #F8FAFC;
            --card-bg: #FFFFFF;
            --border-subtle: #E2E8F0;
            --text-main: #0F172A;
            --text-muted: #64748B;
            --badge-bg: #EFF6FF;
            --badge-text: #1D4ED8;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-main);
        }

        /* Main container spacing and soft neutral backdrop */
        .stApp {
            background-color: var(--surface-bg);
        }

        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1380px !important;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid var(--border-subtle);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem !important;
        }

        /* Hero Header Banner */
        .hero-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            padding: 1.75rem 2rem;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
            border: 1px solid #334155;
        }

        .hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            padding: 0;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .hero-subtitle {
            font-size: 0.92rem;
            color: #CBD5E1;
            margin-top: 0.35rem;
            margin-bottom: 0;
            font-weight: 400;
        }

        /* Status Bar Pill */
        .status-banner {
            background-color: #FFFFFF;
            border: 1px solid var(--border-subtle);
            border-left: 4px solid var(--accent-blue);
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
            margin-bottom: 1.5rem;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            font-size: 0.88rem;
            color: #334155;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }

        .status-badge {
            background-color: var(--badge-bg);
            color: var(--badge-text);
            font-weight: 600;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.8rem;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        div[data-testid="stMetric"]:hover {
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.07);
        }

        div[data-testid="stMetricLabel"] > div {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-muted) !important;
        }

        div[data-testid="stMetricValue"] > div {
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            color: var(--primary-navy) !important;
            font-variant-numeric: tabular-nums;
        }

        /* Section Headings */
        .section-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--primary-navy);
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
            letter-spacing: -0.01em;
        }

        .section-caption {
            font-size: 0.84rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }

        /* Agent Insight Cards */
        .agent-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 1.25rem;
            height: 100%;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }

        .agent-card-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--primary-navy);
            margin-bottom: 0.65rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #F1F5F9;
        }

        .agent-card-body {
            font-size: 0.9rem;
            line-height: 1.6;
            color: #334155;
        }

        /* Primary Action Buttons */
        div.stButton > button[kind="primary"] {
            background-color: var(--accent-blue);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 0.65rem 1.25rem;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
            box-shadow: 0 2px 6px rgba(29, 78, 216, 0.25);
            transition: all 0.2s ease;
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: var(--accent-blue-hover);
            box-shadow: 0 4px 12px rgba(29, 78, 216, 0.35);
        }

        /* Download Button */
        div.stDownloadButton > button {
            background-color: #FFFFFF;
            color: var(--primary-navy);
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.88rem;
            padding: 0.5rem 1.1rem;
            transition: all 0.15s ease;
        }

        div.stDownloadButton > button:hover {
            border-color: var(--accent-blue);
            color: var(--accent-blue);
            background-color: #F8FAFC;
        }

        /* Tabs styling */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            padding: 0.6rem 1.1rem !important;
        }

        /* Sidebar section labels */
        .sidebar-group-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-top: 0.75rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# TABLE RENDERING HELPER (CENTER-ALIGNED & CONSISTENTLY FORMATTED)
# -----------------------------------------------------------------------------
def render_centered_table(df: pd.DataFrame):
    """Render an enterprise-styled, center-aligned HTML table inside a Streamlit HTML component."""
    display_df = df.copy()

    # Format dates and numeric columns consistently for display without altering source DataFrame.
    for column in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[column]):
            display_df[column] = display_df[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(
                lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
            )
        elif pd.api.types.is_integer_dtype(display_df[column]):
            display_df[column] = display_df[column].map(
                lambda x: f"{x:,}" if pd.notnull(x) else ""
            )

    html_table = display_df.to_html(
        index=False,
        escape=True,
        border=0,
        classes="centered-data-table",
        na_rep="—",
    )

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        html, body {{
          margin: 0;
          padding: 0;
          width: 100%;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          color: #1E293B;
          background: transparent;
        }}
        .centered-table-wrap {{
          width: 100%;
          overflow-x: auto;
          background: #FFFFFF;
          border: 1px solid #E2E8F0;
          border-radius: 10px;
          box-sizing: border-box;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }}
        table.centered-data-table {{
          width: 100%;
          border-collapse: collapse;
          table-layout: auto;
          font-size: 13.5px;
          font-variant-numeric: tabular-nums;
        }}
        table.centered-data-table thead th {{
          text-align: center !important;
          vertical-align: middle !important;
          padding: 12px 14px;
          font-weight: 600;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: #475569;
          background: #F1F5F9;
          border-bottom: 1px solid #CBD5E1;
          white-space: nowrap;
          position: sticky;
          top: 0;
          z-index: 1;
        }}
        table.centered-data-table tbody td {{
          text-align: center !important;
          vertical-align: middle !important;
          padding: 10px 14px;
          color: #0F172A;
          border-bottom: 1px solid #F1F5F9;
          white-space: nowrap;
        }}
        table.centered-data-table tbody tr:nth-child(even) {{
          background-color: #FAFAF9;
        }}
        table.centered-data-table tbody tr:hover {{
          background-color: #EFF6FF;
        }}
        table.centered-data-table tbody tr:last-child td {{
          border-bottom: none;
        }}
      </style>
    </head>
    <body>
      <div class="centered-table-wrap">
        {html_table}
      </div>
    </body>
    </html>
    """

    component_height = min(560, max(125, 52 + (len(display_df) + 1) * 40))
    st.components.v1.html(
        table_html,
        height=component_height,
        scrolling=True,
    )


# -----------------------------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">📦 Inventory Optimization System</div>
        <p class="hero-subtitle">
            Agentic demand forecasting &nbsp;•&nbsp; shock-aware uncertainty analysis &nbsp;•&nbsp; inventory planning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.caption("Configure data inputs, forecast horizons, and inventory policy parameters.")

    st.markdown('<div class="sidebar-group-title">1. Data Ingestion</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload daily sales CSV (optional)",
        type=["csv"],
        help="Expected columns: date, sales (optional: store, item, price, promo, temperature).",
    )
    demo_days = st.slider(
        "Demo history (days)",
        min_value=180,
        max_value=900,
        value=540,
        step=30,
        disabled=uploaded is not None,
    )

    st.divider()
    st.markdown('<div class="sidebar-group-title">2. Forecasting & Policy</div>', unsafe_allow_html=True)
    horizon = st.slider("Forecast horizon (days)", 7, 56, 28, 7)
    lead_time = st.number_input("Supplier lead time (days)", min_value=1, max_value=60, value=7)
    service_level = st.slider("Target service level", 0.80, 0.99, 0.95, 0.01)

    st.divider()
    st.markdown('<div class="sidebar-group-title">3. Scenario Simulation</div>', unsafe_allow_html=True)
    shock_mode = st.selectbox(
        "Demand shock scenario",
        ["Auto-detect", "No shock", "Promotion spike", "Supply disruption / demand surge"],
    )

    st.divider()
    st.info(
        "**CSV Schema:** Requires `date` and `sales`. Optional columns: `store`, `item`, `price`, `promo`, `temperature`."
    )


# -----------------------------------------------------------------------------
# DATA LOADING & VALIDATION
# -----------------------------------------------------------------------------
try:
    if uploaded:
        raw = pd.read_csv(uploaded)
        daily = prepare_daily_series(raw)
        data_source = f"Uploaded CSV ({uploaded.name})"
    else:
        raw = make_demo_data(days=demo_days)
        daily = prepare_daily_series(raw)
        data_source = "Synthetic demonstration data"
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

if len(daily) < 60:
    st.warning("Use at least 60 daily observations for meaningful comparison.")
    st.stop()

# Status Strip
st.markdown(
    f"""
    <div class="status-banner">
        <div>
            <span class="status-badge">ACTIVE DATASET</span>
            &nbsp;&nbsp;<strong>{data_source}</strong>
        </div>
        <div>
            <strong>{len(daily):,}</strong> daily observations &nbsp;|&nbsp;
            Window: <strong>{daily['date'].min().date()}</strong> to <strong>{daily['date'].max().date()}</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# MODEL TRAINING & EXECUTIVE KPIs
# -----------------------------------------------------------------------------
with st.spinner("Training baseline models and producing a holdout evaluation..."):
    forecast_bundle = run_forecasts(daily, horizon=horizon)

metrics = evaluate_forecasts(forecast_bundle["actual"], forecast_bundle["predictions"])
pred_df = forecast_bundle["future"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Baseline (by MAE)", metrics.iloc[0]["Model"])
c2.metric("Best Holdout MAE", f"{metrics.iloc[0]['MAE']:.2f}")
c3.metric("Average Daily Demand", f"{daily['sales'].mean():,.1f}")
c4.metric("Peak Daily Demand", f"{daily['sales'].max():,.0f}")

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# STRUCTURED WORKSPACE TABS
# -----------------------------------------------------------------------------
tab_overview, tab_forecast, tab_agents = st.tabs(
    [
        "📊 Demand & Baseline Evaluation",
        "📈 Forecast & Uncertainty",
        "🤖 Agentic Inventory Planner",
    ]
)

# --- TAB 1: HISTORICAL DEMAND & MODEL BENCHMARKING ---
with tab_overview:
    with st.container(border=True):
        st.markdown('<div class="section-header">Historical Daily Demand</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Observed daily unit sales across the active historical window.</div>',
            unsafe_allow_html=True,
        )
        st.line_chart(daily.set_index("date")["sales"], color="#1D4ED8", height=340)

    with st.container(border=True):
        st.markdown('<div class="section-header">Classical Model Comparison</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Evaluation uses a chronological holdout. Models are deliberately simple baselines; tune them for production use.</div>',
            unsafe_allow_html=True,
        )
        render_centered_table(metrics)


# --- TAB 2: FORECAST & UNCERTAINTY INTERVALS ---
with tab_forecast:
    with st.container(border=True):
        header_col, export_col = st.columns([3, 1])
        with header_col:
            st.markdown('<div class="section-header">Future Demand Forecast & Confidence Bounds</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-caption">{horizon}-day forward projection displaying point forecast alongside lower and upper uncertainty bands.</div>',
                unsafe_allow_html=True,
            )
        with export_col:
            st.markdown("<div style='text-align: right; padding-top: 0.25rem;'>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Forecast CSV",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="demand_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.line_chart(
            pred_df.set_index("date")[["forecast", "lower", "upper"]],
            color=["#1D4ED8", "#94A3B8", "#0D9488"],
            height=350,
        )

    with st.container(border=True):
        st.markdown('<div class="section-header">Forecast Schedule Data</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Daily point forecasts and prediction intervals for the selected planning horizon.</div>',
            unsafe_allow_html=True,
        )
        render_centered_table(pred_df)


# --- TAB 3: MULTI-AGENT COLLABORATION & INVENTORY PLANNING ---
with tab_agents:
    with st.container(border=True):
        action_col, info_col = st.columns([2, 3])
        with action_col:
            run_clicked = st.button(
                "Run Forecast → Uncertainty → Inventory Planner",
                type="primary",
                use_container_width=True,
            )
        with info_col:
            st.markdown(
                f"""
                <div style="font-size: 0.86rem; color: #475569; padding-top: 0.45rem;">
                    Active Policy: <strong>{int(lead_time)}d Lead Time</strong> &nbsp;•&nbsp;
                    <strong>{float(service_level):.0%} Service Level</strong> &nbsp;•&nbsp;
                    Scenario: <strong>{shock_mode}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if run_clicked:
            with st.spinner("Agents are collaborating..."):
                result = run_agent_workflow(
                    daily=daily,
                    forecast=pred_df,
                    metrics=metrics,
                    lead_time=int(lead_time),
                    service_level=float(service_level),
                    shock_mode=shock_mode,
                )
            st.session_state["agent_result"] = result

    if "agent_result" in st.session_state:
        result = st.session_state["agent_result"]

        # Inventory Policy KPIs
        a, b, c = st.columns(3)
        a.metric("Shock Status", result["uncertainty"]["shock_status"])
        b.metric("Suggested Reorder Point", f'{result["plan"]["reorder_point"]:,.1f}')
        c.metric("Suggested Safety Stock", f'{result["plan"]["safety_stock"]:,.1f}')

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        # Three-Column Agent Synthesis Cards
        col_f, col_u, col_p = st.columns(3)
        with col_f:
            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-card-header">📈 Forecast Agent</div>
                    <div class="agent-card-body">{result["forecast_agent"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_u:
            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-card-header">⚡ Uncertainty Agent</div>
                    <div class="agent-card-body">{result["uncertainty"]["summary"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p:
            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-card-header">📦 Inventory Planner</div>
                    <div class="agent-card-body">{result["plan"]["recommendation"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        # Collapsible Technical Diagnostics
        with st.expander("🔍 View Technical Planning Details & Agent Workflow Trace", expanded=False):
            detail_col, trace_col = st.columns([3, 2])
            with detail_col:
                st.markdown("**Planning Details (JSON Payload)**")
                st.json(result["plan"])
            with trace_col:
                st.markdown("**Execution Workflow Trace**")
                st.code(" → ".join(result["trace"]), language="text")
    else:
        st.info(
            "Click **Run Forecast → Uncertainty → Inventory Planner** above to execute the multi-agent pipeline and generate safety stock and reorder point recommendations."
        )
