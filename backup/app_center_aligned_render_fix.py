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


def render_centered_table(df: pd.DataFrame):
    """Render a responsive HTML table with every header and cell centered."""
    display_df = df.copy()

    # Format datetime columns for a cleaner UI.
    for column in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[column]):
            display_df[column] = display_df[column].dt.strftime("%Y-%m-%d")

    html_table = display_df.to_html(
        index=False,
        escape=True,
        border=0,
        classes="centered-data-table",
        na_rep="",
    )

    table_html = f"""
    <style>
    .centered-table-wrap {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 9px;
    }}
    table.centered-data-table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: auto;
        font-size: 14px;
    }}
    table.centered-data-table thead th {{
        text-align: center !important;
        vertical-align: middle !important;
        padding: 12px 10px;
        font-weight: 600;
        background: rgba(128, 128, 128, 0.10);
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        white-space: nowrap;
    }}
    table.centered-data-table tbody td {{
        text-align: center !important;
        vertical-align: middle !important;
        padding: 10px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.18);
    }}
    table.centered-data-table tbody tr:last-child td {{
        border-bottom: none;
    }}
    </style>
    <div class="centered-table-wrap">
        {html_table}
    </div>
    """

    st.markdown(textwrap.dedent(table_html).strip(), unsafe_allow_html=True)

st.set_page_config(page_title="DemandShield AI", page_icon="📦", layout="wide")
st.title("📦 DemandShield AI")
st.caption("Agentic demand forecasting • shock-aware uncertainty analysis • inventory planning")

with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload daily sales CSV (optional)", type=["csv"])
    demo_days = st.slider("Demo history (days)", 180, 900, 540, 30)
    horizon = st.slider("Forecast horizon (days)", 7, 56, 28, 7)
    lead_time = st.number_input("Supplier lead time (days)", min_value=1, max_value=60, value=7)
    service_level = st.slider("Target service level", 0.80, 0.99, 0.95, 0.01)
    shock_mode = st.selectbox("Demand shock scenario", ["Auto-detect", "No shock", "Promotion spike", "Supply disruption / demand surge"])
    st.markdown("---")
    st.caption("CSV expected columns: date, sales (optional: store, item, price, promo, temperature).")

try:
    if uploaded:
        raw = pd.read_csv(uploaded)
        daily = prepare_daily_series(raw)
        data_source = "Uploaded CSV"
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

st.info(f"Data source: **{data_source}** · {len(daily):,} daily observations · {daily['date'].min().date()} to {daily['date'].max().date()}")

with st.spinner("Training baseline models and producing a holdout evaluation..."):
    forecast_bundle = run_forecasts(daily, horizon=horizon)

metrics = evaluate_forecasts(forecast_bundle["actual"], forecast_bundle["predictions"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best baseline (by MAE)", metrics.iloc[0]["Model"])
c2.metric("Best holdout MAE", f"{metrics.iloc[0]['MAE']:.2f}")
c3.metric("Average daily demand", f"{daily['sales'].mean():.1f}")
c4.metric("Peak daily demand", f"{daily['sales'].max():.0f}")

st.subheader("Historical demand")
st.line_chart(daily.set_index("date")["sales"])

st.subheader("Classical model comparison")
render_centered_table(metrics)
st.caption("Evaluation uses a chronological holdout. Models are deliberately simple baselines; tune them for production use.")

st.subheader("Forecast and uncertainty")
pred_df = forecast_bundle["future"].copy()
st.line_chart(pred_df.set_index("date")[["forecast", "lower", "upper"]])
render_centered_table(pred_df)

st.subheader("Agent collaboration")
if st.button("Run Forecast → Uncertainty → Inventory Planner", type="primary", use_container_width=True):
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
    a, b, c = st.columns(3)
    a.metric("Shock status", result["uncertainty"]["shock_status"])
    b.metric("Suggested reorder point", f'{result["plan"]["reorder_point"]:.1f}')
    c.metric("Suggested safety stock", f'{result["plan"]["safety_stock"]:.1f}')
    st.markdown("#### Forecast agent")
    st.write(result["forecast_agent"])
    st.markdown("#### Uncertainty agent")
    st.write(result["uncertainty"]["summary"])
    st.markdown("#### Inventory planner")
    st.write(result["plan"]["recommendation"])
    st.markdown("#### Planning details")
    st.json(result["plan"])
    st.markdown("#### Workflow trace")
    st.code(" → ".join(result["trace"]))

st.download_button(
    "Download future forecast CSV",
    data=pred_df.to_csv(index=False).encode("utf-8"),
    file_name="demand_forecast.csv",
    mime="text/csv",
)
