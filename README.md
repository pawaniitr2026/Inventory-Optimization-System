# DemandShield AI — Agentic Demand Forecasting Under Demand Shocks

A Visual Studio Code starter project combining classical forecasting, statistical uncertainty monitoring, and a LangGraph multi-agent inventory planning workflow.

## Features

- Daily demand upload through Streamlit (CSV).
- Synthetic demo dataset with seasonality, trend, noise, and injected demand spikes.
- Classical baseline comparison: 7-day seasonal naive, 14-day moving average, and exponential smoothing.
- Chronological holdout metrics: MAE, RMSE, MAPE.
- Empirical residual forecast interval.
- LangGraph sequence: **Forecast Agent → Uncertainty Agent → Inventory Planner Agent**.
- Optional LangChain chat model for explanatory text via OpenRouter or OpenAI.
- Deterministic inventory calculations remain independent of the LLM narrative.
- Downloadable future forecast CSV.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- Visual Studio Code
- Internet only for package installation and optional LLM API calls

## 2. Setup in VS Code (Windows PowerShell)

```powershell
cd demand_shock_agentic_forecasting
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and optionally add your `OPENROUTER_API_KEY`. The app works without a key using deterministic fallback text. Do not commit `.env` or expose API keys.

## 3. Run

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 4. Input CSV format

Minimum columns:

```csv
date,sales
2025-01-01,120
2025-01-02,134
2025-01-03,128
```

Accepted date column aliases: `date`, `ds`, `day`. Accepted sales aliases: `sales`, `demand`, `units_sold`, `quantity`, `y`.

If the CSV contains multiple stores or items, this starter aggregates them to total daily demand. For item/store-level forecasts, filter to one series or extend the app to select a series before aggregation.

## 5. Data sources

- M5 Forecasting: https://www.kaggle.com/competitions/m5-forecasting-accuracy
- Walmart Sales: https://www.kaggle.com/datasets/yasserh/walmart-dataset
- NOAA Climate Data Online: https://www.ncei.noaa.gov/

These datasets have different schemas and granularity. Normalize them to daily `date` and `sales` first. The included app accepts a normalized CSV; it does not automatically download Kaggle/NOAA datasets.

### Notes for M5
M5 daily unit sales are wide-format (`d_1`, `d_2`, ...). Join the calendar table to map `d_n` to calendar dates, then melt the item/store series into a long table. Start with a single item/store to keep the first experiment manageable.

### Notes for Walmart
The common Walmart dataset is weekly, with columns such as `Date`, `Weekly_Sales`, `Store`, and `Dept`. Rename `Weekly_Sales` to `sales` and `Date` to `date`. The starter app resamples to daily and is best suited to daily data, so for a fair weekly Walmart evaluation, adapt the app to weekly frequency instead of interpolating weekly totals into daily values.

## 6. Agent design

1. **Forecast Agent** — selects the best baseline using holdout MAE and explains the selected model.
2. **Uncertainty Agent** — compares recent demand mean and variability against a historical reference window and labels possible shocks.
3. **Inventory Planner Agent** — estimates lead-time demand, safety stock, reorder point, and order-up-to level.

The workflow uses LangGraph state and sequential edges. LangChain's chat model is optional and only used for narrative generation. Forecast and inventory arithmetic are calculated in Python for repeatability.

## 7. Important limitations / next steps

This is a capstone starter, not a production inventory system.

- Shock detection is a simple heuristic, not a validated change-point detector.
- Prediction intervals use holdout residual standard deviation and are not calibrated conformal intervals.
- Inventory formulas assume approximately independent daily errors and a known lead time.
- No current stock, purchase orders, minimum order quantity, pack size, shelf-life, cost, promotions, weather or supplier constraints are included yet.
- MAPE is unstable when actual demand is zero or close to zero.
- Use rolling-origin backtesting and evaluate stockout rate, fill rate, holding cost, and waste before claiming business improvement.
- NOAA weather data needs location/date alignment and a feature-engineering step before it can be used as a forecast feature.

## 8. Suggested project extensions

- Add weather and promotion features with a feature table keyed by date/store/item.
- Add a shock-aware model that blends a baseline with recent level/trend adjustment.
- Use rolling-origin backtesting instead of one holdout split.
- Add an inventory simulator that compares reorder policies under demand scenarios.
- Add LangSmith tracing and structured Pydantic output schemas.
- Add an audit log to SQLite for forecasts, shock alerts, recommendations, and human overrides.
- Add tests for data validation, forecast metrics, shock thresholds, and reorder calculations.
