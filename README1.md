# 📦 Inventory Optimization System

**Agentic Demand Forecasting • Shock-Aware Uncertainty Analysis • Deterministic Replenishment Planning**

An end-to-end supply chain decision-support application built with **Streamlit**, **statsmodels**, **scikit-learn**, and **LangGraph**. The system ingests daily historical sales data (such as the included `Walmart_customized.csv` or generated synthetic demand series), benchmarks classical time-series models via chronological holdout validation, detects demand shocks and volatility spikes, and orchestrates a three-agent workflow to compute auditable safety stock and reorder point recommendations.

---

## 1. Prerequisites & Dependencies

### System Requirements
* **Python:** `3.9` or higher (recommended: `3.10` or `3.11`)
* **IDE:** **Visual Studio Code (VS Code)** with the official **Python** extension (`ms-python.python`)

### Installing Dependencies via `requirements.txt`
Your workspace already includes a `venv/` virtual environment and a `requirements.txt` file. Activate your virtual environment in the VS Code terminal and install all required packages:

```bash
# Windows (PowerShell / Command Prompt)
.\venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

*(Core packages include `streamlit`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`, and optional agent orchestration libraries `langgraph`, `langchain-openai`, and `langchain-core`.)*

### Environment Variables (`.env`)
All quantitative metrics, shock ratios, safety stock quantities, and reorder points are computed deterministically in Python/NumPy. To enable optional AI-generated natural language explanations (`temperature=0`), populate your root `.env` file:

```env
OPENAI_API_KEY="your-openai-api-key"
# Or use OpenRouter:
# OPENROUTER_API_KEY="your-openrouter-api-key"
# LLM_MODEL="openai/gpt-4o-mini"
```

---

## 2. Project Structure & Key Modules

Your `Inventory-Optimization-System` workspace is organized as follows (`app.py` dynamically adds `src/` to `sys.path`):

```text
Inventory-Optimization-System/
├── .vscode/                   # Optional VS Code launch & debug configuration
│   └── launch.json            # One-click F5 Streamlit debugging config
├── backup/                    # Archived versions and backup scripts
├── src/                       # Core analytical & multi-agent backend package
│   ├── __init__.py            # Package initializer
│   ├── agents.py              # LangGraph 3-agent orchestration & inventory policy math
│   ├── data_utils.py          # Data ingestion, schema mapping, & synthetic data generator
│   └── forecasting.py         # Chronological holdout forecasting & 95% confidence bounds
├── venv/                      # Local Python virtual environment
├── .env                       # API keys & environment configuration
├── .gitignore                 # Git ignore rules (excludes venv/, __pycache__/, .env)
├── app.py                     # Main Streamlit UI & application entry point
├── README.md                  # Project documentation & VS Code setup guide
├── requirements.txt           # Project Python dependencies
└── Walmart_customized.csv     # Sample cleaned time-series sales dataset
```

### Module Descriptions

1. **`app.py` (Presentation & Control Layer)**
   * Configures the Streamlit enterprise theme, custom CSS cards, and sidebar configuration controls (CSV upload, demo history days, forecast horizon, supplier lead time, target service level, and shock scenario override).
   * Organizes outputs into three interactive tabs: **Demand & Baseline Evaluation**, **Forecast & Uncertainty**, and **Agentic Inventory Planner**.
   * Renders center-aligned, formatted HTML tables (`render_centered_table`) and provides one-click CSV export (`demand_forecast.csv`).

2. **`src/data_utils.py` (Data Engineering & Normalization Layer)**
   * `make_demo_data(days=540, seed=7)`: Generates a reproducible daily sales dataset featuring linear growth, weekday/weekend seasonality ($1.25\times$ weekend lift), annual sinusoidal seasonality, Gaussian noise, and two injected demand shock events.
   * `prepare_daily_series(df)`: Detects common column aliases (`date`/`ds`/`day` and `sales`/`demand`/`units_sold`/`quantity`/`y`), aggregates multi-store/item volumes by date, enforces a continuous daily calendar (`asfreq("D")`), and imputes missing gaps via time-based linear interpolation.
   * `load_sales_data(path)`: Convenience wrapper to read a CSV from disk and pass it through `prepare_daily_series()`.

3. **`src/forecasting.py` (Statistical Forecasting & Benchmarking Layer)**
   * `_predict_all(train, horizon)`: Trains three baseline forecasters—**Seasonal Naive (7-day)**, **Moving Average (14-day)**, and **Additive Holt-Winters Exponential Smoothing**.
   * `run_forecasts(daily, horizon=28)`: Splits historical observations chronologically (reserving up to 20% or `horizon` days for validation), selects the winning baseline by lowest holdout MAE, and constructs empirical 95% prediction intervals ($\hat{y}_t \pm 1.96 \cdot \sigma_{\text{res}}$).
   * `evaluate_forecasts(actual, predictions)`: Computes and ranks **MAE**, **RMSE**, and **MAPE (%)** across all evaluated baselines.

4. **`src/agents.py` (Multi-Agent Reasoning & Inventory Optimization Layer)**
   * `forecast_agent(state)`: Audits the selected baseline model and holdout error metrics, generating a cautious summary warning against uncritical extrapolation during regime shifts.
   * `uncertainty_agent(state)`: Compares the trailing 28-day mean and Coefficient of Variation ($\text{CV}$) against the prior 62-day baseline window (days $-90$ to $-28$) to classify shock status (`Auto-detect` or manual scenario override).
   * `inventory_planner_agent(state)`: Maps target service levels ($80\%$–$99\%$) to normal $z$-scores and calculates **Safety Stock** (applying a $1.25\times$ buffer multiplier when a shock or surge is active), **Reorder Point (ROP)**, and **Order-Up-To Level**.
   * `run_agent_workflow(...)`: Compiles and executes the directed acyclic graph (`START → forecast_agent → uncertainty_agent → inventory_planner → END`) via LangGraph or sequential fallback.

5. **`Walmart_customized.csv` (Sample Dataset)**
   * Ready-to-upload historical retail sales dataset formatted with standardized `date` and `sales` columns for testing real-world CSV ingestion in the Streamlit sidebar.

---

## 3. End-to-End Code Flow

```text
[Sidebar Inputs: Walmart_customized.csv / Parameters]
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 1. Data Ingestion (src/data_utils.py)            │
│    • Map date/sales aliases                      │
│    • Aggregate & resample to daily frequency     │
│    • Validate >= 60 daily observations           │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 2. Holdout Forecasting (src/forecasting.py)      │
│    • Chronological train/test split              │
│    • Benchmark 3 baseline models                 │
│    • Select lowest-MAE model & 95% intervals     │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 3. Multi-Agent Pipeline (src/agents.py)          │
│    • Forecast Agent (Model audit)                │
│    • Uncertainty Agent (Shock/CV detection)      │
│    • Inventory Planner (SS & ROP math)           │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ 4. Streamlit UI (app.py)                         │
│    • KPI Scorecards & Line Charts                │
│    • Agent Cards & JSON Diagnostics              │
│    • Downloadable Forecast CSV                   │
└──────────────────────────────────────────────────┘
```

1. **Initialization & Ingestion:** On page load (or parameter change), `app.py` checks whether a CSV (such as `Walmart_customized.csv`) has been uploaded in the sidebar. If present, it parses and normalizes the file via `prepare_daily_series()`; otherwise, it generates synthetic data via `make_demo_data()`. Series with fewer than 60 daily records are halted with a user warning.
2. **Baseline Training & KPI Display:** `run_forecasts()` and `evaluate_forecasts()` execute automatically, populating the top four KPI cards (*Best Baseline*, *Best Holdout MAE*, *Average Daily Demand*, *Peak Daily Demand*) as well as the charts and tables in **Tab 1** and **Tab 2**.
3. **On-Demand Agent Execution:** When the user clicks **Run Forecast → Uncertainty → Inventory Planner** in **Tab 3**, `run_agent_workflow()` passes the historical series, forecast DataFrame, accuracy metrics, and policy parameters through the three agents, storing the output dictionary in `st.session_state["agent_result"]`.
4. **Export & Diagnostics:** Planners can inspect the deterministic JSON payload and execution trace inside the expandable diagnostics drawer or export the future forecast schedule via **Download Forecast CSV**.

---

## 4. Running & Debugging in Visual Studio Code

1. **Open the Workspace:**
   * Launch VS Code and open the `Inventory-Optimization-System` folder (**File → Open Folder...**).

2. **Select the Existing `venv` Interpreter:**
   * Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and run **`Python: Select Interpreter`**.
   * Choose the local virtual environment interpreter: `./venv/Scripts/python.exe` (Windows) or `./venv/bin/python` (macOS/Linux).

3. **Run via Integrated Terminal:**
   * Open the VS Code terminal (``Ctrl+` ``) and run:
     ```bash
     streamlit run app.py
     ```
   * Open `http://localhost:8501` in your browser and optionally upload `Walmart_customized.csv` in the sidebar to test with real retail data.

4. **One-Click Breakpoint Debugging (`F5`):**
   * Create `.vscode/launch.json` in `Inventory-Optimization-System/` to debug `app.py` and `src/*.py` directly with breakpoints:
     ```json
     {
       "version": "0.2.0",
       "configurations": [
         {
           "name": "Run Inventory Optimization (Streamlit)",
           "type": "debugpy",
           "request": "launch",
           "module": "streamlit",
           "args": [
             "run",
             "${workspaceFolder}/app.py",
             "--server.port",
             "8501"
           ],
           "envFile": "${workspaceFolder}/.env",
           "console": "integratedTerminal",
           "justMyCode": true
         }
       ]
     }
     ```
   * Press **`F5`** to launch the Streamlit server with `.env` loaded automatically.

---

## 5. Custom CSV Input Format

When uploading `Walmart_customized.csv` or your own dataset via the sidebar file uploader, the CSV must include one recognized **date** column and one recognized **volume** column (case-insensitive):

| Role | Accepted Column Names | Example Values |
| :--- | :--- | :--- |
| **Date** (Required) | `date`, `ds`, `day` | `2010-02-05`, `2024-01-15` |
| **Demand** (Required) | `sales`, `demand`, `units_sold`, `quantity`, `y` | `49750740.50`, `142.5` |
| **Metadata** (Optional) | `store`, `item`, `price`, `promo`, `temperature` | Automatically aggregated across rows per date |