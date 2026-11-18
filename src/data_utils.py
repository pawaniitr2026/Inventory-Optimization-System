import numpy as np
import pandas as pd

def make_demo_data(days=540, seed=7):
    """Create reproducible daily sales with weekday seasonality, trend and occasional shocks."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    dow = dates.dayofweek.to_numpy()
    trend = np.linspace(45, 60, days)
    weekly = np.where(dow >= 5, 1.25, 0.92)
    annual = 1 + 0.12 * np.sin(np.arange(days) * 2 * np.pi / 365.25)
    noise = rng.normal(0, 5, days)
    sales = trend * weekly * annual + noise
    # Add a few demand shocks, useful for demonstrating the uncertainty agent.
    for start, length, multiplier in [(int(days*.38), 5, 1.9), (int(days*.72), 8, 1.55)]:
        sales[start:start+length] *= multiplier
    sales = np.maximum(0, np.round(sales)).astype(int)
    return pd.DataFrame({"date": dates, "sales": sales})

def prepare_daily_series(df):
    df = df.copy()
    lower = {str(c).strip().lower(): c for c in df.columns}
    date_col = next((lower[k] for k in ["date", "ds", "day"] if k in lower), None)
    sales_col = next((lower[k] for k in ["sales", "demand", "units_sold", "quantity", "y"] if k in lower), None)
    if date_col is None or sales_col is None:
        raise ValueError("CSV needs a date column (date/ds/day) and a sales column (sales/demand/units_sold/quantity/y).")
    df = df.rename(columns={date_col: "date", sales_col: "sales"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])
    # Aggregate multiple stores/items into total daily demand. Filter a single item/store first if desired.
    daily = df.groupby("date", as_index=False)["sales"].sum().sort_values("date")
    daily = daily.set_index("date").asfreq("D")
    daily["sales"] = daily["sales"].interpolate("time").ffill().bfill().clip(lower=0)
    return daily.reset_index()

def load_sales_data(path):
    return prepare_daily_series(pd.read_csv(path))
