import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def _seasonal_naive(train, horizon, season=7):
    vals = np.asarray(train, dtype=float)
    if len(vals) >= season:
        pattern = vals[-season:]
        return np.resize(pattern, horizon)
    return np.repeat(vals[-1] if len(vals) else 0.0, horizon)

def _moving_average(train, horizon, window=14):
    vals = np.asarray(train, dtype=float)
    level = float(np.mean(vals[-min(window, len(vals)):])) if len(vals) else 0.0
    return np.repeat(level, horizon)

def _exp_smoothing(train, horizon):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(
            np.asarray(train, dtype=float),
            trend="add",
            seasonal="add" if len(train) >= 21 else None,
            seasonal_periods=7 if len(train) >= 21 else None,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.maximum(0, np.asarray(model.forecast(horizon), dtype=float))
    except Exception:
        return _moving_average(train, horizon)

def _predict_all(train, horizon):
    return {
        "Seasonal Naive (7-day)": np.maximum(0, _seasonal_naive(train, horizon)),
        "Moving Average (14-day)": np.maximum(0, _moving_average(train, horizon)),
        "Exponential Smoothing": np.maximum(0, _exp_smoothing(train, horizon)),
    }

def run_forecasts(daily, horizon=28):
    y = daily["sales"].astype(float).to_numpy()
    dates = pd.to_datetime(daily["date"])
    h = min(int(horizon), max(7, len(y)//5))
    split = max(30, len(y) - h)
    train, actual = y[:split], y[split:]
    holdout_preds = _predict_all(train, len(actual))
    future_preds = _predict_all(y, horizon)
    # Select best baseline on holdout MAE for primary forecast.
    best_name = min(holdout_preds, key=lambda k: mean_absolute_error(actual, holdout_preds[k]))
    primary = future_preds[best_name]
    # Empirical residual interval using holdout errors; fallback to recent demand variability.
    residuals = actual - holdout_preds[best_name]
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(y[-min(28, len(y)):]))
    sigma = max(sigma, 1.0)
    z = 1.96
    future_dates = pd.date_range(dates.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({
        "date": future_dates,
        "forecast": np.maximum(0, primary[:horizon]),
        "lower": np.maximum(0, primary[:horizon] - z*sigma),
        "upper": np.maximum(0, primary[:horizon] + z*sigma),
        "selected_model": best_name,
    })
    return {
        "actual": actual,
        "predictions": holdout_preds,
        "future": future,
        "best_model": best_name,
        "holdout_actual_dates": dates.iloc[split:].reset_index(drop=True),
        "holdout_predictions": holdout_preds,
    }

def evaluate_forecasts(actual, predictions):
    rows = []
    actual = np.asarray(actual, dtype=float)
    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)[:len(actual)]
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        denom = np.maximum(np.abs(actual), 1.0)
        mape = float(np.mean(np.abs(actual - pred) / denom) * 100)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
