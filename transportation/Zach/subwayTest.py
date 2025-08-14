"""
Test script: train up to latest day-1 and forecast the latest day to compare with actual.

This mirrors subway.py but holds out the last day as the evaluation target so you can
validate your forecasting pipeline against the most recent known data point.

Requirements:
    pip install u8darts lightgbm holidays pandas requests
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Tuple, Any, cast

import pandas as pd
import requests
import warnings

# Suppress benign warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"pkg_resources is deprecated as an API",
)


def ensure_deps() -> None:
    try:
        import darts  # noqa: F401
        from darts import TimeSeries  # noqa: F401
        from darts.models import LightGBMModel  # noqa: F401
    except Exception:
        print(
            "Missing dependencies. Please install them first:\n"
            "  pip install u8darts lightgbm holidays pandas requests",
            file=sys.stderr,
        )
        raise


def get_ny_data(timeout: int = 30) -> pd.DataFrame:
    """Fetch raw data from NY open data API and return as DataFrame.

    Uses SoQL to retrieve more history and server-side filtering/sorting.
    """
    url = "https://data.ny.gov/resource/sayj-mze2.json"
    params = {
        "$select": "date,mode,count",
        "$where": "mode like 'Subway'",
        "$order": "date",
        "$limit": "50000",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("NY API returned no data")
    df = df.copy()
    if "mode" in df.columns:
        df = df[df["mode"].astype(str).str.contains("Subway", regex=False, na=False, case=False)]
    if "date" not in df.columns or "count" not in df.columns:
        missing = {c for c in ["date", "count"] if c not in df.columns}
        raise KeyError(f"Missing required columns in API response: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["date", "count"]).sort_values("date")
    return df[["date", "count"]]


def build_series(df: pd.DataFrame):
    from darts import TimeSeries
    import numpy as np

    daily = df.set_index("date")["count"].asfreq("D").interpolate("time").ffill().bfill()
    series = TimeSeries.from_series(daily, fill_missing_dates=True, freq="D").astype("float32")
    return series


def build_future_covariates(series) -> Any:
    from darts import TimeSeries
    from darts.utils.timeseries_generation import datetime_attribute_timeseries
    import holidays as holidays
    import numpy as np
    import pandas as pd

    idx = pd.date_range(start=series.start_time(), end=series.end_time(), freq="D")
    dow = datetime_attribute_timeseries(idx, attribute="weekday", one_hot=True)
    moy = datetime_attribute_timeseries(idx, attribute="month", one_hot=True)
    weekend_flag = pd.Series((idx.weekday >= 5).astype(np.int8), index=idx)
    is_weekend = TimeSeries.from_series(weekend_flag, freq="D")
    us_holidays = holidays.country_holidays('US', subdiv='NY')
    holiday_flag = pd.Series(idx.map(lambda d: 1 if d in us_holidays else 0), index=idx)
    holidays_ts = TimeSeries.from_series(holiday_flag, freq="D")
    return dow.stack(moy).stack(is_weekend).stack(holidays_ts)


def extend_future_covariates(series, days: int = 1) -> Any:
    from darts import TimeSeries
    from darts.utils.timeseries_generation import datetime_attribute_timeseries
    import holidays as holidays
    import numpy as np
    import pandas as pd

    last_date = series.end_time()
    end = last_date + timedelta(days=days)
    idx = pd.date_range(start=series.start_time(), end=end, freq="D")
    dow = datetime_attribute_timeseries(idx, attribute="weekday", one_hot=True)
    moy = datetime_attribute_timeseries(idx, attribute="month", one_hot=True)
    weekend_flag = pd.Series((idx.weekday >= 5).astype(np.int8), index=idx)
    is_weekend = TimeSeries.from_series(weekend_flag, freq="D")
    us_holidays = holidays.country_holidays('US', subdiv='NY')
    holiday_flag = pd.Series(idx.map(lambda d: 1 if d in us_holidays else 0), index=idx)
    holidays_ts = TimeSeries.from_series(holiday_flag, freq="D")
    return dow.stack(moy).stack(is_weekend).stack(holidays_ts)


def train_and_forecast_holdout(series) -> Tuple[pd.Timestamp, float, float]:
    """Train on data up to latest day-1 and forecast the latest day.

    Returns (forecast_date, predicted_value, actual_value).
    """
    from darts.dataprocessing.transformers import Scaler
    from darts.models import LightGBMModel
    from darts import TimeSeries

    # Define train (up to last-1) and target (last)
    forecast_date = series.end_time()
    train_end = forecast_date - timedelta(days=1)
    train_series = series.slice(end=train_end)

    # Covariates for train range and extended 1 day to forecast_date
    future_covariates = build_future_covariates(train_series)

    # Scale target
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)

    # Model
    lags = [-1, -2, -3, -4, -5, -6, -7, -14, -21]
    model = LightGBMModel(
        lags=lags,
        lags_future_covariates=[0],
        output_chunk_length=1,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=5,
        feature_fraction=0.8,
        random_state=42,
        verbosity=-1,
    )

    model.fit(train_scaled, future_covariates=future_covariates)
    future_covariates_ext = extend_future_covariates(train_series, days=1)
    pred_scaled = model.predict(n=1, future_covariates=future_covariates_ext)
    if isinstance(pred_scaled, list):
        pred_scaled = pred_scaled[0]
    pred_any = scaler.inverse_transform(pred_scaled)
    if isinstance(pred_any, list):
        pred_any = pred_any[0]
    pred_ts = cast(TimeSeries, pred_any)
    pred_value = float(pred_ts.last_value())

    # Actual value at forecast_date from the full series
    actual_value = float(series.slice(forecast_date, forecast_date).last_value())
    return forecast_date, pred_value, actual_value


def main() -> int:
    ensure_deps()
    print("Fetching NY subway data…")
    raw_df = get_ny_data()
    ny_df = clean_data(raw_df)
    print(f"Rows after cleaning: {len(ny_df):,}")

    series = build_series(ny_df)
    start_date = pd.Timestamp(series.start_time()).date()
    end_date = pd.Timestamp(series.end_time()).date()
    print(f"Series span: {start_date} → {end_date}  (n={len(series)})")

    print("Training on up-to-last-day-1 and forecasting the latest day…")
    forecast_date, pred, actual = train_and_forecast_holdout(series)
    abs_err = abs(pred - actual)
    mape = (abs_err / actual * 100.0) if actual != 0 else float('nan')
    print(
        f"Forecast date: {forecast_date.date()} | Predicted: {int(round(pred))} | "
        f"Actual: {int(round(actual))} | Abs Err: {int(round(abs_err))} | MAPE: {mape:.2f}%"
    )

    # Optional CSV output for tests (off by default)
    SAVE_TEST_CSV = False
    if SAVE_TEST_CSV:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(script_dir, "subway_predictions_test.csv")
        file_exists = os.path.isfile(out)
        import csv
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["target_date", "predicted", "actual", "abs_error", "mape", "evaluated_at"])
            w.writerow([
                forecast_date.strftime("%Y-%m-%d"), pred, actual, abs_err, mape, datetime.now().isoformat()
            ])
        print(f"Saved test row to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
