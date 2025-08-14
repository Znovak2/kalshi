"""
Predict the next day and next day+1 (two-step forecast) and append both predictions to subway_predictions.csv.

This file mirrors `subway.py` but configures the model for a 2-step horizon and writes two CSV rows
(one per forecasted date) so the CSV workflow remains compatible.

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


def train_and_forecast_to_sunday(series) -> list[tuple[pd.Timestamp, float]]:
    """Train on full history and forecast up to the upcoming Sunday.

    Returns list of (date, prediction) from next day up to target Sunday (inclusive).
    """
    from darts.dataprocessing.transformers import Scaler
    from darts.models import LightGBMModel
    from darts import TimeSeries

    # Determine target Sunday relative to last observation.
    last_obs = series.end_time()
    # weekday(): Monday=0 ... Sunday=6
    wd = last_obs.weekday()
    # days until upcoming Sunday: if it's already Sunday, target next Sunday (+7)
    days_until_sunday = (6 - wd) if wd != 6 else 7
    horizon = days_until_sunday

    # Build covariates for training span
    future_covariates = build_future_covariates(series)

    # Scale
    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)

    # Configure model for multi-step output
    lags = [-1, -2, -3, -4, -5, -6, -7, -14, -21]
    model = LightGBMModel(
        lags=lags,
        lags_future_covariates=[0],
        output_chunk_length=horizon,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=5,
        feature_fraction=0.8,
        random_state=42,
        verbosity=-1,
    )

    model.fit(series_scaled, future_covariates=future_covariates)

    # Predict 'horizon' steps ahead and invert scaling
    future_covariates_ext = extend_future_covariates(series, days=horizon)
    preds_scaled = model.predict(n=horizon, future_covariates=future_covariates_ext)
    if isinstance(preds_scaled, list):
        preds_scaled = preds_scaled[0]
    preds_any = scaler.inverse_transform(preds_scaled)
    if isinstance(preds_any, list):
        preds_any = preds_any[0]
    preds_ts = cast(TimeSeries, preds_any)

    # Extract predicted values and corresponding dates
    arr = preds_ts.values()
    # arr shape: (timesteps, components). We take the last 'horizon' rows (likely equals arr)
    if arr.ndim == 1:
        flat = [float(x) for x in arr[-horizon:]]
    else:
        flat = [float(row[0]) for row in arr[-horizon:]]

    dates = [last_obs + timedelta(days=i + 1) for i in range(horizon)]
    return list(zip(dates, flat))


def save_predictions(log_path: str, preds: list[tuple[pd.Timestamp, float]]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    now_iso = datetime.now().isoformat()
    import csv

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["target_date", "predicted", "predicted_at"])
        for dt, val in preds:
            writer.writerow([dt.strftime("%Y-%m-%d"), val, now_iso])


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

    print("Training and forecasting up to the upcoming Sunday…")
    preds = train_and_forecast_to_sunday(series)
    for dt, pv in preds:
        print(f"Predicted {dt.date()}: {int(round(pv))}")

    # Append predictions to centralized CSV under transportation/subway
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    log_file = os.path.join(root_dir, "subway_predictions.csv")
    save_predictions(log_file, preds)
    print(f"Logged predictions to: {log_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
