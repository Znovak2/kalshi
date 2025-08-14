"""
Hybrid + calibration predictor to upcoming Sunday.

This script combines:
  - Iterative one-step LightGBM (day-by-day), calibrated with a global bias offset.
  - Multi-step LightGBM (predict full horizon in one shot).

Hybrid rule (configurable via --switch):
  - Use calibrated iterative predictions for steps <= switch.
  - Use multi-step predictions for steps > switch.

Bias calibration strategy:
  - If `msae_detailed.csv` exists in the same folder, compute the global mean error
    for mode == 'iterative' and use its negative as the bias adjustment.
  - Else default to 0 unless overridden with --bias.

Outputs: Appends rows to `subway_predictions.csv` with columns
  [target_date, predicted, predicted_at]

Requirements:
  pip install u8darts lightgbm holidays pandas requests
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import List, Tuple, Any, cast, Union

import pandas as pd
import requests
import warnings

# Suppress noisy warnings
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


def compute_iterative_bias_from_file(folder: str) -> float:
    """If msae_detailed.csv exists, compute global iterative mean error and return -mean_error.

    Searches:
      - folder/msae_detailed.csv
      - parent/msae_detailed.csv
      - parent/phase2/msae_detailed.csv
      - parent/phase2/phase2_data/msae_detailed.csv

    Returns 0.0 if file missing or malformed.
    """
    here = os.path.abspath(folder)
    parent = os.path.dirname(here)
    candidates = [
        os.path.join(here, "msae_detailed.csv"),
        os.path.join(parent, "msae_detailed.csv"),
        os.path.join(parent, "phase2", "msae_detailed.csv"),
        os.path.join(parent, "phase2", "phase2_data", "msae_detailed.csv"),
    ]
    try:
        for path in candidates:
            if os.path.isfile(path):
                df = pd.read_csv(path)
                if {"mode", "predicted", "actual"}.issubset(df.columns):
                    it = df[df["mode"].str.lower() == "iterative"].copy()
                    if not it.empty:
                        mean_err = (it["predicted"] - it["actual"]).mean()
                        return float(-mean_err)
        return 0.0
    except Exception:
        return 0.0


def train_iterative_model(series):
    from darts.dataprocessing.transformers import Scaler
    from darts.models import LightGBMModel

    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)

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

    model.fit(series_scaled, future_covariates=build_future_covariates(series))
    return model, scaler
 
 
def compute_iterative_bias_by_dow_from_file(folder: str) -> dict[int, float]:
    """Compute negative mean error per weekday (0=Mon..6=Sun) for iterative mode."""
    here = os.path.abspath(folder)
    parent = os.path.dirname(here)
    candidates = [
        os.path.join(here, "msae_detailed.csv"),
        os.path.join(parent, "msae_detailed.csv"),
        os.path.join(parent, "phase2", "msae_detailed.csv"),
        os.path.join(parent, "phase2", "phase2_data", "msae_detailed.csv"),
    ]
    try:
        for path in candidates:
            if os.path.isfile(path):
                df = pd.read_csv(path)
                if {"mode", "predicted", "actual", "target_date"}.issubset(df.columns):
                    it = df[df["mode"].str.lower() == "iterative"].copy()
                    if not it.empty:
                        it["target_date"] = pd.to_datetime(it["target_date"])
                        it["weekday"] = it["target_date"].dt.weekday
                        # bias = -mean_error per weekday
                        grp = it.groupby("weekday")
                        bias = {wd: float(-(g["predicted"] - g["actual"]).mean()) for wd, g in grp}
                        # default 0 for missing days
                        return {d: bias.get(d, 0.0) for d in range(7)}
        # no file or no data, return zeros
        return {d: 0.0 for d in range(7)}
    except Exception:
        return {d: 0.0 for d in range(7)}

def compute_optimal_switch_from_file(folder: str, horizon: int) -> int:
    """Determine best hybrid switch step by minimizing MAE from msae_detailed.csv."""
    here = os.path.abspath(folder)
    parent = os.path.dirname(here)
    candidates = [
        os.path.join(here, "msae_detailed.csv"),
        os.path.join(parent, "msae_detailed.csv"),
        os.path.join(parent, "phase2", "msae_detailed.csv"),
        os.path.join(parent, "phase2", "phase2_data", "msae_detailed.csv"),
    ]
    try:
        for path in candidates:
            if os.path.isfile(path):
                df = pd.read_csv(path)
                if {"mode", "step", "abs_error"}.issubset(df.columns):
                    # compute mean abs_error per mode and step
                    df = df[df["mode"].str.lower().isin(["iterative", "multi"])]
                    mae = df.groupby([df["mode"].str.lower(), df["step"]])["abs_error"].mean().unstack(fill_value=float('inf'))
                    steps = list(range(1, horizon + 1))
                    best_s, best_err = 1, float('inf')
                    for s in steps:
                        # error sum: iterative for <=s, multi for >s
                        err_it = mae.get("iterative", pd.Series()).reindex(steps).fillna(float('inf')).iloc[:s].sum()
                        err_mu = mae.get("multi", pd.Series()).reindex(steps).fillna(float('inf')).iloc[s:].sum()
                        total = err_it + err_mu
                        if total < best_err:
                            best_err, best_s = total, s
                    return best_s
        # fallback to midpoint
        return max(1, horizon // 2)
    except Exception:
        return max(1, horizon // 2)


def predict_iterative_calibrated(series, model, scaler, horizon: int, bias_adj: Union[float, dict[int, float]]) -> List[Tuple[pd.Timestamp, float]]:
    from darts import TimeSeries

    last_obs = series.end_time()
    future_covariates = extend_future_covariates(series, days=horizon)
    preds: List[Tuple[pd.Timestamp, float]] = []

    current_scaled = scaler.transform(series)
    for step in range(1, horizon + 1):
        pred_scaled = model.predict(n=1, series=current_scaled, future_covariates=future_covariates)
        if isinstance(pred_scaled, list):
            pred_scaled = pred_scaled[0]
        current_scaled = current_scaled.append(pred_scaled)
        pred_any = scaler.inverse_transform(pred_scaled)
        if isinstance(pred_any, list):
            pred_any = pred_any[0]
        pred_ts = cast(TimeSeries, pred_any)
        # compute prediction date then apply scalar or day-of-week bias adjustment
        pred_date = last_obs + timedelta(days=step)
        if isinstance(bias_adj, dict):
            bias = bias_adj.get(pred_date.weekday(), 0.0)
        else:
            bias = float(bias_adj)
        pred_value = float(pred_ts.last_value()) + bias
        preds.append((pred_date, pred_value))
    return preds


def train_and_predict_multi(series, horizon: int) -> List[Tuple[pd.Timestamp, float]]:
    from darts.dataprocessing.transformers import Scaler
    from darts.models import LightGBMModel
    from darts import TimeSeries

    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)
    future_covariates = extend_future_covariates(series, days=horizon)

    lags = [-1, -2, -3, -4, -5, -6, -7, -14, -21]
    model = LightGBMModel(
        lags=lags,
        lags_future_covariates=[0],
        output_chunk_length=horizon,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=5,
        feature_fraction=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(series_scaled, future_covariates=build_future_covariates(series))

    pred_scaled = model.predict(n=horizon, series=series_scaled, future_covariates=future_covariates)
    if isinstance(pred_scaled, list):
        pred_scaled = pred_scaled[0]
    pred_any = scaler.inverse_transform(pred_scaled)
    if isinstance(pred_any, list):
        pred_any = pred_any[0]
    pred_series = cast(TimeSeries, pred_any)
    # Extract per-step values aligned to dates
    vals = pred_series.values().flatten().tolist()
    dates = [series.end_time() + timedelta(days=i) for i in range(1, horizon + 1)]
    return list(zip(dates, [float(v) for v in vals]))


def compose_hybrid(iterative_cal: List[Tuple[pd.Timestamp, float]], multi: List[Tuple[pd.Timestamp, float]], switch: int) -> List[Tuple[pd.Timestamp, float]]:
    # Both lists should cover 1..horizon sequentially; pick according to switch
    out: List[Tuple[pd.Timestamp, float]] = []
    for i in range(len(iterative_cal)):
        step = i + 1
        dt_it, v_it = iterative_cal[i]
        dt_mu, v_mu = multi[i]
        dt = dt_it if dt_it == dt_mu else dt_it  # prefer iterative date if mismatch
        out.append((dt, v_it if step <= switch else v_mu))
    return out


def save_predictions(log_path: str, preds: List[Tuple[pd.Timestamp, float]]) -> None:
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


def main(argv: list[str] | None = None) -> int:
    import argparse

    ensure_deps()

    parser = argparse.ArgumentParser(description="Hybrid + calibration predictions to upcoming Sunday")
    parser.add_argument("--switch", type=int, default=None, help="Hybrid switch step: <=switch iterative, >switch multi (default: auto-optimal)")
    parser.add_argument("--bias", type=float, default=None, help="Override bias adjustment to add to iterative predictions (default: auto from msae_detailed.csv if available, else 0)")
    args = parser.parse_args(argv)

    print("Fetching NY subway data…")
    raw_df = get_ny_data()
    ny_df = clean_data(raw_df)
    print(f"Rows after cleaning: {len(ny_df):,}")

    series = build_series(ny_df)
    start_date = pd.Timestamp(series.start_time()).date()
    end_date = pd.Timestamp(series.end_time()).date()
    print(f"Series span: {start_date} -> {end_date}  (n={len(series)})")

    last_obs = series.end_time()
    wd = last_obs.weekday()
    days_until_sunday = (6 - wd) if wd != 6 else 7
    horizon = days_until_sunday
    print(f"Forecast horizon to Sunday: {horizon} day(s)")

    # Determine bias adjustment (scalar override or per-day)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.bias is None:
        bias_map = compute_iterative_bias_by_dow_from_file(script_dir)
        bias_adj = bias_map
        print(f"Iterative bias adjustment by weekday: {bias_map}")
    else:
        bias_adj = float(args.bias)
        print(f"Iterative global bias adjustment: {bias_adj:.1f}")

    print("Training iterative model…")
    it_model, scaler = train_iterative_model(series)
    print("Predicting (iterative calibrated)…")
    it_preds_cal = predict_iterative_calibrated(series, it_model, scaler, horizon=horizon, bias_adj=bias_adj)

    print("Training multi-step model…")
    try:
        mu_preds = train_and_predict_multi(series, horizon=horizon)
    except Exception as e:
        print(f"Multi-step model failed ({e}); falling back to iterative for all steps.")
        mu_preds = it_preds_cal[:]

    # Determine hybrid switch step
    if args.switch is None:
        switch = compute_optimal_switch_from_file(script_dir, horizon)
        print(f"Auto-optimal switch step determined: {switch}")
    else:
        switch = max(1, min(args.switch, horizon))
        print(f"Using provided switch step: {switch}")
    print(f"Composing hybrid with switch={switch} (iterative<=switch, multi>switch)…")
    hybrid = compose_hybrid(it_preds_cal, mu_preds, switch=switch)

    for dt, pv in hybrid:
        print(f"Predicted {dt.date()}: {int(round(pv))}")

    # Save CSV in phase3 directory
    log_file = os.path.join(script_dir, "subway_predictions.csv")
    save_predictions(log_file, hybrid)
    print(f"Logged predictions to: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
