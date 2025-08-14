"""
Backtest harness to compute MSAE (and related metrics) for different prediction modes.

This script runs a small rolling evaluation over recent candidate training dates and
compares three modes:
 - single-step (one-day forecast using a LightGBMModel)
 - multi-step (multi-output model to Sunday, from `predictionByDay_B.py`)
 - iterative (one-step model used iteratively, from `predictionByDay_A.py`)

Metrics reported per mode: count, MAE, MSE, Mean Error (bias), MSAE.

Notes:
 - MSAE here is reported as mean(error) to match 'mean signed absolute error' (signed errors preserved).
 - The script imports the multi-step and iterative functions from local modules when available.

Run from repository root or from this folder; it will locate local modules automatically.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from statistics import mean
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

# helpers to load modules by path (safe for local scripts)

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


# locate files in the same folder as this test script
ROOT = Path(__file__).parent
# prefer Zach versions; fallback gracefully
# Candidate filenames for the two modes (support renamed files)
MOD_CANDIDATES = {
    # explicit current filenames used by this harness
    "predictionByDay_B": ["predictionByDay_B.py"],
    "predictionByDay_A": ["predictionByDay_A.py"],
}

mods = {}
for key, candidates in MOD_CANDIDATES.items():
    for fname in candidates:
        p = ROOT / fname
        if p.exists():
            try:
                mods[key] = load_module(key, p)
                print(f"Loaded {fname} as {key}")
            except Exception as e:
                print(f"Warning: failed to load {p}: {e}")
            break

# Local copy of data fetch/clean/build to avoid tight coupling

def get_ny_data(timeout: int = 30) -> pd.DataFrame:
    import requests

    url = "https://data.ny.gov/resource/sayj-mze2.json"
    params = {"$select": "date,mode,count", "$where": "mode like 'Subway'", "$order": "date", "$limit": "50000"}
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mode" in df.columns:
        df = df[df["mode"].astype(str).str.contains("Subway", regex=False, na=False, case=False)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["date", "count"]).sort_values("date")
    return df[["date", "count"]]


def build_series(df: pd.DataFrame):
    from darts import TimeSeries

    daily = df.set_index("date")["count"].asfreq("D").interpolate("time").ffill().bfill()
    return TimeSeries.from_series(daily, fill_missing_dates=True, freq="D").astype("float32")


# single-step predictor implemented inline (so we don't depend on another file)
def single_step_predict(train_series, horizon: int = 1) -> Tuple[pd.Timestamp, float]:
    from darts.dataprocessing.transformers import Scaler
    from darts.models import LightGBMModel

    # Build future_covariates for training span (try local modules first)
    future_covariates = None
    try:
        if mods.get("predictionByDay_A"):
            future_covariates = mods["predictionByDay_A"].build_future_covariates(train_series)
        elif mods.get("predictionByDay_B"):
            future_covariates = mods["predictionByDay_B"].build_future_covariates(train_series)
    except Exception:
        future_covariates = None

    if future_covariates is None:
        from darts.utils.timeseries_generation import datetime_attribute_timeseries
        import pandas as pd
        import numpy as np
        import holidays as holidays
        from darts import TimeSeries
        idx = pd.date_range(start=train_series.start_time(), end=train_series.end_time(), freq="D")
        dow = datetime_attribute_timeseries(idx, attribute="weekday", one_hot=True)
        moy = datetime_attribute_timeseries(idx, attribute="month", one_hot=True)
        weekend_flag = pd.Series((idx.weekday >= 5).astype(np.int8), index=idx)
        is_weekend = TimeSeries.from_series(weekend_flag, freq="D")
        us_holidays = holidays.country_holidays('US', subdiv='NY')
        holiday_flag = pd.Series(idx.map(lambda d: 1 if d in us_holidays else 0), index=idx)
        holidays_ts = TimeSeries.from_series(holiday_flag, freq="D")
        future_covariates = dow.stack(moy).stack(is_weekend).stack(holidays_ts)

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)

    lags = [-1, -2, -3, -4, -5, -6, -7, -14, -21]
    model = LightGBMModel = None
    from darts.models import LightGBMModel as _LGB
    model = _LGB(
        lags=lags,
        lags_future_covariates=[0],
        output_chunk_length=1,
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1,
    )

    model.fit(train_scaled, future_covariates=future_covariates)
    # Build extended covariates covering the forecast horizon
    future_covariates_ext = None
    try:
        if mods.get("predictionByDay_A"):
            future_covariates_ext = mods["predictionByDay_A"].extend_future_covariates(train_series, days=horizon)
        elif mods.get("predictionByDay_B"):
            future_covariates_ext = mods["predictionByDay_B"].extend_future_covariates(train_series, days=horizon)
    except Exception:
        future_covariates_ext = None

    if future_covariates_ext is None:
        # inline fallback to extend covariates
        from darts.utils.timeseries_generation import datetime_attribute_timeseries
        import pandas as pd
        import numpy as np
        import holidays as holidays
        from darts import TimeSeries
        last_date = train_series.end_time()
        end = last_date + timedelta(days=horizon)
        idx = pd.date_range(start=train_series.start_time(), end=end, freq="D")
        dow = datetime_attribute_timeseries(idx, attribute="weekday", one_hot=True)
        moy = datetime_attribute_timeseries(idx, attribute="month", one_hot=True)
        weekend_flag = pd.Series((idx.weekday >= 5).astype(np.int8), index=idx)
        is_weekend = TimeSeries.from_series(weekend_flag, freq="D")
        us_holidays = holidays.country_holidays('US', subdiv='NY')
        holiday_flag = pd.Series(idx.map(lambda d: 1 if d in us_holidays else 0), index=idx)
        holidays_ts = TimeSeries.from_series(holiday_flag, freq="D")
        future_covariates_ext = dow.stack(moy).stack(is_weekend).stack(holidays_ts)

    pred_scaled = model.predict(n=1, future_covariates=future_covariates_ext)
    if isinstance(pred_scaled, list):
        pred_scaled = pred_scaled[0]
    pred_any = scaler.inverse_transform(pred_scaled)
    if isinstance(pred_any, list):
        pred_any = pred_any[0]
    from darts import TimeSeries
    pred_ts = pred_any
    val = float(pred_ts.last_value())
    date = train_series.end_time() + timedelta(days=1)
    return date, val


def evaluate_predictions(actuals: pd.Series, preds: List[Tuple[pd.Timestamp, float]], train_end_date) -> List[dict]:
    """Return list of dicts with detailed error info for each prediction.

    Each dict contains: train_end, mode (filled by caller), target_date, predicted, actual, error, step
    """
    rows = []
    for dt, pv in preds:
        ad = actuals.get(dt, None)
        if ad is None:
            continue
        step = (pd.Timestamp(dt).date() - pd.Timestamp(train_end_date).date()).days
        rows.append({
            "train_end": pd.Timestamp(train_end_date).date(),
            "target_date": pd.Timestamp(dt).date(),
            "predicted": float(pv),
            "actual": float(ad),
            "error": float(pv - ad),
            "abs_error": abs(float(pv - ad)),
            "squared_error": float((pv - ad) ** 2),
            "step": int(step),
        })
    return rows


def compute_metrics(errors: List[float]):
    if not errors:
        return {"count": 0}
    arr = np.array(errors, dtype=float)
    return {
        "count": len(arr),
        "MAE": float(np.mean(np.abs(arr))),
        "MSE": float(np.mean(arr ** 2)),
        "MeanError": float(np.mean(arr)),
        "MSAE": float(np.mean(arr)),  # labeled as MSAE (mean signed absolute error == mean error)
    }


def main():
    raw = get_ny_data()
    df = clean_data(raw)
    full_dates = list(df["date"].dt.date)
    last_date = full_dates[-1]

    # map of actuals
    # use python date objects as keys for consistent lookups
    actuals = {row.date.date(): float(row.count) for row in df.itertuples(index=False)}

    # candidates: dates where upcoming Sunday is available in actuals
    candidates = []
    for d in df["date"]:
        d0 = d.date()
        wd = d0.weekday()
        days_until_sunday = (6 - wd) if wd != 6 else 7
        target = d0 + timedelta(days=days_until_sunday)
        if target <= last_date and (d0 >= (df["date"].min().date() + timedelta(days=30))):
            candidates.append(d0)

    # use the last up to 10 candidates
    candidates = sorted(set(candidates))[-10:]

    # collect detailed rows across all runs
    detailed_rows: List[dict] = []
    mode_errors = {"single": [], "multi": [], "iterative": []}

    for train_end in candidates:
        # train_series up to train_end
        train_df = df[df["date"].dt.date <= train_end]
        train_series = build_series(train_df)

        # single-step
        try:
            s_date, s_val = single_step_predict(train_series)
            s_rows = evaluate_predictions(actuals, [(s_date.date(), s_val)], train_end)
            for r in s_rows:
                r["mode"] = "single"
            detailed_rows.extend(s_rows)
            mode_errors["single"].extend([r["error"] for r in s_rows])
        except Exception as e:
            print(f"single-step failed for {train_end}: {e}")

        # multi-step (multi-output)
        if mods.get("predictionByDay_B"):
            try:
                multi_preds = mods["predictionByDay_B"].train_and_forecast_to_sunday(train_series)
                # convert dates to date() keys
                multi_preds = [(d.date(), v) for d, v in multi_preds]
                m_rows = evaluate_predictions(actuals, multi_preds, train_end)
                for r in m_rows:
                    r["mode"] = "multi"
                detailed_rows.extend(m_rows)
                mode_errors["multi"].extend([r["error"] for r in m_rows])
            except Exception as e:
                print(f"multi-step failed for {train_end}: {e}")

        # iterative
            if mods.get("predictionByDay_A"):
                try:
                    it_preds = mods["predictionByDay_A"].train_and_forecast_iterative(train_series)
                    it_preds = [(d.date(), v) for d, v in it_preds]
                    it_rows = evaluate_predictions(actuals, it_preds, train_end)
                    for r in it_rows:
                        r["mode"] = "iterative"
                    detailed_rows.extend(it_rows)
                    mode_errors["iterative"].extend([r["error"] for r in it_rows])
                except Exception as e:
                    print(f"iterative failed for {train_end}: {e}")

    # Save detailed master CSV
    out_master = ROOT / "phase2_data" / "msae_detailed.csv"
    import csv

    if detailed_rows:
        keys = ["train_end", "mode", "target_date", "step", "predicted", "actual", "error", "abs_error", "squared_error"]
        with open(out_master, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in detailed_rows:
                # ensure key order and include mode
                row = {k: r.get(k) for k in keys}
                w.writerow(row)
        print(f"Saved detailed results to: {out_master}")

    # compute overall metrics and per-horizon breakdowns
    for m in ["single", "multi", "iterative"]:
        metrics = compute_metrics(mode_errors[m])
        print(f"Mode: {m} | {metrics}")

        # per-horizon breakdown
        rows_mode = [r for r in detailed_rows if r.get("mode") == m]
        if not rows_mode:
            continue
        df_mode = pd.DataFrame(rows_mode)
        grouped = df_mode.groupby("step").agg({"abs_error": "mean", "squared_error": "mean", "error": "mean", "target_date": "count"})
        grouped = grouped.rename(columns={"abs_error": "MAE", "squared_error": "MSE", "error": "MeanError", "target_date": "count"})
        print(f"Per-horizon breakdown for mode={m}:\n{grouped}\n")


if __name__ == "__main__":
    main()
