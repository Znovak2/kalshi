"""
NYC Subway Ridership Forecasting

- Fetches MTA ridership data from a public Socrata API.
- Filters to subway-only entries.
- Computes the past-week average.
- Trains a simple OLS model with trend + day-of-week seasonality.
- Forecasts daily counts through the upcoming Sunday and reports RMSE and R².
- Includes basic unit tests.

Python: 3.12
Allowed third-party libraries: pandas, numpy, requests
"""

from __future__ import annotations

import unittest  # Standard library for tests
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import hashlib


API_URL = "https://data.ny.gov/resource/sayj-mze2.json"
DEFAULT_TIMEOUT = 20  # seconds
PAGE_LIMIT = 50000  # use pagination with $offset to exceed 1000 rows

# File paths for logging predictions and accuracy
_THIS_FILE = Path(__file__).resolve()
PREDICTIONS_CSV = _THIS_FILE.with_name("predictions.csv")
ACCURACY_CSV = _THIS_FILE.with_name("accuracy.csv")


def fetch_mta_data(url: str = API_URL, limit: int = PAGE_LIMIT) -> list[dict]:
    """
    Fetch all rows from the MTA Socrata dataset with pagination.

    Parameters
    ----------
    url : str
        Socrata API endpoint URL.
    limit : int
        Number of rows per page.

    Returns
    -------
    list[dict]
        List of raw JSON records.

    Notes
    -----
    - Uses $order=Date ASC to get chronological ordering if available.
    - Gracefully handles network, HTTP, and JSON errors, returning an empty list.
    """
    records: list[dict] = []
    offset = 0

    while True:
        params = {
            "$limit": limit,
            "$offset": offset,
            # Use lowercase 'date' as SoQL generally normalizes field ids to lower snake case
            "$order": "date ASC",
            "$select": "date,mode,count",
        }
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            page = resp.json()
            if not isinstance(page, list):
                # Malformed JSON payload
                return []
        except requests.Timeout:
            return []
        except requests.RequestException:
            return []
        except ValueError:
            # JSON decode error
            return []

        if not page:
            break

        records.extend(page)
        if len(page) < limit:
            break
        offset += limit

        # Safety: cap total rows to avoid unbounded pulls if API misbehaves
        if offset > 1_000_000:
            break

    return records


def _find_key(d: dict, candidates: list[str]) -> str | None:
    """
    Case-insensitive lookup of a key from a set of candidates.

    Parameters
    ----------
    d : dict
        Source dictionary with string keys.
    candidates : list[str]
        Candidate keys ordered by preference.

    Returns
    -------
    Optional[str]
        The matching key in original case if found, else None.
    """
    lower_map = {k.lower(): k for k in d.keys()}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def to_subway_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Convert raw JSON records to a cleaned DataFrame for subway ridership only.

    Parameters
    ----------
    records : list[dict]
        Raw Socrata JSON rows.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'mode', 'count'] with correct dtypes, filtered to subway-like modes.

    Notes
    -----
    - Attempts robust column mapping in case labels vary.
    - Filters to modes that contain 'subway' or 'sir' (Staten Island Railway) case-insensitively.
    """
    if not records:
        return pd.DataFrame(columns=["date", "mode", "count"])

    # Map columns robustly using the first row (assume uniform schema)
    sample = records[0]
    date_key = _find_key(sample, ["date", "report_date", "day"])
    mode_key = _find_key(sample, ["mode", "transportation_type", "type", "category"])
    count_key = _find_key(sample, ["count", "ridership", "traffic", "total"])

    if not (date_key and mode_key and count_key):
        return pd.DataFrame(columns=["date", "mode", "count"])

    df = pd.DataFrame(records)[[date_key, mode_key, count_key]].rename(
        columns={date_key: "date", mode_key: "mode", count_key: "count"}
    )

    # Coerce dtypes
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["mode"] = df["mode"].astype(str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    # Drop malformed rows
    df = df.dropna(subset=["date", "mode", "count"])

    # Filter to subway-like rows
    mode_lower = df["mode"].str.lower()
    is_subway = (
        mode_lower.str.contains("subway")
        | mode_lower.str.contains("staten island railway")
        | mode_lower.str.fullmatch(r"\s*sir\s*")
    )
    df = df.loc[is_subway].copy()

    # Sort by date and aggregate per date in case multiple subway rows exist
    df = df.groupby("date", as_index=False).agg({"count": "sum"})
    df["mode"] = "Subways"

    return df[["date", "mode", "count"]].sort_values("date").reset_index(drop=True)


def most_recent_full_week_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a 7-day Sunday-to-Sunday window if available; otherwise a Mon–Fri workweek.

    Parameters
    ----------
    df : pd.DataFrame
        Clean subway DataFrame with 'date' and 'count'.

    Returns
    -------
    pd.DataFrame
        Windowed DataFrame for averaging. May be empty if data insufficient.
    """
    if df.empty:
        return df

    df = df.sort_values("date").drop_duplicates("date")
    dates = df["date"]

    # Try Sunday-to-Sunday inclusive 7 days
    # Pandas weekday: Monday=0, Sunday=6
    last_sunday = dates[dates.dt.weekday == 6].max()
    if pd.notna(last_sunday):
        window_start = last_sunday - pd.Timedelta(days=6)
        week = df[(df["date"] >= window_start) & (df["date"] <= last_sunday)]
        # Ensure 7 unique calendar days present
        if week["date"].dt.normalize().nunique() == 7:
            return week

    # Fallback: most recent Mon–Fri full workweek (5 days)
    last_friday = dates[dates.dt.weekday == 4].max()
    if pd.notna(last_friday):
        window_start = last_friday - pd.Timedelta(days=4)
        week = df[(df["date"] >= window_start) & (df["date"] <= last_friday)]
        if week["date"].dt.normalize().nunique() == 5:
            return week

    # If neither full window exists, return the last up-to-7 rows
    return df.tail(7)


def average_over_window(df_window: pd.DataFrame) -> float | None:
    """
    Compute mean count over the given window.

    Parameters
    ----------
    df_window : pd.DataFrame
        Windowed DataFrame.

    Returns
    -------
    Optional[float]
        Mean of 'count' if available, else None.
    """
    if df_window.empty:
        return None
    return float(df_window["count"].mean())


def _design_matrix(dates: pd.Series) -> np.ndarray:
    """
    Build design matrix with intercept, linear trend, and day-of-week seasonality.

    Parameters
    ----------
    dates : pd.Series
        Series of pandas Timestamps.

    Returns
    -------
    np.ndarray
        Matrix of shape (n, p): [1, trend, dow(Mon..Sat)] where Sunday is baseline.
    """
    n = len(dates)
    # Trend as day index from the first date
    t = (dates.dt.normalize() - dates.min().normalize()).dt.days.to_numpy(dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-9)  # normalize for stability

    dow = dates.dt.weekday.to_numpy()
    X_list = [np.ones(n), t]

    # Create 6 dummy variables for Monday(0)..Saturday(5); Sunday(6) is baseline
    for d in range(6):
        X_list.append((dow == d).astype(float))

    X = np.vstack(X_list).T
    return X


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Ordinary Least Squares fit.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    tuple
        (beta, cov_beta, sigma2) where:
        - beta: coefficients
        - cov_beta: (X'X)^-1 * sigma2
        - sigma2: residual variance
    """
    # Use lstsq for numerical robustness
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    n, p = X.shape
    # Compute SSE safely if residuals empty (perfect fit case)
    sse = float(((y - y_hat) ** 2).sum())
    dof = max(n - p, 1)
    sigma2 = sse / dof
    # Compute (X'X)^-1 via pinv to handle near singularity
    xtx_inv = np.linalg.pinv(X.T @ X)
    cov_beta = xtx_inv * sigma2
    return beta, cov_beta, sigma2


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Train model on historical data and evaluate metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Clean subway ridership DataFrame.

    Returns
    -------
    dict
        {
            "beta": np.ndarray,
            "cov": np.ndarray,
            "sigma2": float,
            "rmse": float | None,
            "r2": float | None,
            "train_dates": pd.Series,
            "train_counts": pd.Series
        }

    Notes
    -----
    - Uses the last up to 180 observations for modeling.
    - Splits into 75% train / 25% test chronologically when possible (>= 28 rows).
    - Falls back to naive seasonal mean if insufficient data (< 14 rows).
    """
    result = {
        "beta": None,
        "cov": None,
        "sigma2": None,
        "rmse": None,
        "r2": None,
        "train_dates": pd.Series(dtype="datetime64[ns]"),
        "train_counts": pd.Series(dtype=float),
    }

    if df.empty:
        return result

    df = df.sort_values("date").drop_duplicates("date").tail(180).reset_index(drop=True)
    n = len(df)
    if n < 14:
        # Too little data for a reliable regression
        return result

    split = int(max(1, np.floor(n * 0.75)))
    train = df.iloc[:split]
    test = df.iloc[split:] if n - split >= 7 else None

    # Fit on training set
    X_train = _design_matrix(train["date"])
    y_train = train["count"].to_numpy(dtype=float)
    beta, cov_beta, sigma2 = _ols_fit(X_train, y_train)

    # Evaluate on test set if present
    rmse = None
    r2 = None
    if test is not None and not test.empty:
        X_test = _design_matrix(test["date"])
        y_test = test["count"].to_numpy(dtype=float)
        y_pred = X_test @ beta
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        ss_res = float(np.sum((y_test - y_pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2)) or 1.0
        r2 = float(1.0 - ss_res / ss_tot)

    result.update(
        {
            "beta": beta,
            "cov": cov_beta,
            "sigma2": float(sigma2),
            "rmse": rmse,
            "r2": r2,
            "train_dates": train["date"],
            "train_counts": train["count"],
        }
    )
    return result


def upcoming_sunday(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """
    Compute the upcoming Sunday's date (inclusive of today if it's Sunday).

    Parameters
    ----------
    today : Optional[pd.Timestamp]
        Reference date. Defaults to current day.

    Returns
    -------
    pd.Timestamp
        Upcoming Sunday's normalized timestamp.
    """
    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.Timestamp(today).normalize()

    days = (6 - today.weekday()) % 7  # Monday=0, Sunday=6
    return today + pd.Timedelta(days=int(days))


def forecast_to_sunday(
    model: dict, start_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    Forecast daily subway ridership from tomorrow through upcoming Sunday.

    Parameters
    ----------
    model : dict
        Output of train_and_evaluate().
    start_date : Optional[pd.Timestamp]
        Reference date. Defaults to "tomorrow" in local time.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'forecast', 'pi_low', 'pi_high'].

    Notes
    -----
    - If model is missing (insufficient data), returns an empty DataFrame.
    - 95% prediction intervals are computed using sigma2 and cov(beta).
    """
    beta = model.get("beta", None)
    cov = model.get("cov", None)
    sigma2 = model.get("sigma2", None)
    train_dates = model.get("train_dates", pd.Series(dtype="datetime64[ns]"))

    if beta is None or cov is None or sigma2 is None or train_dates.empty:
        return pd.DataFrame(columns=["date", "forecast", "pi_low", "pi_high"])

    if start_date is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.Timestamp(start_date).normalize()

    # Forecast from tomorrow through the upcoming Sunday
    start = today + pd.Timedelta(days=1)
    end = upcoming_sunday(today)
    if start > end:
        return pd.DataFrame(columns=["date", "forecast", "pi_low", "pi_high"])

    future_dates = pd.date_range(start=start, end=end, freq="D")
    # Build X using the combined timeline so trend aligns with training baseline
    # Concatenate for consistent transformation, then slice
    combined = pd.concat([train_dates.reset_index(drop=True), pd.Series(future_dates)], ignore_index=True)
    X_all = _design_matrix(combined)
    X_future = X_all[-len(future_dates) :]

    y_hat = X_future @ beta
    # 95% prediction intervals: variance = x'cov_beta x + sigma2
    # where sigma2 accounts for observation noise (prediction interval, not mean CI).
    variances = np.einsum("ij,jk,ik->i", X_future, cov, X_future) + sigma2
    se = np.sqrt(np.maximum(variances, 0.0))
    z = 1.96

    out = pd.DataFrame(
        {
            "date": future_dates,
            "forecast": y_hat,
            "pi_low": y_hat - z * se,
            "pi_high": y_hat + z * se,
        }
    )
    return out


def _model_id(model: dict) -> str:
    """Create a lightweight model identifier based on coefficients and train end date."""
    beta = model.get("beta")
    dates = model.get("train_dates", pd.Series(dtype="datetime64[ns]"))
    train_end = str(dates.max().date()) if not dates.empty else "NA"
    if beta is None:
        return f"no-model_{train_end}"
    b = np.asarray(beta, dtype=float)
    h = hashlib.sha1(b.tobytes() + train_end.encode("utf-8")).hexdigest()[:10]
    return f"m_{h}_{train_end}"


def log_predictions(forecasts: pd.DataFrame, model: dict, run_time: pd.Timestamp | None = None) -> int:
    """
    Append forecasts for upcoming days to a CSV log with metadata.

    Returns number of rows written.
    """
    if forecasts is None or forecasts.empty:
        return 0

    run_time = pd.Timestamp.utcnow() if run_time is None else pd.Timestamp(run_time)
    run_iso = run_time.isoformat()
    model_id = _model_id(model)
    dates = model.get("train_dates", pd.Series(dtype="datetime64[ns]"))
    train_start = str(dates.min().date()) if not dates.empty else None
    train_end = str(dates.max().date()) if not dates.empty else None
    rmse = model.get("rmse", None)
    r2 = model.get("r2", None)

    to_write = forecasts.copy()
    to_write["run_time_utc"] = run_iso
    to_write["target_date"] = to_write["date"].dt.normalize()
    to_write["model_id"] = model_id
    to_write["train_start"] = train_start
    to_write["train_end"] = train_end
    to_write["rmse"] = rmse
    to_write["r2"] = r2

    cols = [
        "run_time_utc",
        "target_date",
        "forecast",
        "pi_low",
        "pi_high",
        "model_id",
        "train_start",
        "train_end",
        "rmse",
        "r2",
    ]
    to_write = to_write[cols]

    # Append to CSV (create if missing)
    header = not PREDICTIONS_CSV.exists()
    to_write.to_csv(PREDICTIONS_CSV, mode="a", header=header, index=False)
    return len(to_write)


def evaluate_predictions(df: pd.DataFrame, lookback_days: int = 120) -> tuple[int, pd.DataFrame]:
    """
    Compare logged predictions with realized actuals where available and write accuracy rows.

    Returns (n_new, accuracy_df_recent)
    """
    if df is None or df.empty or not PREDICTIONS_CSV.exists():
        # Nothing to evaluate
        if ACCURACY_CSV.exists():
            acc = pd.read_csv(ACCURACY_CSV, parse_dates=["target_date", "run_time_utc"])  # type: ignore[arg-type]
            acc = acc.sort_values("target_date")
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
            acc_recent = acc[acc["target_date"] >= cutoff].copy()
            return 0, acc_recent
        return 0, pd.DataFrame(columns=[
            "run_time_utc","target_date","forecast","actual","error","abs_error","sq_error","mape","model_id","train_end"
        ])

    preds = pd.read_csv(PREDICTIONS_CSV, parse_dates=["target_date", "run_time_utc"])  # type: ignore[arg-type]
    preds = preds.sort_values(["target_date", "run_time_utc"]).dropna(subset=["target_date"]).copy()

    # Actuals by date
    actuals = df.copy()
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.normalize()
    actuals = actuals.groupby("date", as_index=False)["count"].sum().rename(columns={"date": "target_date", "count": "actual"})

    merged = preds.merge(actuals, on="target_date", how="inner")
    if merged.empty:
        # No actuals yet matching
        if ACCURACY_CSV.exists():
            acc = pd.read_csv(ACCURACY_CSV, parse_dates=["target_date", "run_time_utc"])  # type: ignore[arg-type]
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
            acc_recent = acc[acc["target_date"] >= cutoff].copy()
            return 0, acc_recent
        return 0, pd.DataFrame(columns=[
            "run_time_utc","target_date","forecast","actual","error","abs_error","sq_error","mape","model_id","train_end"
        ])

    merged["error"] = merged["forecast"] - merged["actual"]
    merged["abs_error"] = merged["error"].abs()
    merged["sq_error"] = merged["error"] ** 2
    denom = merged["actual"].replace(0, np.nan)
    merged["mape"] = (merged["abs_error"] / denom).clip(upper=10.0)
    merged["train_end"] = merged["train_end"].astype(str)

    # Load existing accuracy to avoid duplicates
    if ACCURACY_CSV.exists():
        acc = pd.read_csv(ACCURACY_CSV, parse_dates=["target_date", "run_time_utc"])  # type: ignore[arg-type]
    else:
        acc = pd.DataFrame(columns=[
            "run_time_utc","target_date","forecast","actual","error","abs_error","sq_error","mape","model_id","train_end"
        ])

    # Deduplicate by (run_time_utc, target_date)
    key_cols = ["run_time_utc", "target_date", "model_id"]
    if not acc.empty:
        existing = acc[key_cols].astype(str).agg("|".join, axis=1)
        new_keys = merged[key_cols].astype(str).agg("|".join, axis=1)
        mask_new = ~new_keys.isin(set(existing))
        to_add = merged.loc[mask_new, [
            "run_time_utc","target_date","forecast","actual","error","abs_error","sq_error","mape","model_id","train_end"
        ]].copy()
    else:
        to_add = merged[[
            "run_time_utc","target_date","forecast","actual","error","abs_error","sq_error","mape","model_id","train_end"
        ]].copy()

    n_new = len(to_add)
    if n_new > 0:
        header = not ACCURACY_CSV.exists()
        to_add.to_csv(ACCURACY_CSV, mode="a", header=header, index=False)

    # Return recent slice for summary
    acc_all = pd.read_csv(ACCURACY_CSV, parse_dates=["target_date", "run_time_utc"])  # type: ignore[arg-type]
    acc_all = acc_all.sort_values("target_date")
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    acc_recent = acc_all[acc_all["target_date"] >= cutoff].copy()
    return n_new, acc_recent


def rolling_bias(acc_df: pd.DataFrame, window: int = 28) -> float | None:
    """Compute mean error (forecast - actual) over the last `window` evaluated predictions."""
    if acc_df is None or acc_df.empty:
        return None
    acc_df = acc_df.sort_values("target_date").tail(window)
    if acc_df.empty:
        return None
    return float(acc_df["error"].mean())


def trailing_avg_on_final_day(
    df: pd.DataFrame,
    model: dict,
    final_day: pd.Timestamp,
) -> tuple[float | None, pd.DataFrame]:
    """
    Compute the 7-day trailing average ending on `final_day` by blending actuals and model predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Clean subway DataFrame with columns ['date','count'].
    model : dict
        Output of train_and_evaluate(). Must contain 'beta', 'cov', 'sigma2', 'train_dates'.
    final_day : pd.Timestamp
        Target final day (typically upcoming Sunday).

    Returns
    -------
    (avg, details_df)
        avg : float | None
            The 7-day trailing average if computable, else None.
        details_df : pd.DataFrame
            Columns: ['date','value','source'] for the 7-day window.
    """
    if df is None or df.empty:
        return None, pd.DataFrame(columns=["date", "value", "source"])

    beta = model.get("beta")
    train_dates = model.get("train_dates", pd.Series(dtype="datetime64[ns]"))

    # Build the 7-day window [final_day-6, final_day]
    final_day = pd.Timestamp(final_day).normalize()
    window = pd.date_range(end=final_day, periods=7, freq="D")

    # Actuals map (ensure one row per date)
    df_use = df[["date", "count"]].copy()
    df_use["date"] = pd.to_datetime(df_use["date"]).dt.normalize()
    actuals = (
        df_use.groupby("date", as_index=False)["count"].sum().set_index("date")["count"]
    )

    # Prepare predictions for the window if model is available
    preds = None
    if beta is not None and not train_dates.empty:
        combined = pd.concat(
            [train_dates.reset_index(drop=True), pd.Series(window)], ignore_index=True
        )
        X_all = _design_matrix(combined)
        X_win = X_all[-len(window) :]
        preds_vals = X_win @ beta
        preds = pd.Series(preds_vals, index=window)

    # Blend actuals (preferred) with predictions
    values = []
    sources = []
    for d in window:
        if d in actuals.index:
            values.append(float(actuals.loc[d]))
            sources.append("actual")
        elif preds is not None and d in preds.index:
            values.append(float(preds.loc[d]))
            sources.append("forecast")
        else:
            values.append(np.nan)
            sources.append("missing")

    details = pd.DataFrame({"date": window, "value": values, "source": sources})
    if details["value"].notna().sum() == len(window):
        return float(np.mean(details["value"])), details
    # If some missing, return mean of available values (best-effort); caller can decide
    if details["value"].notna().sum() >= 3:
        return float(np.nanmean(details["value"])), details
    return None, details

def main() -> None:
    """
    Example usage:
    - Downloads data
    - Produces past-week average
    - Trains model and prints metrics
    - Outputs forecasts through the upcoming Sunday
    """
    # Fetch and prepare data
    records = fetch_mta_data()
    df = to_subway_dataframe(records)

    # Past-week average
    past_window = most_recent_full_week_window(df)
    past_avg = average_over_window(past_window)

    # Train and evaluate
    model = train_and_evaluate(df)

    # Forecasts
    forecasts = forecast_to_sunday(model)

    # Outputs
    print("Summary:")
    print(f"- Rows (subway-only): {len(df)}")
    if not df.empty:
        print(f"- Last observation date: {df['date'].max().date()}")

    if past_avg is not None:
        # Sunday-to-Sunday if available, else weekdays
        days = len(past_window)
        print(f"- Past average over last {days} day(s): {past_avg:,.0f}")
    else:
        print("- Past average: unavailable")

    rmse = model.get("rmse", None)
    r2 = model.get("r2", None)
    if rmse is not None:
        print(f"- RMSE (holdout): {rmse:,.0f}")
    if r2 is not None:
        print(f"- R^2 (holdout): {r2:.3f}")

    if not forecasts.empty:
        print("\nForecasts (through upcoming Sunday):")
        for _, row in forecasts.iterrows():
            d = row["date"].date()
            f = row["forecast"]
            lo = row["pi_low"]
            hi = row["pi_high"]
            print(f"  {d}: {f:,.0f}  (95% PI: {lo:,.0f} – {hi:,.0f})")
    else:
        print("\nForecasts: none (insufficient data or date window empty)")

    # Persist predictions for day-ahead evaluation and compute accuracy for realized dates
    n_logged = log_predictions(forecasts, model)
    n_new, acc_recent = evaluate_predictions(df)

    if n_logged:
        print(f"\nLogged {n_logged} forecast(s) to {PREDICTIONS_CSV.name}.")
    if n_new or (acc_recent is not None and not acc_recent.empty):
        # Summarize last 30 evaluated predictions
        acc_tail = acc_recent.sort_values("target_date").tail(30)
        if not acc_tail.empty:
            mae = float(acc_tail["abs_error"].mean())
            rmse30 = float(np.sqrt(acc_tail["sq_error"].mean()))
            bias30 = float(acc_tail["error"].mean())
            mape = float(acc_tail["mape"].mean()) if "mape" in acc_tail else np.nan
            print("Accuracy (last 30 evaluated predictions):")
            print(f"- MAE: {mae:,.0f}; RMSE: {rmse30:,.0f}; Bias: {bias30:,.0f}; MAPE: {mape*100:.2f}%" if not np.isnan(mape) else f"- MAE: {mae:,.0f}; RMSE: {rmse30:,.0f}; Bias: {bias30:,.0f}")
        else:
            print("No evaluated predictions yet (waiting for actuals).")

    # Final target: 7-day trailing average on the final day (upcoming Sunday)
    today = pd.Timestamp.today().normalize()
    final_day = upcoming_sunday(today)
    trailing_avg, details = trailing_avg_on_final_day(df, model, final_day)
    print("\nFinal target (7-day trailing average on final day):")
    print(f"- Final day: {final_day.date()}")
    if trailing_avg is not None:
        n_actual = int((details["source"] == "actual").sum())
        n_forecast = int((details["source"] == "forecast").sum())
        # Optional bias correction based on recent evaluated predictions
        bias = rolling_bias(acc_recent) if acc_recent is not None and not acc_recent.empty else None
        if bias is not None:
            # Apply bias only to forecast slots
            corrected = details.copy()
            mask_fc = corrected["source"] == "forecast"
            corrected.loc[mask_fc, "value"] = corrected.loc[mask_fc, "value"] - bias
            corrected_avg = float(corrected["value"].mean()) if corrected["value"].notna().all() else float(np.nanmean(corrected["value"]))
            print(
                f"- 7-day trailing avg ending {final_day.date()}: {trailing_avg:,.0f} (bias-corrected: {corrected_avg:,.0f}) "
                f"(sources: {n_actual} actual, {n_forecast} forecast; bias={bias:,.0f})"
            )
        else:
            print(
                f"- 7-day trailing avg ending {final_day.date()}: {trailing_avg:,.0f} "
                f"(sources: {n_actual} actual, {n_forecast} forecast)"
            )
    else:
        print("- 7-day trailing avg: unavailable (insufficient data)")


# ------------------------------ Tests ------------------------------


class TestNycSubwayRiders(unittest.TestCase):
    """Unit tests covering typical and edge cases."""

    def test_empty_records(self):
        df = to_subway_dataframe([])
        self.assertTrue(df.empty)

        model = train_and_evaluate(df)
        self.assertIsNone(model["rmse"])
        self.assertTrue(forecast_to_sunday(model).empty)

    def test_filter_and_average(self):
        # Mixed modes, only subway-like entries should remain
        data = [
            {"date": "2024-01-07", "mode": "Subways", "count": "200"},
            {"date": "2024-01-08", "mode": "Buses", "count": "100"},
            {"date": "2024-01-09", "mode": "Staten Island Railway (SIR)", "count": "50"},
            {"date": "2024-01-10", "mode": "LIRR", "count": "10"},
            {"date": "2024-01-11", "mode": "Subways", "count": "220"},
            {"date": "2024-01-12", "mode": "Subways", "count": "180"},
            {"date": "2024-01-13", "mode": "Subways", "count": "210"},  # Sunday
        ]
        df = to_subway_dataframe(data)
        self.assertEqual(df["date"].nunique(), 5)  # only 5 subway-like days remain aggregated
        # Past week window should fall back to last 5 days if Sunday-to-Sunday not complete
        w = most_recent_full_week_window(df)
        self.assertIn(len(w), (5, 7))
        avg = average_over_window(w)
        self.assertIsInstance(avg, float)

    def test_training_and_forecast_basic(self):
        # Build 60 days of synthetic subway data with weekly seasonality
        dates = pd.date_range("2024-03-01", periods=60, freq="D")
        dow = dates.weekday
        base = 2000 + (np.array(dow) == 5) * 200 + (np.array(dow) == 6) * 300  # Sat/Sun bumps
        trend = np.linspace(0, 100, 60)
        noise = np.random.default_rng(0).normal(0, 50, 60)
        counts = base + trend + noise

        df = pd.DataFrame({"date": dates, "mode": "Subways", "count": counts})
        model = train_and_evaluate(df)
        self.assertIsNotNone(model["beta"])
        fc = forecast_to_sunday(model, start_date=pd.Timestamp("2024-05-01"))
        # There will be forecasts up to Sunday of that week
        self.assertTrue(len(fc) >= 1)
        self.assertTrue({"forecast", "pi_low", "pi_high"}.issubset(fc.columns))


if __name__ == "__main__":
    # Example usage (prints summary, metrics, and forecasts)
    main()

    # To run tests:
    # - Via CLI: python -m unittest c:\Users\znova\OneDrive\Documents\LaptopCodespaces\kalshi\transportation\NycSubwayRiders.py
    # - Or uncomment below:
    # unittest.main()
