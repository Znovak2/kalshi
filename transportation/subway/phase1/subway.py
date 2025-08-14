"""
One-shot script to forecast next-day NYC Subway riders using Darts + LightGBM.

Steps:
- Fetch latest data from NY State API
- Filter to Subway mode and build a daily TimeSeries
- Build calendar/holiday future covariates
- Train LightGBMModel and forecast the next day
- Append forecast to subway_predictions.csv in this folder

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

# Suppress benign warnings:
# - sklearn's feature-name warning from LightGBM's sklearn wrapper during predict
warnings.filterwarnings(
	"ignore",
	category=UserWarning,
	module=r"sklearn\.utils\.validation",
	message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
# - pkg_resources deprecation warning emitted by transitive deps
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
	except Exception as e:  # pragma: no cover - guidance message
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
	"""Filter to Subway mode and return sorted, typed DataFrame with columns [date, count]."""
	if df.empty:
		raise ValueError("NY API returned no data")

	# Some rows may not have 'mode' or 'count'
	df = df.copy()
	if "mode" in df.columns:
		df = df[df["mode"].astype(str).str.contains("Subway", regex=False, na=False, case=False)]
	if "date" not in df.columns or "count" not in df.columns:
		missing = {c for c in ["date", "count"] if c not in df.columns}
		raise KeyError(f"Missing required columns in API response: {missing}")

	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df["count"] = pd.to_numeric(df["count"], errors="coerce")
	df = df.dropna(subset=["date", "count"])  # drop malformed
	df = df.sort_values("date")
	return df[["date", "count"]]


def build_series(df: pd.DataFrame):
	"""Build Darts TimeSeries (daily) from cleaned DataFrame.

	Returns
	-------
	series: darts.TimeSeries
		Target univariate series (float32)
	"""
	from darts import TimeSeries
	import numpy as np

	# Create continuous daily index and fill gaps
	daily = (
		df.set_index("date")["count"].asfreq("D").interpolate("time").ffill().bfill()
	)
	series = TimeSeries.from_series(daily, fill_missing_dates=True, freq="D").astype(
		"float32"
	)
	return series


def build_future_covariates(series) -> Any:
	"""Create calendar and holiday future covariates aligned to the target series span."""
	from darts import TimeSeries
	from darts.utils.timeseries_generation import datetime_attribute_timeseries
	import holidays as holidays
	import numpy as np
	import pandas as pd

	idx = pd.date_range(start=series.start_time(), end=series.end_time(), freq="D")
	# One-hot weekday (0-6) and month (1-12)
	dow = datetime_attribute_timeseries(idx, attribute="weekday", one_hot=True)
	moy = datetime_attribute_timeseries(idx, attribute="month", one_hot=True)

	# Weekend flag
	weekend_flag = pd.Series((idx.weekday >= 5).astype(np.int8), index=idx)
	is_weekend = TimeSeries.from_series(weekend_flag, freq="D")

	# US holiday flag
	us_holidays = holidays.country_holidays('US', subdiv='NY')
	holiday_flag = pd.Series(idx.map(lambda d: 1 if d in us_holidays else 0), index=idx)
	holidays_ts = TimeSeries.from_series(holiday_flag, freq="D")

	future_covariates = dow.stack(moy).stack(is_weekend).stack(holidays_ts)
	return future_covariates


def extend_future_covariates(series, days: int = 1) -> Any:
	"""Extend future covariates to cover 'days' steps after last observed date."""
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


def train_and_forecast(series) -> Tuple[pd.Timestamp, float]:
	"""Train LightGBMModel and forecast 1 day ahead.

	Returns the (next_date, forecast_value).
	"""
	from darts.dataprocessing.transformers import Scaler
	from darts.models import LightGBMModel

	# Build covariates
	future_covariates = build_future_covariates(series)

	# Scale target (trees don't need it, but keeps pipeline consistent)
	scaler = Scaler()
	series_scaled = scaler.fit_transform(series)

	# Model configuration
	lags = [-1, -2, -3, -4, -5, -6, -7, -14, -21]
	model = LightGBMModel(
		lags=lags,
		lags_future_covariates=[0],  # contemporaneous calendar/holiday features
		output_chunk_length=1,
		n_estimators=300,
		learning_rate=0.05,
		num_leaves=31,
		min_data_in_leaf=5,
		feature_fraction=0.8,
		random_state=42,
		verbosity=-1,
	)

	# Fit on full history
	model.fit(series_scaled, future_covariates=future_covariates)

	# Predict next day
	future_covariates_ext = extend_future_covariates(series, days=1)
	next_scaled = model.predict(n=1, future_covariates=future_covariates_ext)
	# Handle potential list returns defensively
	if isinstance(next_scaled, list):
		next_scaled = next_scaled[0]
	next_fcst_any = scaler.inverse_transform(next_scaled)
	if isinstance(next_fcst_any, list):
		next_fcst_any = next_fcst_any[0]
	# Cast for static type-checkers
	from darts import TimeSeries
	next_fcst = cast(TimeSeries, next_fcst_any)
	next_value = float(next_fcst.last_value())
	next_date = series.end_time() + timedelta(days=1)
	return next_date, next_value


def save_prediction(log_path: str, target_date: pd.Timestamp, value: float) -> None:
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	file_exists = os.path.isfile(log_path)
	now_iso = datetime.now().isoformat()
	import csv

	with open(log_path, "a", newline="") as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["target_date", "predicted", "predicted_at"])
		writer.writerow([target_date.strftime("%Y-%m-%d"), value, now_iso])


def main() -> int:
	ensure_deps()

	# 1) Pull and clean
	print("Fetching NY subway data…")
	raw_df = get_ny_data()
	ny_df = clean_data(raw_df)
	print(f"Rows after cleaning: {len(ny_df):,}")

	# 2) Build series
	series = build_series(ny_df)
	start_date = pd.Timestamp(series.start_time()).date()
	end_date = pd.Timestamp(series.end_time()).date()
	print(f"Series span: {start_date} → {end_date}  (n={len(series)})")

	# 3) Train + forecast
	print("Training LightGBMModel and forecasting next day…")
	next_date, next_value = train_and_forecast(series)
	print(f"Next-day forecast: {int(round(next_value))} on {next_date.date()}")

	# 4) Save to CSV (centralized under transportation/subway)
	script_dir = os.path.dirname(os.path.abspath(__file__))
	root_dir = os.path.dirname(script_dir)
	log_file = os.path.join(root_dir, "subway_predictions.csv")
	save_prediction(log_file, next_date, next_value)
	print(f"Logged prediction to: {log_file}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

