#!/usr/bin/env python3
"""
NYC "Tomorrow" Weather Dataset Builder

What it does:
1) Pulls historical daily observations for Central Park (USW00094728) via Meteostat.
2) Builds training features (lags/rolling/seasonality) and tomorrow targets (shift -1).
3) Fetches current NWS + Open-Meteo forecasts and aggregates them for *calendar* tomorrow.
4) Exports:
   - data/obs_daily_nyc.csv               (clean observations)
   - data/train_frame.parquet             (features + targets for model training)
   - data/tomorrow_features.parquet       (one row: features available to predict tomorrow)

Dependencies:
  pip install pandas numpy python-dateutil meteostat requests pyarrow
"""
from __future__ import annotations

import os
import math
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from meteostat import Daily

# ---------------------------- Config ---------------------------------
TZ = ZoneInfo("America/New_York")
LAT, LON = 40.7812, -73.9665  # Central Park-ish
GHCN_ID = "USW00094728"       # NYC Central Park station

OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)

# Required by NWS: identify your app + contact (replace placeholders)
NWS_HEADERS = {
    "User-Agent": "nyc-weather-ml (contact: animalcontroldude@example.com)",
    "Accept": "application/ld+json"
}

# ------------------------- Utilities ---------------------------------
def to_local_date(dt: datetime) -> pd.Timestamp:
    return pd.Timestamp(dt.astimezone(TZ).date())

def tomorrow_date(today: datetime | None = None) -> pd.Timestamp:
    now = datetime.now(TZ) if today is None else today.astimezone(TZ)
    tmr = (now + timedelta(days=1)).date()
    return pd.Timestamp(tmr)

def fahrenheit(c: pd.Series | float) -> pd.Series | float:
    return c * 9.0 / 5.0 + 32.0

def inches(mm: pd.Series | float) -> pd.Series | float:
    return mm / 25.4

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# -------------------- 1) Pull observations ---------------------------
def fetch_observations(start="2000-01-01", end=None) -> pd.DataFrame:
    end = end or datetime.now(TZ).date().isoformat()
    # Meteostat returns UTC-indexed DataFrame; columns: tmin,tmax,prcp,snow,tavg, etc (SI units)
    df = Daily(GHCN_ID, pd.to_datetime(start), pd.to_datetime(end)).fetch()
    if df.empty:
        raise RuntimeError("No observation data returned from Meteostat.")

    # Localize to NYC for clarity (dates are daily; set tz then convert)
    df = df.tz_localize("UTC").tz_convert(TZ)
    df.index = df.index.tz_localize(None)  # keep date-like index (naive, local day)

    # Rename + convert to US customary
    out = pd.DataFrame(index=df.index)
    out["TMAX_F"] = fahrenheit(df["tmax"])
    out["TMIN_F"] = fahrenheit(df["tmin"])
    out["PRCP_IN"] = inches(df["prcp"].fillna(0.0))  # missing -> 0 precip
    # you may keep SNOW if desired:
    out["SNOW_IN"] = inches(df.get("snow", pd.Series(index=df.index, dtype=float)).fillna(0.0))

    # Basic QC
    out = out.sort_index()
    return out

# --------------- 2) Build training features/targets ------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    doy = df.index.day_of_year
    df["DOY_SIN"] = np.sin(2 * np.pi * doy / 366.0)
    df["DOY_COS"] = np.cos(2 * np.pi * doy / 366.0)
    df["MONTH"] = df.index.month
    return df

def add_lags_rollups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["TMAX_F", "TMIN_F", "PRCP_IN"]:
        for lag in [1, 2, 3, 7, 14]:
            df[f"{col}_L{lag}"] = df[col].shift(lag)
        for win in [3, 7, 14]:
            df[f"{col}_MA{win}"] = df[col].rolling(win).mean()
            df[f"{col}_STD{win}"] = df[col].rolling(win).std()
    # Wet/dry streak (simple)
    wet = (df["PRCP_IN"] > 0).astype(int)
    df["WET_STREAK"] = wet.groupby((wet != wet.shift()).cumsum()).cumcount() + 1
    df["WET_STREAK"] = df["WET_STREAK"].where(wet == 1, 0)
    df["DRY_STREAK"] = ((wet == 0).astype(int).groupby(((wet == 0) != (wet == 0).shift()).cumsum())
                        .cumcount() + 1).where(wet == 0, 0)
    return df

def add_targets_tomorrow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TARGET_DATE"] = df.index.shift(1, freq="D")
    df["Y_TMAX_F"] = df["TMAX_F"].shift(-1)
    df["Y_TMIN_F"] = df["TMIN_F"].shift(-1)
    df["Y_RAIN_YN"] = (df["PRCP_IN"].shift(-1) > 0).astype(int)
    return df

def build_training_frame(obs: pd.DataFrame) -> pd.DataFrame:
    base = add_time_features(obs)
    base = add_lags_rollups(base)
    base = add_targets_tomorrow(base)
    # Keep rows where we have all needed features + targets
    feat_cols = [c for c in base.columns if c.startswith(("TMAX_F_", "TMIN_F_", "PRCP_IN_")) or
                 c in ["DOY_SIN", "DOY_COS", "MONTH", "WET_STREAK", "DRY_STREAK"]]
    targets = ["Y_TMAX_F", "Y_TMIN_F", "Y_RAIN_YN"]
    keep = feat_cols + targets + ["TARGET_DATE", "TMAX_F", "TMIN_F", "PRCP_IN", "SNOW_IN"]
    frame = base[keep].dropna().copy()
    # Set TARGET_DATE as the index (what we predict for)
    frame = frame.set_index("TARGET_DATE").sort_index()
    return frame

# ----------- 3) NWS gridpoints + hourly aggregation for tomorrow -----
@dataclass
class NWSGrid:
    office: str
    gridX: int
    gridY: int

def nws_get_grid(lat=LAT, lon=LON) -> NWSGrid:
    url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    r = requests.get(url, headers=NWS_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    props = data.get("properties", {})
    office = props["gridId"]
    gridX = int(props["gridX"])
    gridY = int(props["gridY"])
    return NWSGrid(office, gridX, gridY)

def nws_hourly_forecast(grid: NWSGrid) -> pd.DataFrame:
    url = f"https://api.weather.gov/gridpoints/{grid.office}/{grid.gridX},{grid.gridY}/forecast/hourly"
    r = requests.get(url, headers=NWS_HEADERS, timeout=20)
    r.raise_for_status()
    periods = r.json().get("properties", {}).get("periods", [])
    if not periods:
        return pd.DataFrame()
    rows = []
    for p in periods:
        # NWS timestamps are ISO with timezone, e.g., "2025-08-13T14:00:00-04:00"
        ts = pd.to_datetime(p["startTime"]).tz_convert(TZ)
        rows.append({
            "ts_local": ts,
            "temp_F": safe_float(p.get("temperature")),
            "pop_pct": safe_float((p.get("probabilityOfPrecipitation") or {}).get("value")),
            # Wind gust may be None
            "windGust_mph": safe_float((p.get("windGust") or {}).get("value")),
        })
    df = pd.DataFrame(rows).dropna(subset=["ts_local"]).set_index("ts_local").sort_index()
    # Normalize units (temp already in F from NWS; pop is 0-100)
    return df

def aggregate_nws_for_tomorrow(hourly: pd.DataFrame, tmr: pd.Timestamp) -> pd.Series:
    if hourly.empty:
        return pd.Series(dtype=float)
    start = pd.Timestamp(tmr.date(), tz=TZ)
    end = start + timedelta(days=1)
    day = hourly.loc[(hourly.index >= start) & (hourly.index < end)].copy()
    if day.empty:
        return pd.Series(dtype=float)
    return pd.Series({
        "NWS_TMAX_F": day["temp_F"].max(),
        "NWS_TMIN_F": day["temp_F"].min(),
        "NWS_POP_MAX": (day["pop_pct"].max() if "pop_pct" in day else np.nan),  # 0-100
        "NWS_WINDGUST_MAX": (day["windGust_mph"].max() if "windGust_mph" in day else np.nan),
    })

# ----------- 4) Open-Meteo daily forecast for tomorrow ---------------
def open_meteo_daily(lat=LAT, lon=LON) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_gusts_10m_max",
        ]),
        "timezone": "America/New_York",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    d = r.json().get("daily", {})
    if not d or "time" not in d:
        return pd.DataFrame()
    df = pd.DataFrame(d)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    # Rename + units
    df = df.rename(columns={
        "temperature_2m_max": "OM_TMAX_C",
        "temperature_2m_min": "OM_TMIN_C",
        "precipitation_sum": "OM_PRCP_MM",
        "precipitation_probability_max": "OM_POP_MAX",  # 0-100
        "wind_gusts_10m_max": "OM_WINDGUST_MS",        # m/s
    })
    # Convert to F / inches / mph
    out = pd.DataFrame(index=df.index)
    out["OM_TMAX_F"] = fahrenheit(df["OM_TMAX_C"])
    out["OM_TMIN_F"] = fahrenheit(df["OM_TMIN_C"])
    out["OM_PRCP_IN"] = inches(df["OM_PRCP_MM"])
    out["OM_POP_MAX"] = df["OM_POP_MAX"]
    out["OM_WINDGUST_MPH"] = df["OM_WINDGUST_MS"] * 2.23694
    return out

def build_tomorrow_features(train_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compose the feature row you'd feed your model *today* to predict *tomorrow*.
    Includes:
      - latest obs-based features (lags/rollups/seasonality) for "today"
      - NWS + Open-Meteo forecasts aggregated for *tomorrow* (calendar day)
    Index = TARGET_DATE (tomorrow)
    """
    if train_frame.empty:
        raise RuntimeError("Training frame is empty; build it first.")

    # Get tomorrow and today dates
    tmr = tomorrow_date()
    today = (tmr - pd.Timedelta(days=1))

    # Start from latest row in the training frame corresponding to TARGET_DATE == today
    if today not in train_frame.index:
        # fallback: take last available
        latest_idx = train_frame.index.max()
    else:
        latest_idx = today

    base = train_frame.loc[[latest_idx]].copy()
    base.index = [tmr]  # set row index to TARGET_DATE = tomorrow

    # Fetch NWS + OM forecasts and join
    try:
        grid = nws_get_grid(LAT, LON)
        nws_hr = nws_hourly_forecast(grid)
        s_nws = aggregate_nws_for_tomorrow(nws_hr, tmr)
    except Exception as e:
        print(f"[WARN] NWS fetch failed: {e}")
        s_nws = pd.Series(dtype=float)

    try:
        om = open_meteo_daily(LAT, LON)
        s_om = om.loc[tmr.strftime("%Y-%m-%d")] if tmr in om.index else pd.Series(dtype=float)
    except Exception as e:
        print(f"[WARN] Open-Meteo fetch failed: {e}")
        s_om = pd.Series(dtype=float)

    # Combine
    feat_row = base.copy()
    for src in (s_nws, s_om):
        for k, v in src.items():
            feat_row[k] = v

    return feat_row

# ------------------------------ Main ---------------------------------
def main():
    print("Fetching observations…")
    obs = fetch_observations(start="2000-01-01")
    obs.to_csv(os.path.join(OUTDIR, "obs_daily_nyc.csv"), index_label="date")
    print(f"Saved {len(obs):,} rows -> {OUTDIR}/obs_daily_nyc.csv")

    print("Building training frame…")
    train = build_training_frame(obs)
    # Persist (Parquet preserves dtypes nicely)
    train.to_parquet(os.path.join(OUTDIR, "train_frame.parquet"))
    print(f"Saved {len(train):,} rows -> {OUTDIR}/train_frame.parquet")

    print("Composing 'tomorrow' feature row…")
    tmr_feats = build_tomorrow_features(train)
    tmr_path = os.path.join(OUTDIR, "tomorrow_features.parquet")
    tmr_feats.to_parquet(tmr_path)
    print(f"Saved 1 row -> {tmr_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
