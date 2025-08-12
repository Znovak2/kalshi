"""
NYC Subway Ridership Validation

- Reuses NycSubwayRiders' fetch and filtering logic.
- Computes 7-day average daily riders for specific historical Mon–Sun windows.
- Compares results to provided market "closed" values and reports differences.

Python: 3.12
Allowed third-party libraries: pandas, numpy, requests
"""

from __future__ import annotations

import unittest
import pandas as pd
import numpy as np

# Reuse the exact logic from the primary script to ensure consistent filtering/aggregation.
from transportation.NycSubwayRiders_GPT import fetch_mta_data, to_subway_dataframe


# Historical targets (average daily riders, Mon–Sun)
WEEKS = [
    {
        "label": "2025-07-21 to 2025-07-27",
        "start": "2025-07-21",
        "end": "2025-07-27",
        "target_avg": 3_477_227,
    },
    {
        "label": "2022-05-16 to 2022-05-22",
        "start": "2022-05-16",
        "end": "2022-05-22",
        "target_avg": 2_946_947,
    },
    {
        "label": "2022-05-09 to 2022-05-15",
        "start": "2022-05-09",
        "end": "2022-05-15",
        "target_avg": 2_940_538,
    },
]


def weekly_average(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float | None, int, int]:
    """
    Compute the mean daily riders for [start, end], inclusive, using available days.

    Parameters
    ----------
    df : pd.DataFrame
        Subway-only DataFrame with columns ['date', 'count'] at daily granularity.
    start : pd.Timestamp
        Window start date (inclusive).
    end : pd.Timestamp
        Window end date (inclusive).

    Returns
    -------
    tuple
        (mean, n_present, n_expected)
        - mean: float or None if no data in window
        - n_present: number of days present in df within the window
        - n_expected: expected number of calendar days in window
    """
    if df.empty:
        return None, 0, 0

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        return None, 0, 0

    expected_dates = pd.date_range(start=start, end=end, freq="D")
    n_expected = len(expected_dates)

    window = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if window.empty:
        return None, 0, n_expected

    # Ensure at most one row per date; df should already be aggregated but guard anyway
    # Normalize date into a column and group by that column to avoid FutureWarning and KeyError
    window["date"] = window["date"].dt.normalize()
    window = window.groupby("date", as_index=False, sort=True)["count"].sum()
    n_present = window["date"].nunique()

    mean_val = float(window["count"].mean()) if n_present > 0 else None
    return mean_val, int(n_present), int(n_expected)


def validate_against_targets(df: pd.DataFrame, weeks: list[dict], tolerance_pct: float = 0.10) -> pd.DataFrame:
    """
    Compare computed weekly averages to target values.

    Parameters
    ----------
    df : pd.DataFrame
        Subway-only daily DataFrame.
    weeks : list[dict]
        Each dict has 'label', 'start', 'end', 'target_avg'.
    tolerance_pct : float
        Allowed relative difference to consider a validation 'pass'.

    Returns
    -------
    pd.DataFrame
        Columns: ['label','start','end','target_avg','computed_avg','abs_diff','pct_diff','days_present','days_expected','pass']
    """
    rows = []
    for w in weeks:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w["end"])
        target = float(w["target_avg"])
        computed, n_present, n_expected = weekly_average(df, start, end)
        if computed is None:
            abs_diff = np.nan
            pct_diff = np.nan
            passed = False
        else:
            abs_diff = abs(computed - target)
            denom = target if target != 0 else 1.0
            pct_diff = abs_diff / denom
            # Require most days present and within tolerance to pass
            passed = (n_present >= min(7, n_expected) - 1) and (pct_diff <= tolerance_pct)

        rows.append(
            {
                "label": w["label"],
                "start": start.date(),
                "end": end.date(),
                "target_avg": target,
                "computed_avg": computed,
                "abs_diff": abs_diff,
                "pct_diff": pct_diff,
                "days_present": n_present,
                "days_expected": n_expected,
                "pass": passed,
            }
        )
    return pd.DataFrame(rows)


# ---------- Safe formatting helpers ----------
def _fmt_num(x: float | int | None, decimals: int = 0) -> str:
    """Format optional numeric value with thousands separator; return 'None' for None/NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "None"
    return f"{float(x):,.{decimals}f}"


def _fmt_pct(x: float | None) -> str:
    """Format optional fraction as percent with 2 decimals; return 'None' for None/NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "None"
    return f"{float(x) * 100:.2f}%"


def main() -> None:
    """
    Pulls data, filters to subway-only, validates against historical week targets, and prints a summary.
    """
    records = fetch_mta_data()
    df = to_subway_dataframe(records)
    # Keep only date and count columns
    df = df[["date", "count"]].copy() if not df.empty else df

    results = validate_against_targets(df, WEEKS, tolerance_pct=0.10)

    print("Validation of 7-day average daily subway ridership (Mon–Sun):")
    if results.empty:
        print("- No data available to validate.")
        return

    for _, r in results.iterrows():
        target_s = _fmt_num(r["target_avg"], 0)
        computed_s = _fmt_num(r["computed_avg"], 0)
        diff_s = _fmt_num(r["abs_diff"], 0)
        pct_s = _fmt_pct(r["pct_diff"])
        coverage = f"{int(r['days_present'])}/{int(r['days_expected'])}"
        print(f"- {r['label']}: target={target_s}, computed={computed_s} (diff={diff_s}, {pct_s}); coverage={coverage}; pass={bool(r['pass'])}")


# ------------------------------ Tests ------------------------------


class TestValidation(unittest.TestCase):
    """Unit tests for aggregation and validation helpers."""

    def test_weekly_average_exact(self):
        # Synthetic complete week with known mean
        dates = pd.date_range("2022-05-09", periods=7, freq="D")  # Mon..Sun
        counts = np.arange(1, 8)  # mean = 4
        df = pd.DataFrame({"date": dates, "mode": "Subways", "count": counts})
        mean_val, n_present, n_expected = weekly_average(df, pd.Timestamp("2022-05-09"), pd.Timestamp("2022-05-15"))
        self.assertEqual(n_present, 7)
        self.assertEqual(n_expected, 7)
        self.assertAlmostEqual(mean_val, 4.0, places=6)

    def test_weekly_average_partial(self):
        # Missing two days -> still returns mean of available days
        dates = pd.to_datetime(["2022-05-09", "2022-05-10", "2022-05-12", "2022-05-13", "2022-05-15"])
        df = pd.DataFrame({"date": dates, "mode": "Subways", "count": [100, 110, 120, 130, 150]})
        mean_val, n_present, n_expected = weekly_average(df, pd.Timestamp("2022-05-09"), pd.Timestamp("2022-05-15"))
        self.assertEqual(n_expected, 7)
        self.assertEqual(n_present, 5)
        self.assertAlmostEqual(mean_val, np.mean([100, 110, 120, 130, 150]), places=6)

    def test_validate_structure(self):
        # With empty df we still get rows for each target and pass=False
        empty = pd.DataFrame(columns=["date", "count"])
        res = validate_against_targets(empty, WEEKS, tolerance_pct=0.10)
        self.assertEqual(len(res), len(WEEKS))
        self.assertIn("computed_avg", res.columns)
        self.assertFalse(res["pass"].any())


if __name__ == "__main__":
    # Run validation
    main()

    # To run tests:
    # python -m unittest c:\Users\znova\OneDrive\Documents\LaptopCodespaces\kalshi\transportation\NycSubwayRidersVal.py
    # unittest.main()
