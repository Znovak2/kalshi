"""
Analysis: per-horizon error distributions, hybrid experiments, bias calibration, and residual visualizations.

Reads `msae_detailed.csv` (produced by msae_test.py) and produces:
 - per-horizon summary and boxplot PNG
 - hybrid MAE table (for various switch horizons) and CSV
 - bias calibration results and CSV
 - residuals-over-time PNGs for iterative and multi modes

Run with the project's venv python. Requires pandas, numpy, matplotlib.
"""
from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
IN = ROOT / "phase2_data" / "msae_detailed.csv"
OUT_DIR = ROOT / "phase2_data"
IMG_DIR = ROOT.parent / "subway_methodology" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not IN.exists():
    print(f"Input file not found: {IN}. Run msae_test.py first.")
    raise SystemExit(1)

print("Loading detailed results...")
df = pd.read_csv(IN, parse_dates=["train_end", "target_date"]) 
# normalize

df["step"] = df["step"].astype(int)

# Per-horizon summary
per_h = df.groupby(["mode", "step"]).agg(
    count=("error", "count"),
    MAE=("abs_error", "mean"),
    MSE=("squared_error", "mean"),
    MeanError=("error", "mean"),
).reset_index()
print("Per-horizon summary (first rows):")
print(per_h.head(20))
per_h.to_csv(OUT_DIR / "per_horizon_summary.csv", index=False)

# Boxplot per-horizon (MAE) for each mode (use errors grouped by step)
modes = df["mode"].unique()
for mode in modes:
    d = df[df["mode"] == mode]
    grouped = [d[d["step"] == s]["error"].dropna() for s in sorted(d["step"].unique())]
    labels = [str(s) for s in sorted(d["step"].unique())]
    if not any(len(g) for g in grouped):
        continue
    plt.figure(figsize=(10, 5))
    plt.boxplot(grouped, tick_labels=labels, showfliers=False)
    plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    plt.title(f"Error distribution by step for mode={mode}")
    plt.xlabel("step")
    plt.ylabel("error (predicted - actual)")
    plt.tight_layout()
    out = IMG_DIR / f"boxplot_errors_{mode}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved boxplot: {out}")

# Prepare pivot with iterative and multi predictions for hybrid experiments
pivot = df.pivot_table(index=["train_end", "target_date", "step", "actual"], columns="mode", values=["predicted", "error"]) 
# flatten columns
pivot.columns = [f"{c[1]}_{c[0]}" for c in pivot.columns]
pivot = pivot.reset_index()
# keep rows where both iterative and multi predictions exist
if not ("iterative_predicted" in pivot.columns and "multi_predicted" in pivot.columns):
    print("Both iterative and multi predictions are required for hybrid experiments but not found.")
else:
    # Hybrid experiments: try switch horizons 1..6
    results = []
    for switch in range(1, 8):
        # choose iterative for step <= switch, else multi
        def choose(row):
            if row["step"] <= switch:
                return row["iterative_predicted"]
            else:
                return row["multi_predicted"]
        pivot["hybrid_pred"] = pivot.apply(choose, axis=1)
        pivot["hybrid_err"] = pivot["hybrid_pred"] - pivot["actual"]
        mae = pivot["hybrid_err"].abs().mean()
        mse = (pivot["hybrid_err"] ** 2).mean()
        me = pivot["hybrid_err"].mean()
        results.append({"switch": switch, "MAE": mae, "MSE": mse, "MeanError": me})
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "hybrid_results.csv", index=False)
    print("Hybrid results saved to hybrid_results.csv")
    print(res_df)

    # Try bias calibration for iterative: compute global mean error for iterative and apply
    iterative_mean_error = pivot["iterative_predicted"].sub(pivot["actual"]).mean()
    bias_adj = -iterative_mean_error  # to add to predictions to reduce bias
    print(f"Computed iterative mean error = {iterative_mean_error:.1f}, applying bias adjustment = {bias_adj:.1f}")
    pivot["iterative_calibrated"] = pivot["iterative_predicted"] + bias_adj
    # recompute hybrid with calibration (choose calibrated iterative)
    results_cal = []
    for switch in range(1, 8):
        def choose_cal(row):
            if row["step"] <= switch:
                return row["iterative_calibrated"]
            else:
                return row["multi_predicted"]
        pivot["hybrid_cal_pred"] = pivot.apply(choose_cal, axis=1)
        pivot["hybrid_cal_err"] = pivot["hybrid_cal_pred"] - pivot["actual"]
        mae = pivot["hybrid_cal_err"].abs().mean()
        mse = (pivot["hybrid_cal_err"] ** 2).mean()
        me = pivot["hybrid_cal_err"].mean()
        results_cal.append({"switch": switch, "MAE": mae, "MSE": mse, "MeanError": me})
    res_cal_df = pd.DataFrame(results_cal)
    res_cal_df.to_csv(OUT_DIR / "hybrid_results_calibrated.csv", index=False)
    print("Hybrid calibrated results saved to hybrid_results_calibrated.csv")
    print(res_cal_df)

# Residuals over time plots
for mode in ["iterative", "multi"]:
    if f"{mode}_predicted" not in pivot.columns:
        continue
    plt.figure(figsize=(12, 4))
    # sample to reduce overplotting
    p = pivot.copy()
    p = p.sort_values(["target_date"])
    plt.scatter(p["target_date"], p[f"{mode}_predicted"] - p["actual"], alpha=0.6, s=10)
    plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    plt.title(f"Residuals over time for mode={mode}")
    plt.xlabel("target_date")
    plt.ylabel("error (predicted - actual)")
    plt.tight_layout()
    out = IMG_DIR / f"residuals_time_{mode}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved residuals plot: {out}")

print("Done.")
