# Subway Foot Traffic Forecasting Methodology & Results

## Overview
This report documents the methodology, datasets, and findings from modeling daily subway foot traffic, where the forecast horizon is dynamically set to end on **Sunday** each week. The model is run daily, and the objective is to predict the remaining days of the week using three approaches:

1. **Single-step prediction** – forecasts one day ahead at a time.
2. **Iterative prediction** – forecasts one day ahead, then feeds predictions back into the model to forecast subsequent days.
3. **Multi-step prediction** – forecasts all remaining days in the week in a single run.

---

## Data Sources

- **Detailed Row-Level Results**  
  [`msae_detailed.csv`](../phase%202/msae_detailed.csv) — Per-run, per-step predictions including `train_end`, `mode`, `target_date`, `step`, `predicted`, `actual`, `error`, `abs_error`, and `squared_error`.

- **Per-Horizon Summary**  
  [`per_horizon_summary.csv`](../phase%202/per_horizon_summary.csv) — Aggregated statistics (MAE, MSE, MeanError) for each mode and forecast step.

- **Hybrid Strategy Results**  
  [`hybrid_results.csv`](../phase%203/hybrid_results.csv) — MAE/MSE/MeanError for switch points (no bias calibration).  
  [`hybrid_results_calibrated.csv`](../phase%203/hybrid_results_calibrated.csv) — same metrics after bias calibration.

---

## Error Distribution Analysis

### Multi-Step
![Multi-Step Error Distribution](../phase%202/boxplot_errors_multi.png)

- **Observation:** Highest variance and bias of all methods.  
- Median errors are consistently negative (underprediction), with wide interquartile ranges in early-week steps.
- Variance remains high even toward the week’s end.

---

### Single-Step
![Single-Step Error Distribution](../phase%202/boxplot_errors_single.png)

- **Observation:** Lowest bias (~50k) and relatively low variance for one-day forecasts.
- Suitable for short-term operational decisions, but requires re-running daily to build a full week’s view.

---

### Iterative
![Iterative Error Distribution](../phase%202/boxplot_errors_iterative.png)

- **Observation:** Slightly higher bias than single-step (~61k), but lower variance over a full horizon.
- Error accumulation is present but controlled, with tightening error spread approaching Sunday.

---

## Temporal Residual Patterns

- **Iterative Residuals**  
  ![Residuals Over Time — Iterative](../phase%202/residuals_time_iterative.png)

- **Multi-Step Residuals**  
  ![Residuals Over Time — Multi](../phase%202/residuals_time_multi.png)

Residual patterns confirm that:
- Multi-step bias is persistent across the evaluation window.
- Iterative shows fluctuations but avoids runaway cumulative bias.

---

## Quantitative Comparison

From `per_horizon_summary.csv`:

| Mode       | Avg. MAE  | Avg. MSE (×10⁹) | Avg. MeanError | Avg. abs(MeanError) |
|------------|-----------|-----------------|----------------|---------------------|
| Single     | 50,901    | 5.48            | -49,865        | 49,865              |
| Iterative  | 56,220    | **4.88**        | -53,260        | 53,260              |
| Multi      | **95,472**| **10.37**       | -91,867        | 91,867              |

**Bias & MAE Trends Over Forecast Steps:**
<!-- If a combined trends image is added later, link it here. For now, refer to per-horizon summary CSV -->
See data in [`per_horizon_summary.csv`](../phase%202/per_horizon_summary.csv).

- **Lowest bias:** Single-step  
- **Lowest variance:** Iterative  
- **Worst performer:** Multi-step

---

## Hybrid Strategy Performance

### Without Calibration
See table in [`hybrid_results.csv`](../phase%203/hybrid_results.csv).

- MAE drops from ~87.6k at switch=1 to ~61.5k at switch=5.
- MeanError shrinks toward zero as switch point increases.

### With Bias Calibration
See table in [`hybrid_results_calibrated.csv`](../phase%203/hybrid_results_calibrated.csv).

- MAE drops more sharply, reaching ~42.0k at switch=5.
- MeanError is nearly eliminated, showing the effectiveness of bias adjustment.

---

## Conclusions

1. **Single-step**:  
   - Strength: Minimal bias for next-day forecasting.  
   - Limitation: Requires daily reruns to produce a week-long view.

2. **Iterative**:  
   - Strength: Balances bias and variance; best full-horizon stability.  
   - Limitation: Small bias accumulation over the week.

3. **Multi-step**:  
   - Strength: Produces entire week in one run.  
   - Limitation: High bias and large error spread.

4. **Hybrid + Calibration**:  
   - Strongest overall — combines iterative stability with calibration’s bias removal.  
   - MAE and variance significantly reduced.

---

## Recommended Next Steps
- Adopt **hybrid with calibration** for production forecasting.
- Monitor residual patterns weekly to detect drift.
- Investigate exogenous variable inclusion (weather, events) to further cut variance.

---
*Prepared by: Zach Novak*
