# 🚇 NYC Subway Ridership Prediction

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://python.org/downloads/)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)]()
[![Last Updated](https://img.shields.io/badge/updated-August%202025-blue.svg)]()

Advanced 4-phase machine learning pipeline for forecasting NYC subway ridership with hybrid LightGBM models, automated bias calibration, and real-time data integration.

## 🚀 Quick Start

```bash
# Activate virtual environment (from repository root)
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Navigate to subway directory
cd transportation/subway

# Run the complete workflow
python predictSubwayDailyAverage.py --verbose
```

**Output:** `daily_average_subway_prediction.csv` with weekly average ridership prediction.

## 🏗️ Architecture Overview

### 4-Phase Workflow

1. **Phase 1: Hybrid Prediction Generation**
   - Iterative (day-by-day) LightGBM forecasting with bias calibration
   - Multi-step LightGBM forecasting for longer horizons
   - Automated optimal switch point determination

2. **Phase 2: Model Evaluation & Calibration**
   - Historical backtesting across multiple train/test splits
   - Day-of-week specific bias calculation
   - Performance analysis and visualization

3. **Phase 3: Hybrid Strategy Application**
   - Combines iterative + multi-step predictions optimally
   - Applies day-of-week bias corrections automatically

4. **Phase 4: Weekly Averaging**
   - Integrates actual ridership data for partial weeks
   - Calculates 7-day rolling averages
   - Outputs final prediction summary

### Key Features

- **🔄 Real-time Data:** Automatic integration via NYC Open Data API
- **🧠 Hybrid Models:** Combines iterative and multi-step LightGBM approaches
- **📊 Smart Calibration:** Day-of-week specific bias corrections
- **⚡ Auto-optimization:** Determines optimal model switching automatically
- **📈 Comprehensive Evaluation:** Backtesting with MAE, MSE, and residual analysis

## 📁 Project Structure

```
transportation/subway/
├── predictSubwayDailyAverage.py    # Master orchestration script
├── daily_average_subway_prediction.csv  # Final output
├── README.md                       # This documentation
│
├── phase1/                         # [Legacy - kept for reference]
│   ├── predictionByDay_A.py       # Single-step LightGBM
│   └── predictionByDay_B.py       # Multi-step LightGBM
│
├── phase2/                         # Model evaluation & calibration
│   ├── msae_test.py               # Backtesting framework
│   ├── msae_analysis.py           # Performance analysis & visualization
│   └── phase2_data/               # Generated evaluation data
│       ├── msae_detailed.csv     # Detailed prediction results
│       └── hybrid_results*.csv   # Hybrid strategy analysis
│
├── phase3/                         # Production prediction
│   ├── hybridPrediction.py       # Main hybrid forecasting script
│   └── subway_predictions.csv    # Generated predictions log
│
├── phase4/                         # Weekly averaging & output
│   ├── weeklyAverage.py          # Weekly calculation with real data
│   └── phase4_data/              # Generated summaries
│       ├── weekly_average.json   # Detailed breakdown
│       └── weekly_summary.csv    # Summary statistics
│
├── subway_methodology/            # Documentation & visualizations
│   ├── images/                   # Generated charts and plots
│   └── documents/                # Methodology documentation
```

## ⚙️ Configuration & Usage

### Basic Usage

```bash
# Run complete workflow with detailed logging
python predictSubwayDailyAverage.py --verbose

# Skip specific phases (if data already exists)
python predictSubwayDailyAverage.py --skip-phases 1,2

# Run individual components
python phase3/hybridPrediction.py --switch 3 --bias 50000
python phase4/weeklyAverage.py --save-main
```

### Advanced Options

**hybridPrediction.py arguments:**
- `--switch INT`: Override automatic switch determination (default: auto-optimal)
- `--bias FLOAT`: Override day-of-week bias with global adjustment (default: auto-calibrated)

**weeklyAverage.py arguments:**
- `--save-main`: Copy final result to main directory
- `--horizon DAYS`: Specify forecast horizon (default: to next Sunday)

## 📊 Model Performance

### Historical Backtest Results (Latest)

| Model Type | Step 1 MAE | Step 2 MAE | Step 3 MAE | Overall MAE |
|------------|------------|------------|------------|-------------|
| **Iterative** | 56,794 | 49,554 | 51,025 | 56,923 |
| **Multi-step** | 56,753 | 79,774 | 94,468 | 87,635 |
| **Hybrid (calibrated)** | 82,605 | 75,672 | 63,487 | **48,595** |

### Bias Calibration by Day-of-Week

| Day | Bias Adjustment | Interpretation |
|-----|-----------------|----------------|
| Monday | +165,585 | Historically under-predicted |
| Tuesday | +107,347 | Moderate under-prediction |
| Wednesday | +30,954 | Slight under-prediction |
| Thursday | +23,021 | Slight under-prediction |
| Friday | +78,174 | Moderate under-prediction |
| Saturday | +35,202 | Slight under-prediction |
| Sunday | +49,909 | Moderate under-prediction |

## 🔧 Dependencies

**Core Requirements:**
```
pandas>=2.0.0
numpy>=1.20.0
requests>=2.25.0
u8darts>=0.30.0
lightgbm>=4.0.0
holidays>=0.20
```

**Additional (auto-installed via requirements.txt):**
- `scikit-learn` - Data preprocessing and metrics
- `matplotlib` - Visualization
- `tqdm` - Progress bars

## 🔍 Data Sources

- **Primary:** [NYC Open Data - MTA Daily Ridership](https://data.ny.gov/resource/sayj-mze2.json)
- **Features:** Date, ridership count, weekday/weekend flags, holidays (US/NY)
- **Update Frequency:** Daily (typically T+1 day lag)
- **Coverage:** March 2020 - Present (~1,991 records as of August 2025)

## 🧪 Testing & Validation

### Smoke Test
```bash
# Verify complete workflow
python predictSubwayDailyAverage.py --verbose
# Expected: Success message + daily_average_subway_prediction.csv created
```

### Validation Checks
- ✅ Data freshness (API connectivity)
- ✅ Model convergence (training metrics)
- ✅ Prediction reasonableness (range validation)
- ✅ File outputs (CSV/JSON integrity)

## 🚨 Troubleshooting

### Common Issues

**1. API Connection Errors**
```
ERROR: requests.exceptions.RequestException
```
- **Solution:** Check internet connectivity, verify NYC Open Data API status

**2. Missing Dependencies**
```
ModuleNotFoundError: No module named 'darts'
```
- **Solution:** Ensure virtual environment is activated, run `pip install -r requirements.txt`

**3. Prediction Range Warnings**
```
WARNING: Predicted value outside expected range
```
- **Solution:** Review input data quality, check for anomalies in recent ridership

### Performance Optimization

- **Memory:** Large datasets may require chunking for backtesting
- **Speed:** Consider reducing `n_estimators` for faster prototyping
- **Accuracy:** Increase lookback window during low-variance periods

## 🏆 Model Methodology

### Feature Engineering
- **Temporal:** Weekday, month, weekend/holiday flags
- **Lags:** 1-7, 14, 21 day lookbacks
- **Moving averages:** Short and long-term trends

### Hybrid Strategy
1. **Iterative Model:** Day-by-day predictions with compounding
2. **Multi-step Model:** Direct multi-horizon forecasting
3. **Switch Point:** Automatically determined via cross-validation
4. **Bias Correction:** Day-of-week specific adjustments

### Evaluation Metrics
- **MAE:** Mean Absolute Error (primary metric)
- **MSE:** Mean Squared Error (outlier sensitivity)
- **MSAE:** Mean Signed Absolute Error (bias detection)

## 🔮 Future Enhancements

- [ ] **Advanced features:** Weather, events, economic indicators
- [ ] **Model ensemble:** Multiple algorithm combination
- [ ] **Seasonal decomposition:** Trend/seasonal component modeling
- [ ] **Confidence intervals:** Uncertainty quantification

## 📄 License & Attribution

This project is part of the Kalshi ML Projects suite. See the main repository for license details.

**Data Attribution:** NYC Open Data / Metropolitan Transportation Authority