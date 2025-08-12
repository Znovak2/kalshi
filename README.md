# Subway Ridership Prediction

This project fetches New York subway ridership data from the NYC Open Data API, cleans and processes it, and trains a regression model to forecast the next day's total count.

## Features

- Downloads JSON data from `https://data.ny.gov/resource/sayj-mze2.json`
- Filters to subway entries and computes rolling statistics
- Builds lag-based features for supervised learning
- Trains a `RandomForestRegressor` to predict next-day ridership
- Logs predictions with timestamps to CSV for later accuracy tracking

## Prerequisites

- Python 3.7+
- Packages: `requests`, `pandas`, `scikit-learn`, `numpy`

Install dependencies via:

```bash
pip install requests pandas scikit-learn numpy
```

## Usage

1. Run the main script to fetch, clean, train, and predict:

   ```bash
   python subway.py
   ```

2. Check the console for the next-day forecast, MAE, and RMSE.

3. Review or archive predictions in `predictions_log.csv`.

## Project Structure

```
kalshi/
├── subway.py          # Main script: fetch, clean, model, predict
├── subway.ipynb       # Jupyter notebook with interactive exploration
└── predictions_log.csv# (auto-generated) logged forecasts
```

## License

MIT License. See `LICENSE` for details.
