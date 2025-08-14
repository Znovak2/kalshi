# Kalshi ML Projects

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A collection of machine learning projects and trading algorithms for predictive modeling and market analysis.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Znovak2/kalshi.git
   cd kalshi
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
kalshi/
├── transportation/         # Transportation analytics
│   └── subway/            # NYC Subway ridership prediction
├── weather/               # Weather prediction models
├── tutorial/              # Learning materials and examples
├── data/                  # Datasets and data files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Available Projects

### 🚇 [NYC Subway Ridership Prediction](transportation/subway/)
Advanced 4-phase forecasting pipeline that predicts weekly average daily subway ridership using hybrid LightGBM models with day-of-week bias calibration.

**Features:**
- Real-time data integration via NYC Open Data API
- Hybrid iterative/multi-step LightGBM forecasting
- Automated bias calibration by day-of-week
- Historical backtesting and model evaluation
- Weekly averaging with actual data integration

**Quick Run:**
```bash
cd transportation/subway
python predictSubwayDailyAverage.py --verbose
```

### 🌤️ Weather Prediction Models
*[Coming Soon]* - Weather forecasting models and analysis tools.

### 📚 Tutorials
Example implementations and learning materials for getting started with the project frameworks.

## 🧰 Technology Stack

- **Core:** Python 3.7+, pandas, NumPy
- **ML Libraries:** scikit-learn, LightGBM, u8darts (TimeSeries)
- **Data Sources:** NYC Open Data API, various market data APIs
- **Visualization:** matplotlib, plotly
- **Development:** Jupyter notebooks, pytest

## 🔧 Development Setup

1. **Activate virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   # macOS/Linux  
   source .venv/bin/activate
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests** (when available)
   ```bash
   pytest tests/
   ```

## 📊 Performance & Results

Each project includes detailed methodology, backtesting results, and performance metrics. See individual project README files for specific results and benchmarks.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Tags

`machine-learning` `time-series` `forecasting` `transportation` `nyc-data` `lightgbm` `python` `data-science`
