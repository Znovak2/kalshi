# Phase 4: Weekly Ridership Averaging

This phase calculates the daily average ridership from a complete week of data, automatically combining prediction data with actual historical data to provide robust weekly averages for business decision making.

## Script: `weeklyAverage.py`

**Intelligent Data Integration**: The script automatically:
1. **Identifies missing days** in the current week from prediction data
2. **Fetches actual ridership data** from NYC Open Data API for missing days
3. **Combines actual + predicted data** for a complete 7-day weekly average
4. **Provides transparency** about data sources (actual vs predicted per day)

This makes the system robust regardless of when during the week it's run or how many days have predictions.

### Usage

**Basic usage (automatically fetches missing actual data):**
```bash
python weeklyAverage.py
```

**Allow partial weeks (if API unavailable):**
```bash
python weeklyAverage.py --allow-partial
```

**Custom paths:**
```bash
python weeklyAverage.py --input path/to/predictions.csv --output path/to/results.json
```

**Quiet mode:**
```bash
python weeklyAverage.py --quiet
```

### Current Example Output

For the week of August 11-17, 2025:
- **Daily Average**: 3,424,967 riders
- **Data Mix**: 2 days actual + 5 days predicted
- **Complete Week**: Monday-Sunday (7 days)
- **Data Sources**: 
  - Monday-Tuesday: Actual data from NYC API
  - Wednesday-Sunday: Hybrid model predictions

### Outputs

The script generates two output files in the `phase4_data/` directory:

1. **`weekly_summary.csv`** - Business summary with:
   - Period dates and type (complete_week/partial_period)
   - Data mix indicator (mixed/predictions_only)
   - Number of days included
   - Daily average and total period ridership
   - Calculation date

2. **`weekly_average.json`** - Detailed breakdown including:
   - Daily average ridership
   - Complete week indicator and mixed data flag
   - Day-by-day breakdown with data source per day
   - Min/max daily ridership range
   - Calculation timestamp

### Robustness Features

✅ **Week Detection**: Automatically finds Monday-Sunday week containing predictions  
✅ **Missing Data Handling**: Fetches actual data from NYC API for missing days  
✅ **Data Source Transparency**: Labels each day as "actual" or "predicted"  
✅ **Fallback Mode**: Works with partial weeks if API unavailable  
✅ **Error Handling**: Graceful handling of API timeouts or missing data  

### Business Value

This system provides **consistent weekly averages** regardless of operational timing:

- **Monday Runs**: Uses all actual data from previous week
- **Wednesday Runs**: Combines Mon-Tue actual + Wed-Sun predictions  
- **Friday Runs**: Combines Mon-Thu actual + Fri-Sun predictions
- **Any Day**: Always provides complete weekly context for planning

### API Integration

**Data Source**: NYC Open Data API (`https://data.ny.gov/resource/sayj-mze2.json`)  
**Query Method**: Targeted date filtering with SoQL  
**Fallback**: Graceful degradation to prediction-only mode if API unavailable  
**Timeout**: 30-second request timeout with error handling  

This ensures **reliable weekly business metrics** with optimal data accuracy.
