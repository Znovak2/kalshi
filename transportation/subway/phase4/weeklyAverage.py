"""
Weekly Average Ridership Calculator

This script calculates the daily average ridership from the most recent complete week
of predictions in subway_predictions.csv. Since predictions are always made to Sunday,
this provides a rolling 7-day average for the current forecast period.

The script:
1. Reads the latest predictions from subway_predictions.csv
2. Identifies the most recent Sunday and the preceding 7 days (Mon-Sun)
3. Calculates the daily average across those 7 days
4. Outputs the result with context about the calculation period

This metric is useful for:
- Understanding overall weekly ridership trends
- Comparing week-over-week changes
- Providing a single summary metric for the forecast period
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests


def get_actual_ridership_data(target_dates: list, timeout: int = 30) -> dict:
    """
    Fetch actual ridership data from NYC Open Data API for specified dates.
    
    Args:
        target_dates: List of date objects or date strings to fetch data for
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary mapping date strings to ridership counts
    """
    if not target_dates:
        return {}
    
    # Convert dates to strings in YYYY-MM-DD format
    date_strs = []
    for d in target_dates:
        if isinstance(d, str):
            date_strs.append(d)
        else:
            date_strs.append(d.strftime('%Y-%m-%d'))
    
    # Build SoQL query to get data for specific dates
    date_filter = " OR ".join([f"date = '{d}T00:00:00.000'" for d in date_strs])
    
    url = "https://data.ny.gov/resource/sayj-mze2.json"
    params = {
        "$select": "date,count",
        "$where": f"mode like 'Subway' AND ({date_filter})",
        "$order": "date",
        "$limit": "50"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        
        # Convert to dictionary mapping date -> ridership
        result = {}
        for record in data:
            date_str = record['date'][:10]  # Extract YYYY-MM-DD from timestamp
            ridership = float(record['count'])
            result[date_str] = ridership
            
        return result
        
    except Exception as e:
        print(f"Warning: Could not fetch actual ridership data: {e}")
        return {}


def get_complete_week_data(predictions_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Combine prediction data with actual data to create a complete week dataset.
    
    Args:
        predictions_df: DataFrame with prediction data (columns: target_date, predicted, predicted_at)
        
    Returns:
        Tuple of (complete_week_df, has_mixed_data)
        - complete_week_df: DataFrame with columns [date, ridership, data_type]
        - has_mixed_data: True if both actual and predicted data are used
    """
    if predictions_df.empty:
        raise ValueError("No prediction data provided")
    
    # Get the date range from predictions
    predictions_df['target_date'] = pd.to_datetime(predictions_df['target_date']).dt.date
    min_pred_date = predictions_df['target_date'].min()
    max_pred_date = predictions_df['target_date'].max()
    
    # Find the Monday of the week containing the prediction dates
    # We'll build a complete Mon-Sun week
    latest_date = max_pred_date
    days_since_monday = latest_date.weekday()  # Monday = 0, Sunday = 6
    
    # Find the Monday of this week
    monday_of_week = latest_date - timedelta(days=days_since_monday)
    
    # Build complete week (Mon-Sun)
    complete_week_dates = []
    for i in range(7):
        complete_week_dates.append(monday_of_week + timedelta(days=i))
    
    print(f"Building complete week: {monday_of_week} to {monday_of_week + timedelta(days=6)}")
    
    # Identify which dates need actual data vs predictions
    prediction_dates = set(predictions_df['target_date'].tolist())
    missing_dates = [d for d in complete_week_dates if d not in prediction_dates]
    
    result_data = []
    has_mixed_data = len(missing_dates) > 0
    
    # Add prediction data
    latest_predictions = get_latest_predictions(predictions_df)
    for _, row in latest_predictions.iterrows():
        result_data.append({
            'date': row['target_date'],
            'ridership': row['predicted'],
            'data_type': 'predicted'
        })
    
    # Fetch and add actual data for missing dates
    if missing_dates:
        print(f"Fetching actual data for {len(missing_dates)} missing dates: {missing_dates}")
        actual_data = get_actual_ridership_data(missing_dates)
        
        for missing_date in missing_dates:
            date_str = missing_date.strftime('%Y-%m-%d')
            if date_str in actual_data:
                result_data.append({
                    'date': missing_date,
                    'ridership': actual_data[date_str],
                    'data_type': 'actual'
                })
                print(f"  Found actual data for {date_str}: {actual_data[date_str]:,.0f} riders")
            else:
                print(f"  Warning: No actual data found for {date_str}")
    
    # Convert to DataFrame and sort by date
    complete_df = pd.DataFrame(result_data)
    if not complete_df.empty:
        complete_df = complete_df.sort_values('date').reset_index(drop=True)
    
    return complete_df, has_mixed_data


def find_most_recent_week(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Find the most recent complete week (Monday-Sunday) in the predictions.
    
    Args:
        df: DataFrame with target_date, predicted, predicted_at columns
        
    Returns:
        tuple of (week_data, monday_date, sunday_date)
        
    Raises:
        ValueError: If no complete week is found
    """
    # Get the most recent Sunday
    df_sorted = df.sort_values('target_date', ascending=False)
    
    # Find the most recent Sunday
    most_recent_sunday = None
    for _, row in df_sorted.iterrows():
        date = pd.Timestamp(row['target_date'])
        if date.weekday() == 6:  # Sunday = 6
            most_recent_sunday = date
            break
    
    if most_recent_sunday is None:
        raise ValueError("No Sunday found in predictions - cannot calculate weekly average")
    
    # Calculate the Monday of that week
    monday_of_week = most_recent_sunday - timedelta(days=6)
    
    # Filter for the complete week
    week_mask = (df['target_date'] >= monday_of_week.date()) & (df['target_date'] <= most_recent_sunday.date())
    week_data = df[week_mask].copy()
    
    # Check if we have all 7 days
    unique_dates = week_data['target_date'].nunique()
    if unique_dates < 7:
        raise ValueError(f"Incomplete week found: only {unique_dates} days available for week {monday_of_week.date()} to {most_recent_sunday.date()}")
    
    return week_data, monday_of_week, most_recent_sunday


def get_latest_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each target_date, get the most recent prediction (latest predicted_at).
    
    Args:
        df: DataFrame with target_date, predicted, predicted_at columns
        
    Returns:
        DataFrame with the latest prediction for each target date
    """
    df['predicted_at'] = pd.to_datetime(df['predicted_at'])
    
    # Get the most recent prediction for each target_date
    latest_predictions = df.sort_values('predicted_at').groupby('target_date').last().reset_index()
    
    return latest_predictions


def calculate_weekly_average(csv_path: str, verbose: bool = True, allow_partial: bool = False) -> dict:
    """
    Calculate the daily average ridership for a complete week using predictions and actual data.
    
    Args:
        csv_path: Path to subway_predictions.csv
        verbose: Whether to print detailed information
        allow_partial: If True, allow calculation on partial weeks (less than 7 days)
        
    Returns:
        Dictionary with calculation results and metadata
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")
    
    # Read predictions
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError("No predictions found in the CSV file")
    
    # Ensure proper data types
    df['target_date'] = pd.to_datetime(df['target_date']).dt.date
    
    if verbose:
        print(f"Loaded {len(df):,} prediction records")
        print(f"Date range: {df['target_date'].min()} to {df['target_date'].max()}")
    
    # Get complete week data (predictions + actual data)
    try:
        complete_week_df, has_mixed_data = get_complete_week_data(df)
        
        if complete_week_df.empty:
            raise ValueError("No complete week data could be assembled")
        
        num_days = len(complete_week_df)
        is_complete_week = (num_days == 7)
        
        if not is_complete_week and not allow_partial:
            raise ValueError(f"Incomplete week found: only {num_days} days available")
            
    except Exception as e:
        if not allow_partial:
            raise
        # Fallback to partial week with only predictions
        if verbose:
            print(f"Warning: {e}")
            print("Falling back to partial week calculation with available predictions...")
        
        latest_preds = get_latest_predictions(df)
        complete_week_df = pd.DataFrame({
            'date': latest_preds['target_date'],
            'ridership': latest_preds['predicted'],
            'data_type': 'predicted'
        }).sort_values('date').reset_index(drop=True)
        
        has_mixed_data = False
        is_complete_week = False
        num_days = len(complete_week_df)
    
    # Calculate daily average
    daily_average = complete_week_df['ridership'].mean()
    
    # Calculate some additional stats
    min_day = complete_week_df['ridership'].min()
    max_day = complete_week_df['ridership'].max()
    total_period = complete_week_df['ridership'].sum()
    
    # Day-by-day breakdown
    daily_breakdown = {}
    for _, row in complete_week_df.iterrows():
        date = pd.Timestamp(row['date'])
        day_name = date.strftime('%A')
        daily_breakdown[day_name] = {
            'date': row['date'],
            'ridership': int(round(row['ridership'])),
            'data_type': row['data_type']
        }
    
    # Determine period description
    start_date = complete_week_df['date'].min()
    end_date = complete_week_df['date'].max()
    
    results = {
        'daily_average': daily_average,
        'period_description': f"{start_date} to {end_date}",
        'start_date': start_date,
        'end_date': end_date,
        'total_period_ridership': total_period,
        'num_days': num_days,
        'is_complete_week': is_complete_week,
        'has_mixed_data': has_mixed_data,
        'min_daily': min_day,
        'max_daily': max_day,
        'daily_breakdown': daily_breakdown,
        'calculation_timestamp': datetime.now().isoformat()
    }
    
    if verbose:
        period_type = "Complete Week (7 days)" if is_complete_week else f"Partial Period ({num_days} days)"
        data_mix = "Mixed actual + predicted data" if has_mixed_data else "Prediction data only"
        
        print(f"\n=== {period_type} Ridership Summary ===")
        print(f"Period: {results['period_description']}")
        print(f"Data source: {data_mix}")
        print(f"Daily Average: {daily_average:,.0f} riders")
        print(f"Total Period: {total_period:,.0f} riders ({num_days} days)")
        print(f"Range: {min_day:,.0f} to {max_day:,.0f} riders")
        
        print(f"\nDaily Breakdown:")
        for day_name, info in daily_breakdown.items():
            data_source = f"({info['data_type']})"
            print(f"  {day_name:9} ({info['date']}) {data_source:>12}: {info['ridership']:,} riders")
    
    return results


def save_results(results: dict, output_path: str) -> None:
    """Save calculation results to a JSON file."""
    import json
    
    # Convert dates to strings for JSON serialization
    results_serializable = results.copy()
    results_serializable['start_date'] = str(results_serializable['start_date'])
    results_serializable['end_date'] = str(results_serializable['end_date'])
    
    # Convert daily breakdown dates
    for day_info in results_serializable['daily_breakdown'].values():
        day_info['date'] = str(day_info['date'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate weekly average ridership from predictions")
    parser.add_argument("--input", type=str, default=None, 
                       help="Path to subway_predictions.csv (default: ../subway_predictions.csv)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results JSON (default: phase4_data/weekly_average.json)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--allow-partial", action="store_true",
                       help="Allow calculation on partial weeks (less than 7 days)")
    parser.add_argument("--save-to-main", action="store_true",
                       help="Save summary CSV to main transportation/subway directory")
    
    args = parser.parse_args(argv)
    
    # Determine input path
    if args.input:
        csv_path = args.input
    else:
        # Default: look in phase3 directory
        script_dir = Path(__file__).parent
        csv_path = script_dir.parent / "phase3" / "subway_predictions.csv"
    
    csv_path = str(csv_path)
    
    # Determine output path  
    if args.output:
        output_path = args.output
    else:
        script_dir = Path(__file__).parent
        output_path = script_dir / "phase4_data" / "weekly_average.json"
    
    output_path = str(output_path)
    
    try:
        # Try complete week first, then allow partial if requested
        try:
            results = calculate_weekly_average(csv_path, verbose=not args.quiet, allow_partial=False)
        except ValueError:
            if args.allow_partial:
                if not args.quiet:
                    print("\nNo complete week available. Calculating with available days...\n")
                results = calculate_weekly_average(csv_path, verbose=not args.quiet, allow_partial=True)
            else:
                if not args.quiet:
                    print("No complete week available. Use --allow-partial to calculate with available days.")
                raise
        
        # Save results
        save_results(results, output_path)
        
        # Also save CSV summary
        summary_path = str(Path(output_path).parent / "weekly_summary.csv")
        period_type = "complete_week" if results['is_complete_week'] else "partial_period"
        data_mix = "mixed" if results['has_mixed_data'] else "predictions_only"
        summary_data = {
            'period': [results['period_description']],
            'period_type': [period_type],
            'data_mix': [data_mix],
            'num_days': [results['num_days']],
            'daily_average_ridership': [int(round(results['daily_average']))],
            'total_period_ridership': [int(round(results['total_period_ridership']))],
            'calculation_date': [datetime.now().date()]
        }
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)
        
        # Also save to main directory if requested
        if args.save_to_main:
            main_summary_path = str(Path(output_path).parent.parent.parent / "daily_average_subway_prediction.csv")
            pd.DataFrame(summary_data).to_csv(main_summary_path, index=False)
            if not args.quiet:
                print(f"Main summary saved to: {main_summary_path}")
        
        # Print key result
        if not args.quiet:
            period_desc = "complete week" if results['is_complete_week'] else f"partial period ({results['num_days']} days)"
            data_desc = "with mixed actual+predicted data" if results['has_mixed_data'] else "predictions only"
            print(f"\n>>> Key Result: {results['daily_average']:,.0f} average daily riders")
            print(f"   Period: {results['period_description']} ({period_desc})")
            print(f"   Data: {data_desc}")
            print(f"   Summary saved to: {summary_path}")
        
        return 0
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
