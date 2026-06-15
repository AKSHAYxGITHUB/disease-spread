"""
Data Loader — Disease Spread Prediction System
Final Version: Data validation + 3-day rolling average smoothing.
Rolling average applied in _validate_series() before model input.
"""

import os
import pandas as pd
import numpy as np

_DF_CACHE = None
_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset.csv')


def _validate_series(series: pd.Series, disease: str = None) -> pd.Series:
    """
    Data Validation & Smoothing Layer:
    1. Clips extreme outliers (> 3x recent mean)
    2. Smooths single-day spikes (current > 2x previous)
    3. Applies 3-day rolling average to reduce noise (Faculty requirement)
    4. Applies domain-specific caps (COVID < 50, Dengue < 500, Malaria < 400)

    Does NOT touch dataset.csv — operates on in-memory series only.
    """
    if series.empty:
        return series

    series = series.copy().astype(float)
    
    # --- 1. Domain Constraints (New) ---
    if disease == "COVID-19":
        # Realistic cap for post-2023 pandemic phase
        series = series.clip(upper=50)
    elif disease == "Dengue":
        series = series.clip(upper=500)
    elif disease == "Malaria":
        series = series.clip(upper=400)

    # --- 2. Clip extreme outliers relative to recent 30-day context ---
    recent_mean = series.tail(30).mean()
    if recent_mean > 0:
        cap = recent_mean * 3
        series = series.clip(upper=cap)

    # --- 3. Smooth single-day spikes (Fixed logic) ---
    values = series.values.copy()
    for i in range(2, len(values)):
        prev = values[i - 1]
        prev2 = values[i - 2]
        if prev > 0 and values[i] > 2 * prev:
            # Replace sudden spike with average of previous two days
            values[i] = (prev + prev2) / 2
    series[:] = values

    # --- 4. Apply 3-day rolling average (removes spikes, stabilizes predictions) ---
    series = series.rolling(window=3, min_periods=1).mean()

    return np.round(series)


def load_data(force_reload=False):
    """Load and preprocess the dataset with in-memory caching and real-time synthesis."""
    global _DF_CACHE

    if _DF_CACHE is not None and not force_reload:
        return _DF_CACHE

    if not os.path.exists(_DATASET_PATH):
        raise FileNotFoundError(
            f"dataset.csv not found at {_DATASET_PATH}. "
            "Run data_generator.py first."
        )

    # Parse dates at read time and downcast the numeric columns to keep the
    # in-memory footprint small (important on low-RAM hosting).
    df = pd.read_csv(
        _DATASET_PATH,
        parse_dates=['Date'],
        dtype={'Temperature': 'float32', 'Humidity': 'float32', 'Rainfall': 'float32'},
    )
    df = df.sort_values('Date').reset_index(drop=True)

    # --- Real-time synthesis: Ensure data up to today ---
    latest_date = df['Date'].max()
    today = pd.to_datetime(pd.Timestamp.now().date())
    
    if latest_date < today:
        print(f"Dataset lagging ({latest_date.date()}). Synthesizing up to {today.date()}...")
        new_rows = []
        diff_days = (today - latest_date).days
        
        # Get all disease/region pairs
        groups = df[['Disease', 'Region']].drop_duplicates().values
        
        for d, r in groups:
            last_cases = df[(df['Disease'] == d) & (df['Region'] == r)]['Cases'].iloc[-1]
            last_temp = df['Temperature'].iloc[-1]
            last_hum = df['Humidity'].iloc[-1]
            last_rain = df['Rainfall'].iloc[-1]
            
            for i in range(1, diff_days + 1):
                new_date = latest_date + pd.Timedelta(days=i)
                # Add slight random variation to last known cases
                variation = np.random.randint(-2, 3)
                new_cases = max(1, int(last_cases + variation))
                
                new_rows.append({
                    'Date': new_date,
                    'Disease': d,
                    'Region': r,
                    'Cases': new_cases,
                    'Temperature': last_temp,
                    'Humidity': last_hum,
                    'Rainfall': last_rain
                })
        
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df = df.sort_values('Date').reset_index(drop=True)

    # Forward-fill missing values per group
    df['Cases'] = df.groupby(['Disease', 'Region'])['Cases'].transform(
        lambda x: x.ffill().bfill()
    )
    df['Temperature'] = df['Temperature'].ffill().bfill()
    df['Humidity'] = df['Humidity'].ffill().bfill()
    df['Rainfall'] = df['Rainfall'].ffill().bfill()

    _DF_CACHE = df
    return df


def get_series(disease: str, region: str) -> pd.Series:
    """Return validated + smoothed daily Cases time-series for disease+region."""
    df = load_data()
    mask = (df['Disease'] == disease) & (df['Region'] == region)
    series = df.loc[mask].set_index('Date')['Cases'].astype(float)
    series = series.sort_index()
    return _validate_series(series, disease=disease)   # Smoothing + constraints applied here


def get_current_date():
    """Return the absolute latest date in the dataset."""
    df = load_data()
    return df['Date'].max()


def get_latest_row(disease: str, region: str) -> dict:
    """Return the most recent data row for a disease+region pair."""
    df = load_data()
    mask = (df['Disease'] == disease) & (df['Region'] == region)
    row = df.loc[mask].iloc[-1]
    return row.to_dict()


def get_all_diseases() -> list:
    return ['Flu', 'Dengue', 'COVID-19', 'Malaria', 'Tuberculosis']


def get_all_regions() -> list:
    df = load_data()
    return sorted(df['Region'].unique().tolist())


def get_summary_stats() -> dict:
    """Aggregate stats for the dashboard — uses 7-day average for hotspot."""
    df = load_data()
    latest_date = df['Date'].max()
    week_ago = latest_date - pd.Timedelta(days=7)
    recent = df[df['Date'] >= week_ago]

    total_cases_week = int(recent['Cases'].sum())

    hotspot = (
        recent.groupby('Region')['Cases'].mean().idxmax()
    )

    disease_weekly = (
        recent.groupby('Disease')['Cases'].sum().to_dict()
    )

    return {
        'total_cases_week': total_cases_week,
        'hotspot_region': hotspot,
        'disease_weekly': disease_weekly,
        'latest_date': latest_date.strftime('%d %b %Y'),
    }