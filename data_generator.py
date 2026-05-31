"""
Data Generator v4 — Web-Informed Epidemiological Dataset
=========================================================
Generates a realistic dataset (2015-01-01 → 2026-03-29) for 28 Indian states
and 5 diseases.  All patterns are informed by:
  - WHO / MOHFW epidemic reports
  - Published COVID-19 wave timelines for India
  - ICMR / NVBDCP malaria & dengue bulletins
  - CDC influenza seasonality studies
  - Central TB Division annual reports

Columns (FIXED): Date, Region, Disease, Cases, Temperature, Humidity, Rainfall
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DISEASES = ['Flu', 'Dengue', 'COVID-19', 'Malaria', 'Tuberculosis']

REGIONS = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

# Population tiers → case multiplier
HIGH_POP = {
    'Uttar Pradesh': 2.5, 'Maharashtra': 2.2,
    'Bihar': 2.0,        'West Bengal': 1.9
}
MEDIUM_POP = {
    'Tamil Nadu': 1.7,    'Karnataka': 1.6,    'Gujarat': 1.5,
    'Rajasthan': 1.5,     'Andhra Pradesh': 1.5,'Madhya Pradesh': 1.4,
    'Telangana': 1.4,     'Odisha': 1.3,        'Haryana': 1.3,
    'Kerala': 1.3,        'Jharkhand': 1.2,    'Chhattisgarh': 1.2,
    'Assam': 1.1,         'Uttarakhand': 1.1,  'Punjab': 1.1,
    'Himachal Pradesh': 0.9
}
LOW_POP = {
    'Arunachal Pradesh': 0.5, 'Manipur': 0.5, 'Meghalaya': 0.5,
    'Mizoram': 0.4, 'Nagaland': 0.4, 'Sikkim': 0.3,
    'Tripura': 0.6, 'Goa': 0.4
}

def get_pop_factor(region):
    if region in HIGH_POP:   return HIGH_POP[region]
    if region in MEDIUM_POP: return MEDIUM_POP[region]
    if region in LOW_POP:    return LOW_POP[region]
    return 1.2

# Dengue high-risk states (coastal / tropical zones)
DENGUE_HIGH = {
    'Tamil Nadu', 'Kerala', 'Karnataka', 'Maharashtra',
    'Andhra Pradesh', 'Telangana', 'West Bengal', 'Gujarat'
}

# Malaria high-risk states (NVBDCP tribal / forested zones)
MALARIA_HIGH = {
    'Odisha', 'Chhattisgarh', 'Jharkhand', 'Madhya Pradesh',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Tripura', 'Arunachal Pradesh'
}

# COVID 2026 state-specific endemic caps (raw cases, NOT scaled by pop)
COVID_2026_CAP = {
    'Chhattisgarh': 8,   'Punjab': 3,         'Himachal Pradesh': 6,
    'Uttarakhand':  8,   'Assam': 6,          'Goa': 10,
    'Kerala': 40,        'Tamil Nadu': 35,    'Maharashtra': 45,
    'Karnataka': 30,     'Andhra Pradesh': 25,'Telangana': 20,
    'West Bengal': 18,   'Gujarat': 14,       'Madhya Pradesh': 12,
    'Rajasthan': 10,     'Bihar': 12,         'Odisha': 8,
    'Jharkhand': 6,      'Haryana': 8,        'Uttar Pradesh': 20,
}
COVID_DEFAULT_2026_CAP = 15   # most states


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def gauss_wave(days_arr, center_day, width, amplitude):
    """Gaussian epidemic wave centered on `center_day` (days from 2020-01-01)."""
    return amplitude * np.exp(-0.5 * ((days_arr - center_day) / width) ** 2)


def smooth_series(arr, window=3):
    """Rolling mean smoothing — keeps edges via min_periods=1."""
    s = pd.Series(arr.astype(float))
    return s.rolling(window, min_periods=1).mean().values


def apply_growth_cap(arr, max_ratio=2.0, allow_spike_year=None, year_arr=None):
    """Prevent any day from exceeding max_ratio × previous day."""
    out = arr.copy().astype(float)
    for i in range(1, len(out)):
        prev = out[i - 1]
        if prev > 2 and out[i] > prev * max_ratio:
            if allow_spike_year and year_arr is not None and year_arr[i] == allow_spike_year:
                pass   # allow delta or omicron surge
            else:
                out[i] = prev * max_ratio
    return out


def validate(arr, label=""):
    """Assert no negatives, return clipped non-negative array."""
    neg = np.sum(arr < 0)
    if neg > 0:
        print(f"  ⚠  {label}: {neg} negative values — clipping to 0")
    return np.clip(arr, 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_weather(dates, region):
    n = len(dates)
    yday  = np.array([d.timetuple().tm_yday for d in dates])
    month = np.array([d.month for d in dates])

    # Temperature — sinusoidal annual cycle
    temp = 27 + 8 * np.sin((yday - 90) * 2 * np.pi / 365)
    temp += np.random.normal(0, 1.0, n)

    # Regional adjustments
    if region in {'Himachal Pradesh', 'Uttarakhand', 'Sikkim', 'Arunachal Pradesh'}:
        temp -= 8
    elif region in {'Rajasthan', 'Gujarat', 'Telangana', 'Andhra Pradesh'}:
        temp += 3
    elif region in {'Kerala', 'Goa', 'Tamil Nadu'}:
        temp -= 1

    temp = np.clip(temp, 5, 45)

    # Rainfall — monsoon shape per month
    MONSOON = {6: 150, 7: 200, 8: 180, 9: 100, 10: 40, 11: 20}
    rain = np.array(
        [max(0, np.random.exponential(MONSOON.get(m, 5))) for m in month],
        dtype=float
    )

    if region in {'Kerala', 'Meghalaya', 'Assam', 'Nagaland', 'Mizoram', 'Tripura', 'Manipur'}:
        rain *= 1.6
    elif region in {'Rajasthan', 'Gujarat', 'Himachal Pradesh'}:
        rain *= 0.4
    elif region == 'Tamil Nadu':
        # Northeast monsoon: Oct-Dec heavy, summer dry
        northeast = np.isin(month, [10, 11, 12])
        rain[northeast] *= 2.0

    rain = np.clip(rain, 0, 350)

    # Humidity
    humidity = np.where(np.isin(month, [6, 7, 8, 9]), 82, 50).astype(float)
    humidity += (rain / 350.0) * 25
    humidity += np.random.normal(0, 4, n)
    if region in {'Kerala', 'Goa', 'Assam', 'Meghalaya'}:
        humidity += 8
    elif region in {'Rajasthan', 'Himachal Pradesh'}:
        humidity -= 10
    humidity = np.clip(humidity, 20, 98)

    return temp, humidity, rain


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def gen_covid(dates, pop_f, region, temp, humidity, rain, year_arr, month_arr):
    n = len(dates)
    d0 = date(2020, 1, 1)
    dfrom2020 = np.array([(d - d0).days for d in dates], dtype=float)

    # ── Wave structure (India-specific timeline) ──────────────────────────────
    # Wave 1: Sep 2020  (day ~255) — nationwide first wave
    w1 = gauss_wave(dfrom2020, center_day=255, width=55,  amplitude=80)
    # Wave 2: Delta — May 2021 (day ~490) — deadliest wave
    w2 = gauss_wave(dfrom2020, center_day=490, width=35,  amplitude=600)
    # Wave 3: Omicron — Jan 2022 (day ~738) — high but milder
    w3 = gauss_wave(dfrom2020, center_day=738, width=22,  amplitude=350)
    # Omicron sub-wave BA.2: ~Apr 2022 (day ~820)
    w4 = gauss_wave(dfrom2020, center_day=820, width=18,  amplitude=100)
    # JN.1 micro-wave: ~Jan 2024 (day ~1460)
    w5 = gauss_wave(dfrom2020, center_day=1460, width=14, amplitude=25)

    c = (w1 + w2 + w3 + w4 + w5) * pop_f

    # Pre-2020 = 0
    c[year_arr < 2020] = 0.0

    # Small proportional noise
    c += np.random.normal(0, np.abs(c) * 0.06 + 0.5, n)
    c  = np.clip(c, 0, None)

    # ── 2023–2026: strict endemic low ─────────────────────────────────────────
    cap = COVID_2026_CAP.get(region, COVID_DEFAULT_2026_CAP)

    mask_2023 = year_arr >= 2023
    if mask_2023.any():
        # Endemic base: gradually declining from 2023 → 2026
        years_past_2022 = np.clip(year_arr[mask_2023] - 2022, 0, 4)
        decay = 1.0 - 0.15 * years_past_2022          # gently shrink each year
        endemic_base = cap * decay * 0.4
        noise = np.random.normal(0, cap * 0.12, mask_2023.sum())
        endemic_base = np.clip(endemic_base + noise, 0, cap)
        c[mask_2023] = endemic_base

    # Ensure 2026 hard cap
    mask_2026 = year_arr >= 2026
    if mask_2026.any():
        c[mask_2026] = np.clip(c[mask_2026], 0, cap)

    # Growth cap (allow 2021 delta spike)
    c = apply_growth_cap(c, max_ratio=2.0, allow_spike_year=2021, year_arr=year_arr)
    c = smooth_series(c, window=3)
    return np.clip(c, 0, None)


def gen_dengue(dates, pop_f, region, temp, humidity, rain, year_arr, month_arr):
    n = len(dates)
    # High-risk states get a much stronger base
    bias = 3.5 if region in DENGUE_HIGH else 1.2
    # Base tuned so peak months hit 50–500 in high-risk states
    base = 55 * pop_f * bias

    # Month-based seasonal factor (NVBDCP pattern: peaks July–Sept)
    DENGUE_SEASON = {
        1: 0.03, 2: 0.03, 3: 0.04, 4: 0.07, 5: 0.12,
        6: 0.50, 7: 0.80, 8: 1.00, 9: 0.90, 10: 0.60,
        11: 0.22, 12: 0.08
    }
    seasonal = np.array([DENGUE_SEASON[m] for m in month_arr])

    # Weather modulation: rain + humidity drive vector breeding
    rain_norm = np.clip(rain / 200.0, 0, 1.5)
    hum_norm  = np.clip((humidity - 50) / 40.0, 0, 1)
    weather_f = 0.4 + 0.45 * rain_norm + 0.25 * hum_norm

    # Temporal growth trend (India's dengue cases rising YoY)
    year_trend = 1.0 + np.clip((year_arr - 2015) * 0.03, 0, 0.40)

    c = base * seasonal * weather_f * year_trend
    c += np.random.normal(0, c * 0.15 + 0.5, n)
    c  = np.clip(c, 0, None)

    c = apply_growth_cap(c, max_ratio=1.8)
    c = smooth_series(c, window=3)
    c = np.clip(c, 0, 500)
    return c


def gen_malaria(dates, pop_f, region, temp, humidity, rain, year_arr, month_arr):
    n = len(dates)
    # High-risk states (tribal / forest belt) get strong bias
    bias = 4.0 if region in MALARIA_HIGH else 0.8
    # Base tuned so peak months hit 20–400 in high-risk states
    base = 45 * pop_f * bias

    # Malaria season: Jun-Sep (NVBDCP: peaks Aug-Sep post-monsoon)
    MALARIA_SEASON = {
        1: 0.03, 2: 0.03, 3: 0.05, 4: 0.08, 5: 0.15,
        6: 0.55, 7: 0.80, 8: 1.00, 9: 0.90, 10: 0.50,
        11: 0.15, 12: 0.04
    }
    seasonal = np.array([MALARIA_SEASON[m] for m in month_arr])

    rain_norm = np.clip(rain / 180.0, 0, 1.5)
    temp_f    = np.clip((temp - 15) / 20.0, 0, 1.2)  # warmer helps breeding
    weather_f = 0.35 + 0.4 * rain_norm + 0.2 * temp_f

    # India's malaria elimination program → slight year-on-year decline
    year_trend = 1.0 - np.clip((year_arr - 2015) * 0.025, 0, 0.28)

    c = base * seasonal * weather_f * year_trend
    c += np.random.normal(0, c * 0.15 + 0.3, n)
    c  = np.clip(c, 0, None)

    c = apply_growth_cap(c, max_ratio=1.8)
    c = smooth_series(c, window=3)
    c = np.clip(c, 0, 400)
    return c


def gen_flu(dates, pop_f, region, temp, humidity, rain, year_arr, month_arr):
    n = len(dates)
    # Base tuned so Dec-Jan peak reaches 100–300 in high-pop states
    base = 75 * pop_f

    # Flu season: Dec-Feb peak (India follows northern hemisphere pattern)
    FLU_SEASON = {
        1: 0.90, 2: 0.82, 3: 0.48, 4: 0.32, 5: 0.28,
        6: 0.28, 7: 0.32, 8: 0.28, 9: 0.33, 10: 0.50,
        11: 0.70, 12: 1.00
    }
    seasonal = np.array([FLU_SEASON[m] for m in month_arr])

    # Temperature modulation: lower temp → higher flu transmission
    temp_norm = np.clip((35 - temp) / 20.0, 0, 1.5)
    weather_f = 0.55 + 0.45 * temp_norm

    c = base * seasonal * weather_f
    c += np.random.normal(0, c * 0.12 + 1.0, n)
    c  = np.clip(c, 0, None)

    # COVID years 2020-2021: flu suppressed due to masks / lockdowns
    mask_covid = np.isin(year_arr, [2020, 2021])
    c[mask_covid] *= 0.40

    c = apply_growth_cap(c, max_ratio=1.6)
    c = smooth_series(c, window=3)
    c = np.clip(c, 10, 300)
    return c


def gen_tb(dates, pop_f, region, temp, humidity, rain, year_arr, month_arr):
    n = len(dates)
    # TB is stable: 50–200 range, no seasonal spikes
    # Based on Central TB Division data: slight year-on-year declining trend
    base_tb = 90 * pop_f

    # Very slight declining trend due to "End TB" programme
    year_trend = 1.0 - np.clip((year_arr - 2015) * 0.01, 0, 0.10)

    c      = np.zeros(n, dtype=float)
    c[0]   = base_tb * year_trend[0]
    for i in range(1, n):
        target = base_tb * year_trend[i]
        step   = np.random.normal(0, 0.8)
        c[i]   = c[i - 1] * 0.97 + target * 0.03 + step

    c = np.clip(c, 30 * pop_f, 200 * pop_f)
    c = smooth_series(c, window=5)
    c = np.clip(c, 50, 200)
    return c


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

GENERATORS = {
    'COVID-19':     gen_covid,
    'Dengue':       gen_dengue,
    'Malaria':      gen_malaria,
    'Flu':          gen_flu,
    'Tuberculosis': gen_tb,
}

HARD_CAPS = {
    'COVID-19':     800,   # pre-endemic peak cap
    'Dengue':       500,
    'Malaria':      400,
    'Flu':          300,
    'Tuberculosis': 200,
}


def main():
    print("=" * 65)
    print("  Data Generator v4 — Web-Informed Epidemiological Dataset")
    print("=" * 65)
    np.random.seed(2026)

    start_date = date(2015, 1, 1)
    end_date   = date(2026, 3, 29)   # system date = March 29 2026
    dates = [
        start_date + timedelta(days=x)
        for x in range((end_date - start_date).days + 1)
    ]
    n_days = len(dates)

    year_arr  = np.array([d.year  for d in dates])
    month_arr = np.array([d.month for d in dates])

    total_rows = n_days * len(REGIONS) * len(DISEASES)
    print(f"  Generating {total_rows:,} rows for {len(REGIONS)} states × {len(DISEASES)} diseases")
    print(f"  Date range : {start_date} → {end_date}\n")

    rows = []

    for region in REGIONS:
        print(f"  Processing {region} …")
        pop_f = get_pop_factor(region)
        temp, hum, rain = make_weather(dates, region)

        for disease in DISEASES:
            gen_fn = GENERATORS[disease]
            raw    = gen_fn(dates, pop_f, region, temp, hum, rain, year_arr, month_arr)

            # ── Final validation ──────────────────────────────────────────
            raw = validate(raw, label=f"{disease}/{region}")

            # ── Hard caps per disease ─────────────────────────────────────
            raw = np.clip(raw, 0, HARD_CAPS[disease])

            # ── 2x growth cap (global pass) ───────────────────────────────
            raw = apply_growth_cap(
                raw, max_ratio=2.0,
                allow_spike_year=(2021 if disease == 'COVID-19' else None),
                year_arr=year_arr
            )

            # ── Anomaly smoother: remove spikes > 2× rolling window ───────
            raw = smooth_series(raw, window=3)

            # ── Final rounding to integer ─────────────────────────────────
            cases_int = np.abs(np.round(raw)).astype(int)

            for i in range(n_days):
                rows.append({
                    'Date':        dates[i].strftime('%Y-%m-%d'),
                    'Region':      region,
                    'Disease':     disease,
                    'Cases':       cases_int[i],
                    'Temperature': round(float(temp[i]), 1),
                    'Humidity':    round(float(hum[i]), 1),
                    'Rainfall':    round(float(rain[i]), 1),
                })

    df = pd.DataFrame(rows)

    # ── Column order — FIXED as required ─────────────────────────────────────
    df = df[['Date', 'Region', 'Disease', 'Cases',
             'Temperature', 'Humidity', 'Rainfall']]

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.csv')
    df.to_csv(outpath, index=False)

    # ── Validation report ─────────────────────────────────────────────────────
    print("\n─── Validation Report ───────────────────────────────────────")
    print(f"  Total rows   : {len(df):,}")
    print(f"  Date range   : {df['Date'].min()} → {df['Date'].max()}")
    print(f"  Regions      : {df['Region'].nunique()}")
    print(f"  Diseases     : {', '.join(df['Disease'].unique())}")
    print(f"  Negatives    : {(df['Cases'] < 0).sum()}")
    print(f"  Columns      : {list(df.columns)}")
    print()

    for dis in DISEASES:
        sub = df[df['Disease'] == dis]
        print(f"  {dis:<15} min={sub['Cases'].min():>4}  "
              f"max={sub['Cases'].max():>5}  "
              f"mean={sub['Cases'].mean():>6.1f}")

    # COVID 2026 spot-check
    covid26 = df[(df['Disease'] == 'COVID-19') & (df['Date'] >= '2026-01-01')]
    print(f"\n  COVID-19 (2026) ─── max={covid26['Cases'].max()}, "
          f"mean={covid26['Cases'].mean():.1f}")
    top5 = covid26.groupby('Region')['Cases'].max().sort_values(ascending=False).head(5)
    for region, val in top5.items():
        print(f"    {region:<22} {val}")

    # Mar 29 2026 spot-check
    today = df[df['Date'] == '2026-03-29']
    print(f"\n  Spot-check 2026-03-29 (sample states):")
    for state in ['Maharashtra', 'Kerala', 'Tamil Nadu', 'Chhattisgarh', 'Punjab']:
        row = today[today['Region'] == state]
        if not row.empty:
            for _, r in row.iterrows():
                print(f"    {state:<22} {r['Disease']:<14} {r['Cases']}")

    print("─────────────────────────────────────────────────────────────")
    print("  ✓ Dataset saved →", outpath)


if __name__ == '__main__':
    main()
