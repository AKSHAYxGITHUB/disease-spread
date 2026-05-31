"""
ARIMA Model — Disease Spread Prediction System
Final Version: Robust ARIMA with fallback to linear trend if statsmodels fails.
Returns forecast list + actual history for RMSE evaluation.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")


def forecast(series, disease: str, region: str, steps: int = 14) -> dict:
    """
    Fit ARIMA on the series and return a forecast.

    Returns:
        dict with 'forecast' (list of ints, length=steps)
                  'actual'   (list of ints — full history for RMSE)
    """
    values = series.dropna().astype(float).tolist()

    if len(values) < 10:
        # Fallback: repeat last known value
        last = int(values[-1]) if values else 0
        return {
            'forecast': [last] * steps,
            'actual': [int(v) for v in values],
        }

    # ── Try ARIMA via statsmodels ─────────────────────────────────────────────
    try:
        from statsmodels.tsa.arima.model import ARIMA

        # Auto-select order: use (2,1,2) as robust default for disease data
        model  = ARIMA(values, order=(2, 1, 2))
        fitted = model.fit()
        pred   = fitted.forecast(steps=steps)
        result = [max(0, int(round(v))) for v in pred]

        return {
            'forecast': result,
            'actual': [int(round(v)) for v in values],
        }

    except Exception:
        pass

    # ── Fallback: Linear trend extrapolation ──────────────────────────────────
    try:
        x      = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)   # slope + intercept
        slope, intercept = coeffs

        future_x = np.arange(len(values), len(values) + steps)
        pred     = slope * future_x + intercept
        result   = [max(0, int(round(v))) for v in pred]

        return {
            'forecast': result,
            'actual': [int(round(v)) for v in values],
        }

    except Exception:
        last = int(values[-1]) if values else 0
        return {
            'forecast': [last] * steps,
            'actual': [int(v) for v in values],
        }