"""
Neural Network Forecaster — Disease Spread Prediction System
============================================================
A compact feed-forward neural network (Multi-Layer Perceptron) that learns
non-linear patterns from recent case windows and forecasts future cases.

This is the "neural net" slot of the ensemble. It deliberately uses
scikit-learn's MLPRegressor instead of a TensorFlow/Keras LSTM so the whole
app runs comfortably inside constrained hosting (e.g. a 512 MB free tier),
where importing TensorFlow alone would exhaust memory. The public interface
(`forecast`) is unchanged, so the rest of the system is unaffected.

Design:
  - Sliding window of 7 days → predict next day (recursive multi-step forecast).
  - Min-max scaling per series; predictions clipped to prevent runaway values.
  - In-memory model cache keyed on (disease, region, training-length).
  - Weighted moving-average fallback for very short series or any failure.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

WINDOW = 7
_MODEL_CACHE = {}


def _moving_average_forecast(values: list, steps: int) -> list:
    """Weighted moving average — fast, dependency-free baseline."""
    window  = min(7, len(values))
    base    = values[-window:]
    weights = np.arange(1, window + 1, dtype=float)
    avg     = float(np.average(base, weights=weights))
    if len(values) >= 3:
        trend = (values[-1] - values[-3]) / 2.0
        trend = max(min(trend, avg * 0.08), -avg * 0.08)
    else:
        trend = 0.0
    result, current = [], avg
    for _ in range(steps):
        current = max(0.0, current + trend)
        result.append(int(round(current)))
    return result


def forecast(series, disease: str, region: str, steps: int = 14) -> dict:
    values = series.dropna().astype(float).tolist()

    if len(values) < WINDOW + 5:
        return {'forecast': _moving_average_forecast(values, steps)}

    try:
        from sklearn.neural_network import MLPRegressor

        # Min-max scale to [0, 1] for stable training.
        v_min   = min(values)
        v_max   = max(values)
        v_range = (v_max - v_min) if v_max != v_min else 1.0
        scaled  = [(v - v_min) / v_range for v in values]

        # Build sliding-window supervised samples.
        X, y = [], []
        for i in range(len(scaled) - WINDOW):
            X.append(scaled[i: i + WINDOW])
            y.append(scaled[i + WINDOW])
        X = np.array(X)
        y = np.array(y)

        # Cache on the training length too, so a backtest (trained on a
        # truncated series) never serves the live model trained on the full one.
        cache_key = (disease, region, len(values))
        if cache_key not in _MODEL_CACHE:
            model = MLPRegressor(
                hidden_layer_sizes=(24, 12),
                activation='relu',
                solver='adam',
                alpha=1e-3,                 # L2 regularization
                learning_rate_init=0.01,
                max_iter=400,
                random_state=42,            # deterministic forecasts
            )
            model.fit(X, y)
            _MODEL_CACHE[cache_key] = model

        model = _MODEL_CACHE[cache_key]

        # Recursive multi-step forecast.
        window_vals = list(scaled[-WINDOW:])
        predictions = []
        for _ in range(steps):
            inp  = np.array(window_vals[-WINDOW:]).reshape(1, -1)
            pred = float(model.predict(inp)[0])
            # Clip to [0, 1.2] to stop runaway predictions.
            pred = max(0.0, min(pred, 1.2))
            predictions.append(pred)
            window_vals.append(pred)

        result = [max(0, int(round(p * v_range + v_min))) for p in predictions]
        return {'forecast': result}

    except Exception:
        return {'forecast': _moving_average_forecast(values, steps)}
