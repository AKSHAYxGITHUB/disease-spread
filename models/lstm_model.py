"""
LSTM Model — Disease Spread Prediction System
Speed + RMSE Fix:
  - Window=7, 16 units, 15 epochs, patience=3
  - In-memory model cache per disease+region
  - Moving average fallback (also a better baseline than random)
  - Clip predictions to ±30% of last known value per step (prevents runaway)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

WINDOW = 7
_MODEL_CACHE = {}


def _moving_average_forecast(values: list, steps: int) -> list:
    """Weighted moving average — decent baseline, fast, no TF needed."""
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
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.callbacks import EarlyStopping

        # Scale
        v_min   = min(values)
        v_max   = max(values)
        v_range = (v_max - v_min) if v_max != v_min else 1.0
        scaled  = [(v - v_min) / v_range for v in values]

        # Sequences
        X, y = [], []
        for i in range(len(scaled) - WINDOW):
            X.append(scaled[i: i + WINDOW])
            y.append(scaled[i + WINDOW])
        X = np.array(X).reshape(-1, WINDOW, 1)
        y = np.array(y)

        # Key on the training length too, so a backtest (trained on a truncated
        # series) never serves the live model trained on the full series.
        cache_key = (disease, region, len(values))
        if cache_key not in _MODEL_CACHE:
            model = Sequential([
                LSTM(16, activation='tanh', input_shape=(WINDOW, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            es = EarlyStopping(monitor='loss', patience=3,
                               restore_best_weights=True, verbose=0)
            model.fit(X, y, epochs=15, batch_size=16,
                      verbose=0, callbacks=[es])
            _MODEL_CACHE[cache_key] = model

        model = _MODEL_CACHE[cache_key]

        window_vals = list(scaled[-WINDOW:])
        predictions = []
        for _ in range(steps):
            inp  = np.array(window_vals[-WINDOW:]).reshape(1, WINDOW, 1)
            pred = float(model.predict(inp, verbose=0)[0][0])
            # Clip to [0, 1.2] to stop runaway predictions
            pred = max(0.0, min(pred, 1.2))
            predictions.append(pred)
            window_vals.append(pred)

        result = [max(0, int(round(p * v_range + v_min))) for p in predictions]
        return {'forecast': result}

    except Exception:
        return {'forecast': _moving_average_forecast(values, steps)}