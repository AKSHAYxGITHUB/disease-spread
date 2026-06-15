"""
Centralized Prediction Engine - Final Submission
Provides a unified `predict(disease, region, days)` function to ensure consistent
outputs across manual, auto, and dashboard modes, plus an honest `backtest()`
that holds out recent data to measure real forecast error (RMSE) and accuracy.
"""
import math

from models import arima_model, lstm_model, seir_model
import data_loader
import risk_analyzer


# ── Ensemble & evaluation helpers ──────────────────────────────────────────────

def _ensemble(f_arima: list, f_lstm, f_seir: list, days: int) -> list:
    """Weighted combination of the three model forecasts (LSTM optional)."""
    combined_forecast = []
    for i in range(days):
        cases_a = f_arima[i]
        cases_s = f_seir[i]
        cases_l = f_lstm[i] if f_lstm else None

        if cases_l is not None:
            # 0.5 LSTM + 0.3 ARIMA + 0.2 SEIR
            combined = int(0.5 * cases_l + 0.3 * cases_a + 0.2 * cases_s)
        else:
            # Fallback if LSTM missing: increase ARIMA weight
            combined = int(0.7 * cases_a + 0.3 * cases_s)

        combined_forecast.append(max(0, combined))
    return combined_forecast


def rmse(actual: list, predicted) -> float:
    """Root mean squared error over the overlapping window; None if no overlap."""
    if not actual or not predicted:
        return None
    length = min(len(actual), len(predicted))
    if length == 0:
        return None
    sq_err = sum((a - p) ** 2 for a, p in zip(actual[:length], predicted[:length]))
    return round(math.sqrt(sq_err / length), 2)


def accuracy_from_mape(actual: list, predicted) -> float:
    """Accuracy = 100 - MAPE (%), clamped to [0, 100]; None if not computable."""
    if not actual or not predicted:
        return None
    pairs = [(a, p) for a, p in zip(actual, predicted) if a > 0]
    if not pairs:
        return None
    mape = sum(abs(a - p) / a for a, p in pairs) / len(pairs)
    return round(max(0.0, min(100.0, (1.0 - mape) * 100.0)), 1)


def best_model(rmse_arima, rmse_lstm, rmse_seir) -> str:
    """Return the name of the model with the lowest RMSE."""
    scores = {}
    if rmse_arima is not None:
        scores['ARIMA'] = rmse_arima
    if rmse_lstm is not None:
        scores['LSTM'] = rmse_lstm
    if rmse_seir is not None:
        scores['SEIR'] = rmse_seir
    if not scores:
        return 'ARIMA'
    return min(scores, key=scores.get)


def backtest(disease: str, region: str, horizon: int = 14) -> dict:
    """
    Honest holdout backtest: hide the last `horizon` days, forecast them using
    ONLY the earlier data, then compare each model against the held-out actuals.

    This is a true out-of-sample evaluation (unlike comparing a future forecast
    to recent history). Returns per-model RMSE, the best model, and an ensemble
    accuracy. Returns None if there isn't enough history to hold out a window.
    """
    series = data_loader.get_series(disease, region)
    values = series.dropna().astype(float)

    # Need enough history for a model to train AND a window to hold out.
    if len(values) < horizon + 15:
        return None

    train = values.iloc[:-horizon]
    test  = [int(round(v)) for v in values.iloc[-horizon:].tolist()]

    latest_row = data_loader.get_latest_row(disease, region)
    temp = latest_row.get('Temperature', 25.0)
    hum  = latest_row.get('Humidity', 60.0)
    rain = latest_row.get('Rainfall', 50.0)

    f_arima = arima_model.forecast(train, disease, region, steps=horizon)['forecast']

    f_lstm = None
    try:
        f_lstm = lstm_model.forecast(train, disease, region, steps=horizon)['forecast']
    except Exception:
        pass

    f_seir = seir_model.simulate(
        disease=disease, current_cases=int(train.iloc[-1]),
        temp=temp, humidity=hum, rain=rain, steps=horizon
    )['forecast']

    f_combined = _ensemble(f_arima, f_lstm, f_seir, horizon)

    rmse_arima = rmse(test, f_arima)
    rmse_lstm  = rmse(test, f_lstm) if f_lstm else None
    rmse_seir  = rmse(test, f_seir)

    return {
        'horizon':    horizon,
        'rmse_arima': rmse_arima,
        'rmse_lstm':  rmse_lstm,
        'rmse_seir':  rmse_seir,
        'best_model': best_model(rmse_arima, rmse_lstm, rmse_seir),
        'accuracy':   accuracy_from_mape(test, f_combined),
    }


def predict(disease: str, region: str, days: int = 14) -> dict:
    """
    Unified prediction engine.
    Ensures identical forecast, risk, and explanations across all endpoints.
    """
    series  = data_loader.get_series(disease, region)
    current = int(series.iloc[-1])
    latest_row = data_loader.get_latest_row(disease, region)

    # Environmental parameters for SEIR
    temp = latest_row.get('Temperature', 25.0)
    hum  = latest_row.get('Humidity', 60.0)
    rain = latest_row.get('Rainfall', 50.0)

    # 1. Run Models
    res_arima = arima_model.forecast(series, disease, region, steps=days)
    f_arima   = res_arima['forecast']
    
    res_lstm = None
    try:
        res_lstm = lstm_model.forecast(series, disease, region, steps=days)
    except Exception:
        pass
    
    f_lstm = res_lstm['forecast'] if res_lstm else None

    seir_r = seir_model.simulate(
        disease=disease, 
        current_cases=current, 
        temp=temp, 
        humidity=hum, 
        rain=rain, 
        steps=days
    )
    f_seir = seir_r['forecast']

    # 2. Ensemble Weights
    combined_forecast = _ensemble(f_arima, f_lstm, f_seir, days)

    # 3. Risk Engine & Explanation
    risk_info = risk_analyzer.classify_risk(
        current_cases=current, 
        forecast_values=combined_forecast,
        temp=temp,
        hum=hum,
        rain=rain
    )
    
    alert_msg = risk_analyzer.generate_alert(
        disease=disease, 
        region=region, 
        risk=risk_info['risk'], 
        days=days,
        growth_pct=risk_info['growth_pct'], 
        reason=risk_info.get('reason', '')
    )

    models_used = "LSTM, ARIMA, SEIR (Ensemble)" if f_lstm else "ARIMA, SEIR (Ensemble)"

    return {
        'disease': disease,
        'region': region,
        'days': days,
        'current_cases': current,
        'forecast': combined_forecast,
        'f_arima': f_arima,
        'f_lstm': f_lstm,
        'f_seir': f_seir,
        'seir_r_full': seir_r,
        'seir_r0': seir_r['R0_value'],
        'actual_history': res_arima['actual'],
        'risk_info': risk_info,
        'alert_msg': alert_msg,
        'explanation': risk_info.get('reason', ''),
        'models_used': models_used,
        'environmental': {
            'temp': temp,
            'humidity': hum,
            'rainfall': rain
        }
    }
