"""
Centralized Prediction Engine - Final Submission
Provides a unified `predict(disease, region, days)` function to ensure consistent
outputs across manual, auto, and dashboard modes.
"""
from models import arima_model, lstm_model, seir_model
import data_loader
import risk_analyzer

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
