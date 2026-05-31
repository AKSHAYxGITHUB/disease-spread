"""
Risk Analyzer — Disease Spread Prediction System
Final Production Fix: safe_growth, realistic classification, explainability.
"""

from datetime import datetime
import math


# ── Safe Growth Calculation ───────────────────────────────────────────────────

def safe_growth(current: float, previous: float) -> float:
    """
    Calculate growth % with a minimum baseline to avoid unrealistic spikes.
    e.g. 2 → 6 would give 200% without floor; with floor of 10 it gives 60%.
    Capped between -50% and +100%.
    """
    previous = max(previous, 10)
    growth = ((current - previous) / previous) * 100
    growth = max(min(growth, 100.0), -50.0)
    return round(growth, 2)


# ── Risk Classification ───────────────────────────────────────────────────────

def classify_risk(current_cases: int, forecast_values: list,
                  temp: float = 25.0, hum: float = 60.0, rain: float = 0.0) -> dict:
    """
    Classify risk based on safe growth %, absolute case volume, and trend.

    Returns:
        dict with keys: risk, growth, growth_pct, latest_cases, trend, score,
                        alert_needed, reason
    """
    if not forecast_values or current_cases <= 0:
        return {
            'risk': 'Low',
            'growth': 0.0,
            'growth_pct': 0.0,
            'score': 10,
            'alert_needed': False,
            'trend': 0,
            'latest_cases': 0,
            'reason': 'Insufficient data to assess risk.',
        }

    latest_cases = forecast_values[-1]

    # Safe growth using the dedicated helper
    growth = safe_growth(latest_cases, current_cases)

    # Trend = net change from first to last forecast point
    trend = int(forecast_values[-1] - forecast_values[0])

    # ── Combined Risk Classification Logic ────────────────────────────────────
    if latest_cases < 20:
        risk = "Low"
    elif latest_cases < 50:
        if growth > 30:
            risk = "Medium"
        else:
            risk = "Low"
    else:  # latest_cases >= 50
        if growth > 30 and trend > 20:
            risk = "High"
        elif growth > 10:
            risk = "Medium"
        else:
            risk = "Low"

    # ── Risk Score ────────────────────────────────────────────────────────────
    score = int(growth * (latest_cases / 100))
    score = max(5, min(99, score))

    # ── Explanation Engine ────────────────────────────────────────────────────
    if latest_cases < 20:
        reason = (
            f"Risk is LOW — total projected cases ({int(latest_cases)}) remain "
            f"very low regardless of growth rate ({growth:.1f}%)."
        )
    elif risk == "High":
        reason = (
            f"Risk is HIGH — cases projected at {int(latest_cases)} with "
            f"{growth:.1f}% growth and a rising trend of +{trend} cases."
        )
    elif risk == "Medium":
        reason = (
            f"Risk is MEDIUM — moderate growth of {growth:.1f}% with "
            f"{int(latest_cases)} cases projected. Monitoring advised."
        )
    else:
        reason = (
            f"Risk is LOW — growth ({growth:.1f}%) and case volume "
            f"({int(latest_cases)}) are within manageable limits."
        )

    return {
        'risk': risk,
        'growth': round(growth, 1),
        'growth_pct': round(growth, 1),   # backward compatibility
        'latest_cases': int(latest_cases),
        'score': score,
        'alert_needed': (risk == 'High' and latest_cases > 50),
        'trend': trend,
        'reason': reason,
    }


# ── Alert Generator ───────────────────────────────────────────────────────────

def generate_alert(disease: str, region: str, risk: str, days: int,
                   growth_pct: float, reason: str = '') -> str:
    """Generate a human-readable alert message."""
    ts = datetime.now().strftime('%d %b %Y %H:%M')
    detail = f' [{reason}]' if reason else ''

    if risk == 'High':
        return (
            f"⚠ CRITICAL [{ts}]: {disease} outbreak projected in {region} "
            f"within {days} days. Growth rate: +{growth_pct:.1f}%.{detail} "
            f"Immediate preventive action recommended."
        )
    elif risk == 'Medium':
        return (
            f"⚡ WARNING [{ts}]: Moderate rise in {disease} cases expected in "
            f"{region} over {days} days (+{growth_pct:.1f}%).{detail} Monitor closely."
        )
    else:
        return (
            f"✔ STATUS NORMAL [{ts}]: {disease} in {region} — "
            f"low transmission risk.{detail} No immediate action required."
        )


# ── Auto Surveillance ─────────────────────────────────────────────────────────

def run_auto_surveillance(preprocessing_module) -> list:
    """
    Run surveillance across all diseases for highest-burden region.
    Uses centralized prediction_engine for consistency.
    """
    import prediction_engine

    diseases = preprocessing_module.get_all_diseases()
    regions = preprocessing_module.get_all_regions()
    results = []

    for disease in diseases:
        # Find hotspot region via 7-day average
        best_region = regions[0]
        best_avg = -1
        for r in regions:
            try:
                s = preprocessing_module.get_series(disease, r)
                avg = s.tail(7).mean()
                if avg > best_avg:
                    best_avg = avg
                    best_region = r
            except Exception:
                continue

        try:
            result = prediction_engine.predict(disease, best_region, 14)
            results.append({
                'name': disease,
                'region': best_region,
                'current': result['current_cases'],
                'forecast': result['forecast'],
                'risk': result['risk_info']['risk'],
                'score': result['risk_info']['score'],
                'growth_pct': result['risk_info']['growth_pct'],
                'alert': result['alert_msg'],
                'reason': result['explanation'],
            })
        except Exception as e:
            results.append({
                'name': disease,
                'region': best_region,
                'current': 0,
                'forecast': [],
                'risk': 'Unknown',
                'score': 0,
                'growth_pct': 0,
                'alert': f'Error: {e}',
                'reason': '',
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results