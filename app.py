"""
Disease Spread Prediction System — Phase 2 (Final Production)
Flask application with ARIMA / SEIR / LSTM models, risk classification,
model evaluation (RMSE), and best model selection on dashboard.
"""

from flask import Flask, render_template, request, jsonify
import io, base64, os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np

# --- Phase 2 modules ---
import data_loader
import risk_analyzer

app = Flask(__name__)

# ── Charting helpers ──────────────────────────────────────────────────────────

BG   = '#0a192f'
CARD = '#172a45'
CYAN = '#64ffda'
RED  = '#ff4d4d'
ORG  = '#ff9f43'
WHT  = '#e6f1ff'


def _fig_base():
    fig = Figure(figsize=(8, 3.5), facecolor=BG)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor((100/255, 255/255, 218/255, 0.15))
    ax.tick_params(colors=WHT, labelsize=8)
    ax.title.set_color(WHT)
    return fig, ax


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf8')


def make_forecast_chart_multi(actual, f_arima, f_lstm, f_seir, disease, days):
    fig, ax = _fig_base()

    x_actual = list(range(len(actual)))
    ax.plot(x_actual, actual, color=CYAN, linewidth=1.8,
            marker='o', markersize=3, label='Actual Cases')
    ax.fill_between(x_actual, actual, color=CYAN, alpha=0.07)

    forecast_start = len(actual)
    x_fc = list(range(forecast_start - 1, forecast_start + days))

    line_arima = [actual[-1]] + f_arima
    ax.plot(x_fc, line_arima, color='#ff9f43', linewidth=2,
            linestyle='--', marker='s', markersize=3, label='ARIMA Forecast')

    line_lstm = [actual[-1]] + f_lstm
    ax.plot(x_fc, line_lstm, color='#54a0ff', linewidth=2,
            linestyle='-.', marker='^', markersize=3, label='LSTM Forecast')

    line_seir = [actual[-1]] + f_seir
    ax.plot(x_fc, line_seir, color='#ff4d4d', linewidth=2,
            linestyle=':', marker='x', markersize=3, label='SEIR Projection')

    ax.axvline(x=forecast_start - 1, color=(1, 1, 1, 0.2), linestyle=':', linewidth=1)
    ax.set_title(f'{disease} — Multi-Model AI Forecast ({days} days)', color=WHT, fontsize=10)
    ax.legend(facecolor=CARD, edgecolor=CYAN, labelcolor=WHT, fontsize=8)
    ax.set_xlabel('Days', color=WHT, fontsize=8)
    ax.set_ylabel('Cases', color=WHT, fontsize=8)
    fig.tight_layout(pad=0.5)
    return _fig_to_base64(fig)


def make_seir_chart(seir_result, disease):
    fig, ax = _fig_base()
    fig.set_figwidth(8)
    steps = seir_result['steps']
    t = list(range(steps))

    ax.plot(t, seir_result['S'], color='#54a0ff', linewidth=1.5, label='Susceptible')
    ax.plot(t, seir_result['E'], color=ORG,       linewidth=1.5, label='Exposed')
    ax.plot(t, seir_result['I'], color=RED,        linewidth=2,   label='Infected')
    ax.plot(t, seir_result['R'], color=CYAN,       linewidth=1.5, label='Recovered')

    ax.set_title(f'{disease} SEIR Projection (30 days) | R₀ = {seir_result["R0_value"]}',
                 color=WHT, fontsize=10)
    ax.legend(facecolor=CARD, edgecolor=CYAN, labelcolor=WHT, fontsize=8)
    ax.set_xlabel('Days', color=WHT, fontsize=8)
    ax.set_ylabel('Population', color=WHT, fontsize=8)
    fig.tight_layout(pad=0.5)
    return _fig_to_base64(fig)


def make_bar_chart(labels, values, title='Regional Threat Analysis'):
    fig, ax = _fig_base()
    colors     = [CYAN, ORG, RED, '#54a0ff', '#ffffff']
    bar_colors = (colors * ((len(values) // len(colors)) + 1))[:len(values)]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='none', width=0.6)
    ax.set_title(title, color=WHT, fontsize=10)
    ax.tick_params(axis='x', labelrotation=15, labelsize=7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', color=WHT, fontsize=7)
    fig.tight_layout(pad=0.5)
    return _fig_to_base64(fig)


def make_trend_chart(series_dict):
    """Mini trend sparklines for dashboard — one line per disease."""
    fig = Figure(figsize=(8, 3.5), facecolor=BG)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(CARD)
    palette = [CYAN, ORG, RED, '#54a0ff', '#a29bfe']
    for (disease, values), color in zip(series_dict.items(), palette):
        ax.plot(values, color=color, linewidth=1.8, label=disease)
    ax.set_title('30-Day Case Trend (All Diseases)', color=WHT, fontsize=10)
    ax.legend(facecolor=CARD, edgecolor=CYAN, labelcolor=WHT, fontsize=8,
              loc='upper left', framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor((100/255, 255/255, 218/255, 0.15))
    ax.tick_params(colors=WHT, labelsize=8)
    fig.tight_layout(pad=0.5)
    return _fig_to_base64(fig)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manual', methods=['GET', 'POST'])
def manual():
    diseases = data_loader.get_all_diseases()
    regions  = data_loader.get_all_regions()

    if request.method == 'POST':
        disease   = request.form.get('disease', 'Flu')
        region_in = request.form.get('region', '')
        days      = int(request.form.get('days', 14))

        available_regions = data_loader.get_all_regions()
        match  = next((r for r in available_regions
                       if r.lower() == region_in.strip().lower()), None)
        region = match if match else (available_regions[0] if available_regions else 'Maharashtra')

        try:
            import prediction_engine

            result      = prediction_engine.predict(disease, region, days)
            f_arima     = result['f_arima']
            f_lstm      = result['f_lstm']
            f_seir      = result['f_seir']
            actual_vals = result['actual_history']

            # ── Table Data ────────────────────────────────────────────────
            table_data = []
            today    = data_loader.get_current_date()
            combined = result['forecast']
            for i in range(days):
                date_label = (today + timedelta(days=i + 1)).strftime('%d/%m/%Y')
                table_data.append({
                    'date':     date_label,
                    'combined': combined[i],
                    'arima':    f_arima[i],
                    'lstm':     f_lstm[i] if f_lstm else f_arima[i],
                    'seir':     f_seir[i],
                })

            chart_url = make_forecast_chart_multi(
                actual_vals, f_arima,
                f_lstm if f_lstm else f_arima,
                f_seir, disease, days
            )

            return render_template('manual.html',
                results=True,
                diseases=diseases, regions=regions,
                selected_disease=disease, selected_region=region,
                selected_days=str(days),
                risk=result['risk_info']['risk'],
                growth_pct=result['risk_info']['growth_pct'],
                score=result['risk_info']['score'],
                alert_msg=result['alert_msg'],
                table_data=table_data,
                chart_url=chart_url,
                seir_r0=result['seir_r0'],
                current_cases=result['current_cases'],
                reason=result['explanation'])

        except Exception as e:
            return f"Error processing request: {str(e)}"

    return render_template('manual.html', results=False,
                           diseases=diseases, regions=regions,
                           selected_disease=None, selected_region=None,
                           selected_days='14')


@app.route('/auto', methods=['GET', 'POST'])
def auto():
    if request.method == 'POST' or request.args.get('run'):
        ranked   = risk_analyzer.run_auto_surveillance(data_loader)
        high_risk = [r for r in ranked if r['risk'] == 'High']

        if high_risk:
            names     = ', '.join(d['name'] for d in high_risk)
            alert_msg = f"CRITICAL ALERT: High transmission detected for {names}."
        else:
            alert_msg = "STATUS NORMAL: No critical anomalies detected across all monitored diseases."

        top        = ranked[0]
        top_series = data_loader.get_series(top['name'], top['region'])
        spark_data = top_series.tail(30).astype(int).tolist()

        bar_chart = make_bar_chart(
            [r['name'] for r in ranked],
            [r['score'] for r in ranked],
            title='Disease Risk Score Comparison'
        )

        hotspot_data = data_loader.get_summary_stats()

        return render_template('auto.html',
            results=True,
            ranking=ranked,
            alert_msg=alert_msg,
            bar_chart=bar_chart,
            hotspot=hotspot_data,
            spark_data=spark_data,
            spark_disease=top['name'])

    return render_template('auto.html', results=False)


@app.route('/auto_surveillance')
def auto_surveillance():
    surveillance_data = risk_analyzer.run_auto_surveillance(data_loader)
    return render_template('auto.html', alerts=surveillance_data)


@app.route('/dashboard')
def dashboard():
    import pandas as _pd
    import prediction_engine

    summary  = data_loader.get_summary_stats()
    diseases = data_loader.get_all_diseases()

    ranked     = risk_analyzer.run_auto_surveillance(data_loader)
    high_count = sum(1 for r in ranked if r['risk'] == 'High')

    # 30-day trend per disease
    trend_dict = {}
    for d in diseases:
        try:
            s = data_loader.get_series(d, 'Maharashtra')
            trend_dict[d] = s.tail(30).astype(int).tolist()
        except Exception:
            trend_dict[d] = [0] * 30

    trend_chart = make_trend_chart(trend_dict)

    # Regional bar chart
    df      = data_loader.load_data(force_reload=True)
    latest  = df['Date'].max()
    week_ago = latest - _pd.Timedelta(days=7)
    recent  = df[df['Date'] >= week_ago].copy()
    recent['Cases'] = _pd.to_numeric(recent['Cases'], errors='coerce').fillna(0)
    region_totals = recent.groupby('Region')['Cases'].sum().nlargest(5)

    bar_chart = make_bar_chart(
        region_totals.index.tolist(),
        [int(v) for v in region_totals.values.tolist()],
        title='Top 5 High-Risk Regions (7-Day Total Cases)'
    )

    # SEIR chart for top-risk disease
    top_disease = ranked[0]['name']
    top_region  = ranked[0]['region']
    top_pred    = prediction_engine.predict(top_disease, top_region, 30)
    seir_chart  = make_seir_chart(top_pred['seir_r_full'], top_disease)

    # ── Honest holdout backtest for the headline (top-risk) disease ──────────
    eval_days = 14
    bt = prediction_engine.backtest(top_disease, top_region, eval_days)
    rmse_arima = bt['rmse_arima'] if bt else None
    rmse_lstm  = bt['rmse_lstm']  if bt else None
    rmse_seir  = bt['rmse_seir']  if bt else None
    best_model = bt['best_model'] if bt else 'ARIMA'
    accuracy   = bt['accuracy']   if bt else None   # real backtest accuracy, not hardcoded

    stats = {
        'monitored':   len(diseases),
        'high_risk':   high_count,
        'reports':     summary['total_cases_week'],
        'accuracy':    accuracy,
        'eval_disease': top_disease,
        'eval_days':    eval_days,
        'hotspot':     summary['hotspot_region'],
        'latest_date': summary['latest_date'],
    }

    alerts = [r for r in ranked if r['risk'] in ('High', 'Medium')]

    return render_template('dashboard.html',
        stats=stats,
        latest_date=stats['latest_date'],
        trend_chart=trend_chart,
        bar_chart=bar_chart,
        seir_chart=seir_chart,
        seir_disease=top_disease,
        seir_r0=top_pred['seir_r0'],
        alerts=alerts,
        ranked=ranked,
        # ── RMSE values (new) ──
        rmse_arima=rmse_arima,
        rmse_lstm=rmse_lstm,
        rmse_seir=rmse_seir,
        best_model=best_model,
    )


@app.route('/about')
def about():
    return render_template('about.html')


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Initializing Disease Spread Prediction System (Phase 2) — Live ML Models Active")

    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    if not os.path.exists(dataset_path):
        print("Dataset not found — generating synthetic dataset...")
        import data_generator  # noqa: F401
        print("Dataset ready.")

    app.run(debug=True)