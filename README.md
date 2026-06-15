# AI-Based Multi-Disease Spread Prediction System (Phase 2)

> **Academic / educational prototype.** This is a student team project built to demonstrate
> time-series and epidemiological forecasting. It runs entirely on a **synthetic dataset** and
> is **not** intended for real medical, public-health, or government decision-making.

This is the Phase-2 system for the "National Disease Surveillance & Early Warning System".
It features a Flask backend and a vanilla HTML/CSS/JS frontend powered by fully integrated ML models (ARIMA, LSTM, SEIR).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Application**:
    ```bash
    python app.py
    ```

3.  **Access**:
    Open your browser and navigate to `http://127.0.0.1:5000/`.

## Features

-   **Manual Prediction Mode**: Select inputs and view AI-driven contagion risks.
-   **Auto Surveillance Mode**: View live disease risk rankings and alerts based on automated inference.
-   **Government Dashboard**: Overview of monitored diseases and high-risk alerts.
-   **About Page**: Project motivation and team details.

## Model Evaluation
Model accuracy shown on the dashboard comes from an **honest holdout backtest**: the most recent
14 days are hidden, forecast from the earlier history only, and then compared against the held-out
actuals (true out-of-sample RMSE). No accuracy figures are hardcoded.

## Disclaimer
This project runs in Phase 2 with active ML models (ARIMA, LSTM, SEIR) for spread prediction. The
underlying dataset is **synthetic** — algorithmically generated with realistic constraints,
geographical correlations, and meteorological limits for demonstration purposes only. It does not
represent real outbreaks, and its forecasts must not be used for any actual medical, public-health,
or policy decisions.

## Team
Anandhu SA · Abhijith MR · Deedshith DS · Akshay P P
