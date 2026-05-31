# AI-Based Multi-Disease Spread Prediction System (Phase 2)

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

## Disclaimer
This project is currently running in Phase 2 with active ML models (ARIMA, LSTM, SEIR) for spread prediction. Note that the underlying dataset is synthetic but is generated with realistic constraints, geographical correlations, and meteorological limits for demonstration purposes.
