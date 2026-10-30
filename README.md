# URBANFLOW
### NYC Taxi Demand Forecasting Using Machine Learning

URBANFLOW is an applied machine-learning project for forecasting New York City yellow taxi pickup demand at 15-minute intervals across 262 taxi zones.

The project combines large-scale data processing with DuckDB, time-series feature engineering, LightGBM regression, chronological evaluation, and an interactive Streamlit dashboard.

## Project Overview

The study uses NYC Taxi and Limousine Commission (TLC) Yellow Taxi Trip Records from January through December 2025.

| Component | Details |
|---|---|
| Data source | NYC TLC Yellow Taxi Trip Records |
| Period | January to December 2025 |
| Accepted pickup events | 48,613,289 |
| Forecasting observations | 9,004,416 |
| Geographic coverage | 262 taxi zones |
| Temporal resolution | 15 minutes |
| Primary model | LightGBM V2, L2 regression |
| Baseline | Previous 15-minute demand |
| Dashboard | Streamlit |

## Data Processing

Individual trip records are aggregated into 15-minute zone-level pickup counts using DuckDB.

The complete aggregation grid includes zone-time combinations with zero recorded pickups. Seven initial days are used to establish the historical context required by the lag features.

The final model dataset includes calendar variables, lagged demand observations and historical rolling averages.

## Model Development

The research investigates LightGBM regression configurations using a chronological train-validation-test strategy.

- Training: January 8 to August 31, 2025
- Validation: September 1 to October 31, 2025
- Testing: November 1 to December 31, 2025

The final model was selected based on validation performance, then evaluated on the held-out test period.

## Final Results

| Metric | LightGBM V2 | Persistence Baseline |
|---|---:|---:|
| Test MAE | 1.330524 | 1.580073 |
| Relative MAE improvement | 15.79% | Reference |

Additional analysis found that LightGBM outperformed persistence on 60 of 61 test days and in 178 of 262 taxi zones.

A seven-day moving-block bootstrap estimated a 95% confidence interval of approximately 14.04% to 17.32% for the relative MAE improvement.

## Interactive Dashboard

The Streamlit application allows users to explore:

- Historical actual versus predicted pickup demand
- Zone-level forecasting performance
- Daily demand patterns and prediction errors
- Comparisons with the persistence baseline
- Downloadable historical prediction records

**Important:** The dashboard is a historical test-set replay for November and December 2025. It does not provide live or real-time forecasts.

## Project Structure

```text
urbanflow-nyc-taxi-demand-forecasting/
    app.py
    requirements.txt
    README.md
    data/
        urbanflow_dashboard_test_v2.parquet
```

This is the initial dashboard deployment structure. The complete preprocessing, training and research-evaluation notebooks will be added to the repository separately.

## Run Locally

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

The dashboard loads the bundled historical prediction dataset. It does not require retraining LightGBM.

## Data Source

Original taxi records are published by the New York City Taxi and Limousine Commission:

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The original trip records are not included in this deployment repository.

## Research Limitations

The existing model was developed using data from a single calendar year. Its performance on future years has not been independently established.

The model is less reliable in low-demand taxi zones, and the original fixed 15-minute timestamp representation has a daylight-saving-time limitation.

The reported results are comparisons against specified historical baselines, not proof of superiority over every forecasting approach.

## Future Work

Potential extensions include out-of-year generalization testing, improved sparse-demand forecasting, temporal feature enhancements and a more advanced geographic dashboard.

## Project Status

Core model development and frozen-model evaluation are complete. Dashboard deployment and final repository documentation are in progress.
