# URBANFLOW
### NYC Taxi Demand Forecasting with Machine Learning

**A large-scale, zone-level taxi-demand forecasting study using LightGBM, DuckDB, and historical NYC TLC trip records.**

URBANFLOW forecasts New York City yellow taxi pickup demand at **15-minute intervals across 262 taxi zones**. The project combines large-scale data engineering, time-series feature construction, gradient-boosted regression, chronological evaluation, statistical analysis, and an interactive Streamlit dashboard.

**Author:** Divyanshu Beniwal  
**Institution:** Delhi Skill and Entrepreneurship University, Bhai Parmanand Campus  
**Field:** Data Science

**Project Links**

- **Live Dashboard:** (https://urbanflow-nyc-taxi-forecasting-xwbpuhxjwksgdms5pd7qhj.streamlit.app/)
- **Research Paper:** [Read the research paper](reports/URBANFLOW_Research_Paper.pdf)
- **Source Repository:** [GitHub](https://github.com/div12-space/URBANFLOW-NYC-Taxi-Forecasting)

---

## 1. Project Overview

URBANFLOW investigates whether gradient-boosted decision trees can improve short-horizon NYC taxi pickup-demand forecasting compared with historical-demand baselines.

The study uses official **2025 NYC Taxi and Limousine Commission (TLC) Yellow Taxi Trip Records**.

| Property | Value |
|---|---|
| Original data source | NYC TLC Yellow Taxi Trip Records |
| Study period | January to December 2025 |
| Accepted taxi pickups | 48,613,289 |
| Full-year aggregated observations | 9,180,480 |
| Final forecasting observations | 9,004,416 |
| Geographic coverage | 262 NYC taxi zones |
| Forecast resolution | 15 minutes |
| Forecasting model | LightGBM V2 (L2 objective) |
| Training framework | Python, DuckDB, LightGBM |
| Visualization | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud |

The model is evaluated as a historical one-step-ahead forecasting system. It does not currently perform live taxi-demand prediction.

---

## 2. Research Objective

The central research question is:

**Can a LightGBM regression model reduce 15-minute taxi pickup-count forecasting error across NYC taxi zones relative to historical-demand baselines?**

The study also investigates:

- Forecasting reliability across high- and low-demand zones.
- Temporal variation in prediction accuracy.
- Performance during extreme pickup-demand periods.
- Statistical uncertainty in the measured improvement.
- Limitations of sparse-demand forecasting.

---

## 3. Data Processing Pipeline

The original trip-level data is transformed into a structured zone-time forecasting dataset.

```text
NYC TLC Yellow Taxi Trip Records
              |
              v
       Data Validation
              |
              v
    DuckDB SQL Aggregation
              |
              v
  15-Minute Zone-Level Demand
              |
              v
       Complete Time Grid
              |
              v
       Feature Engineering
              |
              v
  Chronological Data Splitting
              |
              v
    LightGBM Model Development
              |
              v
       Frozen Evaluation
              |
              v
    Historical Streamlit Dashboard
```

Each forecasting observation represents the number of recorded taxi pickups in a given zone during one 15-minute interval.

The complete annual grid contains:

**262 zones x 96 intervals/day x 365 days = 9,180,480 observations**

The first seven days provide historical context for weekly lag features.

After this warm-up period, **9,004,416 observations** remain for supervised forecasting.

The retained forecasting targets represent **47,990,987 recorded pickups**.

The distinction matters: approximately 48.6 million individual trip events were processed into approximately 9 million aggregated forecasting observations. These quantities represent different levels of data granularity.

---

## 4. Feature Engineering

The selected LightGBM model uses 11 input features.

| Feature | Description |
|---|---|
| `zone_id` | NYC taxi zone identifier |
| `hour` | Hour of day |
| `minute` | Minute within the hour |
| `day_of_week` | Day-of-week indicator |
| `is_weekend` | Weekend indicator |
| `lag_1` | Previous 15-minute pickup count |
| `lag_4` | Pickup count one hour earlier |
| `lag_96` | Pickup count one nominal day earlier |
| `lag_672` | Pickup count one nominal week earlier |
| `rolling_mean_4` | Mean demand over previous four intervals |
| `rolling_mean_96` | Mean demand over previous 96 intervals |

Lag and rolling statistics are calculated using historical observations from the same taxi zone.

The original SQL rolling windows exclude the current target observation.

---

## 5. Model Development

Four LightGBM configurations were investigated:

| Model | Regression objective |
|---|---|
| V1 | Absolute-error regression (L1) |
| V2 | Squared-error regression (L2) |
| V3 | Poisson regression |
| V4 | Weighted absolute-error regression |

Model selection was performed using chronological validation data.

### Chronological Dataset Partition

| Partition | Period | Observations |
|---|---|---:|
| Training | Jan 8 - Aug 31, 2025 | 5,935,872 |
| Validation | Sep 1 - Oct 31, 2025 | 1,534,272 |
| Testing | Nov 1 - Dec 31, 2025 | 1,534,272 |

The final LightGBM V2 model was selected using validation results and then evaluated on the held-out November-December test period.

Predictions were clipped to zero to enforce nonnegative pickup counts.

---

## 6. Experimental Results

### Final Test Performance

The selected LightGBM V2 model achieved a mean absolute error of **1.330524**, compared with **1.580073** for previous-interval persistence.

| Model | MAE |
|---|---:|
| LightGBM V2 | **1.330524** |
| Previous 15-minute persistence | 1.580073 |
| Previous-day seasonal baseline | 2.216162 |
| Previous-week seasonal baseline | 2.232044 |

**Relative improvement over persistence: 15.79%**

Additional LightGBM V2 test metrics:

- RMSE: **3.331967**
- WAPE: **24.10%**
- Test observations: **1,534,272**

All reported metrics refer to historical, held-out observations from November and December 2025.

### Statistical Evaluation

A paired seven-day moving-block bootstrap with 5,000 replications estimated a 95% confidence interval of approximately **14.04% to 17.32%** for the relative MAE improvement.

This interval characterizes uncertainty under the chosen resampling procedure within the historical test period. It does not guarantee performance in future years.

### Temporal and Geographic Reliability

The model outperformed persistence on:

- **60 of 61 test days**
- **178 of 262 taxi zones**
- **All 79 high-activity zones**

However, all 69 zones with fewer than 1,000 recorded test-period pickups performed worse than the persistence baseline.

These results highlight substantial geographic variation in forecasting reliability.

---

## 7. Interactive Dashboard

URBANFLOW includes a publicly deployed Streamlit dashboard for exploring historical predictions.

### Available Features

- Select an NYC taxi zone.
- Select a date from November or December 2025.
- Compare actual and predicted pickup demand.
- Examine persistence-baseline forecasts.
- Inspect zone-level MAE.
- Visualize prediction errors.
- Download selected historical prediction records as CSV.

**Live Application:** PASTE_YOUR_FULL_STREAMLIT_URL_HERE

The dashboard reads a precomputed Parquet file containing predictions from the frozen LightGBM V2 model.

It is a historical forecasting demonstration, not a live inference endpoint.

---

## 8. Repository Structure

```text
URBANFLOW-NYC-Taxi-Forecasting/
|
|-- README.md
|-- LICENSE
|
|-- URBANFLOW_Deployment/
|   |-- app.py
|   |-- requirements.txt
|   |
|   `-- data/
|       `-- urbanflow_dashboard_test_v2.parquet
|
`-- reports/
    `-- URBANFLOW_Research_Paper.pdf
```

The deployed application and its historical prediction data are included in this repository.

The complete original raw-data processing, training, and saved-model reconstruction workflow is maintained separately in the author's research notebooks and is not bundled in this dashboard release.

---

## 9. Run the Dashboard Locally

Clone the repository:

```bash
git clone https://github.com/div12-space/URBANFLOW-NYC-Taxi-Forecasting.git
cd URBANFLOW-NYC-Taxi-Forecasting
```

Install the Python dependencies:

```bash
pip install -r URBANFLOW_Deployment/requirements.txt
```

Start the Streamlit application:

```bash
streamlit run URBANFLOW_Deployment/app.py
```

The application will load the historical prediction dataset from its local `data` directory.

**Note:** Running this dashboard does not retrain LightGBM or regenerate the original predictions.

---

## 10. Reproducibility

The reported model results were reproduced using the saved LightGBM V2 bundle and the original feature-engineered Parquet dataset.

Reproducing the frozen evaluation requires the same:

- Model bundle and ordered feature names.
- Taxi-zone categorical encoding.
- Chronological test-period selection.
- Feature-engineered observations.
- Nonnegative prediction clipping.

The dashboard itself uses precomputed test predictions and provides a reproducible interface for exploring those results.

The repository's deployment files alone do not reproduce the complete model-training process from raw NYC TLC records.

---

## 11. Research Paper

**URBANFLOW: Fifteen-Minute NYC Yellow Taxi Pickup-Demand Forecasting with Gradient-Boosted Trees**

Author: Divyanshu Beniwal

[Read the full technical research paper](reports/URBANFLOW_Research_Paper.pdf)

The manuscript documents the methodology, experimental design, evaluation metrics, statistical evidence, reliability analysis, and limitations.

It is an independent technical research manuscript and has not undergone peer review.

---

## 12. Limitations

URBANFLOW V1.0 has several important limitations.

**Single-year evaluation:** All training, validation, and testing data belong to 2025. Performance in subsequent years has not been established.

**Sparse-demand reliability:** Low-activity zones may be better served by simple historical baselines.

**Historical information dependency:** Forecasting relies on preceding realized pickup counts being available at prediction time.

**Daylight-saving treatment:** The historical processing pipeline uses timezone-naive timestamps and fixed 15-minute grids. New York daylight-saving transitions are not fully resolved.

**Benchmark coverage:** The results do not establish superiority over forecasting architectures or methods that were not evaluated.

**Deployment scope:** The public dashboard replays historical test predictions. It does not ingest real-time taxi records or generate new operational forecasts.

---

## 13. Future Research

Potential extensions include:

- Testing generalization on unseen 2026 data.
- Multi-year training and evaluation.
- Improving sparse-demand forecasts.
- Evaluating additional time-series forecasting models.
- Explicit timezone-aware data processing.
- Prediction uncertainty and calibration.
- Interactive geographic demand visualization.
- Operational forecasting with continuously updated data.

These are future research directions, not completed capabilities of the current release.

---

## 14. Data Sources and References

1. [NYC Taxi and Limousine Commission - TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
2. [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
3. [URBANFLOW Technical Research Paper](reports/URBANFLOW_Research_Paper.pdf)

---

## 15. License

This repository's original source code is distributed under the [MIT License](LICENSE).

The original NYC TLC trip records remain subject to their source publisher's applicable terms and documentation.

---

## Project Author

**Divyanshu Beniwal**  
B.Tech. Data Science  
Delhi Skill and Entrepreneurship University  
Bhai Parmanand Campus, Delhi, India

**GitHub:** [div12-space](https://github.com/div12-space)

---

*URBANFLOW V1.0 - Historical NYC Taxi Demand Forecasting Research and Interactive Dashboard*
