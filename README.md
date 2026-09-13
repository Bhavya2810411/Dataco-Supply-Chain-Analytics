# Supply Chain Analytics: Late Delivery Risk, Demand Forecasting & Profitability

End-to-end analytics project on the **DataCo Smart Supply Chain Dataset** (180,519 orders), covering four
connected problems that mirror how real supply chain teams operate: predicting delivery risk, forecasting
demand, finding where profit quietly leaks out, and testing a sustainability-driven shipping trade-off.

**Live dashboard:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(update once deployed)*

## Problem

Supply chains lose money in three predictable ways:
1. **Late deliveries** erode customer trust and violate SLAs
2. **Poor demand forecasting** causes stockouts or excess inventory
3. **Unprofitable order patterns** (discounting, shipping mode choices) quietly erode margin

This project builds a model or analysis for each, plus a sustainability angle, and ties every finding
back to a concrete, quantified business recommendation rather than stopping at a metric.

## Data

- Source: [DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (public, Kaggle)
- 180,519 orders, Jan 2015–Jan 2018, across 23 regions, 4 shipping modes, and 3 customer segments
- Fields include order/shipping dates (scheduled vs. actual), shipping mode, region, category, sales,
  discount, profit ratio, and delivery status

> Download the dataset from Kaggle and place the CSV at `data/DataCoSupplyChainDataset.csv`. See
> `data/README.md` for details. Not required to run the dashboard — only to regenerate the analysis.

## Project Structure

```
dataco-supply-chain-analytics/
├── app.py                       # Streamlit dashboard (model preview + all results)
├── data/                        # raw data (not committed) + instructions
├── src/
│   ├── 01_eda.py                    # data quality, leakage check, distributions
│   ├── 02_feature_engineering.py    # leakage-free feature set + train/test split
│   ├── 03_late_delivery_model.py    # XGBoost classifier + SHAP
│   ├── 04_demand_forecast.py        # ARIMA vs. XGBoost-on-lags comparison
│   ├── 05_profitability_analysis.py # segmentation across category/region/discount
│   └── 06_sustainability_angle.py   # shipping mode emissions proxy vs. SLA risk
├── outputs/                     # generated charts, model artifacts, comparison tables
├── requirements.txt
├── REPORT.md                    # full write-up with real results — interview cheat-sheet
└── README.md
```

## Running the analysis

```bash
pip install -r requirements.txt
cd src
python 01_eda.py
python 02_feature_engineering.py
python 03_late_delivery_model.py
python 04_demand_forecast.py
python 05_profitability_analysis.py
python 06_sustainability_angle.py
```

Run in order — each script reads the previous step's saved output from `outputs/`.

## Running the dashboard

```bash
streamlit run app.py
```

Reads the model, encoders, and charts already saved in `outputs/` — no need to rerun the pipeline first
if `outputs/` is already populated. Five views: an interactive late-delivery risk predictor with a live
SHAP explanation, plus results pages for each of the four analyses below.

## Models & Results

| Analysis | Approach | Key Result |
|---|---|---|
| Late Delivery Risk | XGBoost classifier + SHAP | 0.75 ROC-AUC. Shipping Mode = 70% of feature importance — First Class orders are late 95.2% of the time vs. 37.9% for Standard Class |
| Demand Forecast | ARIMA vs. naive baseline vs. XGBoost-on-lags | ARIMA(2,1,2) won: 66.2 MAE vs. 72.0 (naive, ~8% worse) and 68.3 (XGBoost) |
| Profitability | Segmentation across category/region/discount | 18.71% of orders are loss-making; profit falls from $23.94/order (0–5% discount) to $18.41/order (20–30% discount) |
| Sustainability | Shipping-mode emissions proxy vs. late-delivery trade-off | Hypothesis returned 0 qualifying orders — scheduled days don't vary independently of shipping mode, reinforcing that First Class isn't a flexible option, it's structurally unreliable |

**Key methodological note:** `Days for shipping (real)` and `Delivery Status` directly determine the
late-delivery label and are excluded from the classifier to avoid target leakage — confirmed directly in
`01_eda.py` (mean real shipping days: 4.09 for late orders vs. 2.78 for on-time orders, nearly identical
to the label itself).

## Full write-up

See [`REPORT.md`](./REPORT.md) for detailed findings, business recommendations, and limitations for each
of the four analyses.

## Status

- [x] Project scaffolding
- [x] All 6 pipeline scripts run end to end on real data
- [x] REPORT.md filled in with real results
- [x] Streamlit dashboard built
- [ ] Deployed to Streamlit Community Cloud (update the live link above once done)
