# Supply Chain Analytics: Late Delivery Risk, Demand Forecasting & Profitability

End-to-end analytics project on the **DataCo Smart Supply Chain Dataset** (~180K orders), covering three
connected problems that mirror how real supply chain teams operate: predicting delivery risk, forecasting
demand, and finding where profit quietly leaks out.

## Problem

Supply chains lose money in three predictable ways:
1. **Late deliveries** erode customer trust and violate SLAs
2. **Poor demand forecasting** causes stockouts or excess inventory
3. **Unprofitable order patterns** (discounting, shipping mode choices) quietly erode margin

This project builds a model and analysis for each, then ties every finding back to a concrete business
recommendation rather than stopping at a metric.

## Data

- Source: DataCo Smart Supply Chain Dataset (public, Kaggle)
- ~180,000 orders across multiple countries, product categories, customer segments, and shipping modes
- Fields include order/shipping dates (scheduled vs. actual), shipping mode, region, category, sales,
  discount, profit ratio, and delivery status

> Download the dataset from Kaggle ("DataCo Smart Supply Chain Dataset") and place the CSV at
> `data/DataCoSupplyChainDataset.csv`. See `data/README.md` for details.

## Project Structure

```
dataco-supply-chain-analytics/
├── data/                       # raw data (not committed) + instructions
├── src/
│   ├── 01_eda.py                    # data quality, leakage check, distributions
│   ├── 02_feature_engineering.py    # leakage-free feature set + train/test split
│   ├── 03_late_delivery_model.py    # XGBoost classifier + SHAP
│   ├── 04_demand_forecast.py        # ARIMA vs. XGBoost-on-lags comparison
│   ├── 05_profitability_analysis.py # segmentation across category/region/discount
│   └── 06_sustainability_angle.py   # shipping mode emissions proxy vs. SLA risk
├── outputs/                    # generated charts, cleaned data, model artifacts
├── requirements.txt
├── REPORT.md                   # final write-up template — interview cheat-sheet
└── README.md
```

Run the scripts in order (01 → 06) from inside `src/`; each one reads the outputs of the previous step.

## Models

| Step | Question | Approach |
|---|---|---|
| Late Delivery Risk | Which shipments are likely to arrive late? | XGBoost classifier + SHAP |
| Demand Forecast | How much demand should we expect per category/region? | ARIMA baseline vs. feature-based model |
| Profitability | Where is margin quietly disappearing? | Segmentation across region/category/discount/shipping mode |
| Sustainability | Can we ship greener without hurting SLA? | Shipping-mode emissions proxy vs. late-delivery trade-off |

**Key methodological note:** `Days for shipment (scheduled)` and `Days for shipping (real)` directly
determine the late-delivery label and are excluded from the classifier to avoid target leakage. This is
called out explicitly in `01_eda.py`.

## Key Insight & Business Recommendation

(Filled in as each stage completes — the goal is a specific, quantified recommendation per model, e.g.
"Shipments via mode X from region Y show N times higher late-delivery risk; reallocating this lane could
cut late deliveries by an estimated Z%.")

## Status

- [x] Project scaffolding
- [x] EDA script
- [x] Feature engineering script
- [x] Late delivery model script
- [x] Demand forecast script
- [x] Profitability analysis script
- [x] Sustainability angle script
- [ ] Run all scripts on the real dataset and fill in `REPORT.md` with actual numbers
