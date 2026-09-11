# Final Report — Supply Chain Analytics

Fill in the bracketed values after running each script on the real dataset. This doc is meant to be your
interview cheat-sheet and your GitHub project write-up in one place.

## Problem

Supply chains lose money three ways: late deliveries erode SLAs and trust, poor demand forecasts cause
stockouts or excess inventory, and certain order patterns (discounting, shipping choices) quietly erode
margin. This project addresses all three, plus a sustainability trade-off often ignored in student
projects.

## Data

DataCo Smart Supply Chain Dataset — ~180K orders, multiple regions, categories, shipping modes, and
customer segments. See `data/README.md` for the source.

## Models & Methods

| Analysis | Method | File |
|---|---|---|
| Late delivery risk | XGBoost classifier + SHAP | `src/03_late_delivery_model.py` |
| Demand forecasting | ARIMA baseline vs. XGBoost on lag features | `src/04_demand_forecast.py` |
| Profitability | Segmentation across category/region/discount | `src/05_profitability_analysis.py` |
| Sustainability trade-off | Shipping mode emissions proxy vs. SLA risk | `src/06_sustainability_angle.py` |

**Leakage handling:** `Days for shipping (real)` and `Delivery Status` are excluded from the classifier —
both are only known after a shipment completes and would make the model "cheat." This is called out
explicitly in `01_eda.py` and `02_feature_engineering.py`, and is worth stating proactively in an
interview — it signals you understand the difference between a good offline metric and a usable model.

## Key Insights (fill in after running on real data)

1. **Late delivery:** [Top driver from SHAP] is the strongest predictor; [worst category/mode/region]
   shows a late-delivery rate of [X%] vs. [Y%] for the best-performing group.
2. **Demand forecast:** [Best model] outperformed the naive baseline by [X%] MAE reduction, suggesting
   [category] demand has [trend/seasonality pattern worth noting].
3. **Profitability:** [Worst category/region] is the least profitable segment; [discount finding — does
   heavier discounting actually correlate with lower profit?].
4. **Sustainability:** [X%] of orders use a high-emission shipping mode despite a flexible delivery
   window — a mode-shift rule could reduce emissions intensity with no SLA cost.

## Business Recommendations

Each recommendation should follow this shape: **what to change → why (tied to the data) → estimated
impact.** Avoid stopping at "the model has 0.8 ROC-AUC" — that's a modeling result, not a business
recommendation. Example shape:

> "Shipments via [mode] from [region] show [Nx] higher late-delivery risk. Flagging these at order time
> using the model's probability score lets ops proactively expedite or communicate delays, which could
> reduce late-delivery incidents in that lane by an estimated [X%]."

## What I'd Do With More Time / Data

- SARIMA with explicit seasonal terms, or per-category forecasts instead of one top category
- Real per-mode carbon emissions factors instead of an illustrative proxy
- A cost model connecting late-delivery risk directly to estimated customer-churn cost, to put everything
  in one currency (dollars) rather than three separate metrics

## Resume Bullet (ready to use)

> Built an end-to-end supply chain analytics pipeline (Python, XGBoost, SHAP, ARIMA) on 180K+ orders —
> predicted late-delivery risk ([X]% ROC-AUC), forecasted category-level demand, and identified
> loss-making segments and a sustainability-driven shipping optimization, translating into [X]
> quantified business recommendations.
