# 📦 Supply Chain Analytics

**Late Delivery Risk · Demand Forecasting · Profitability · Sustainability**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-EB6E4B)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2)](https://shap.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://dataco-supply-chain-analytics-vqushnhevif9daqrhmpa9z.streamlit.app/)

An end-to-end analytics pipeline on **180,519 real supply chain orders** — built to answer four questions
a real operations team asks: *Will this shipment be late? How much should we stock? Where is margin
leaking? Can we ship greener without breaking our SLA?*

🔗 **[Try the live dashboard →](https://dataco-supply-chain-analytics-vqushnhevif9daqrhmpa9z.streamlit.app/)**

---

## 🎯 Headline Result

> **Shipping Mode alone explains 70% of late-delivery risk.** First Class orders are late **95.2%** of the
> time — versus **37.9%** for Standard Class. Paying for faster shipping is, counterintuitively, the
> single strongest predictor that an order will arrive late.

---

## 🧩 The Problem

Supply chains bleed money in three predictable ways, plus one that's easy to ignore:

| # | Problem | Cost if ignored |
|---|---|---|
| 1 | Late deliveries | Broken SLAs, lost customer trust |
| 2 | Bad demand forecasts | Stockouts or excess inventory |
| 3 | Unprofitable order patterns | Quiet, compounding margin loss |
| 4 | Shipping without a sustainability lens | Unnecessary emissions for no service gain |

This project builds one model or analysis per problem — and every finding ends in a specific,
quantified business recommendation, not just a metric.

---

## 📊 Data

**[DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)** (public, Kaggle)

- 180,519 orders · Jan 2015 – Jan 2018
- 23 order regions · 4 shipping modes · 3 customer segments

> Download the CSV from Kaggle and place it at `data/DataCoSupplyChainDataset.csv` to rerun the pipeline
> yourself. Not required just to run the dashboard — see `data/README.md`.

---

## 🗂️ Project Structure

```
dataco-supply-chain-analytics/
├── app.py                       # 🖥️  Streamlit dashboard — model preview + all results
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
├── REPORT.md                    # 📄 full write-up with real results — interview cheat-sheet
└── README.md
```

---

## 🚀 Quickstart

**Run the analysis pipeline:**
```bash
pip install -r requirements.txt
cd src
python 01_eda.py               # → outputs/dataco_clean.parquet
python 02_feature_engineering.py
python 03_late_delivery_model.py
python 04_demand_forecast.py
python 05_profitability_analysis.py
python 06_sustainability_angle.py
```
Run in order — each script reads the previous step's saved output.

**Launch the dashboard:**
```bash
streamlit run app.py
```
Reads the model, encoders, and charts already saved in `outputs/`. Five views: an interactive late-delivery
risk predictor with a live SHAP explanation, plus a results page for each analysis below.

---

## 🔬 Models & Results

| Analysis | Approach | Key Result |
|---|---|---|
| 🚚 **Late Delivery Risk** | XGBoost classifier + SHAP | **0.75 ROC-AUC.** Shipping Mode = 70% of feature importance — First Class late **95.2%** of the time vs. **37.9%** for Standard Class |
| 📈 **Demand Forecast** | ARIMA vs. naive baseline vs. XGBoost-on-lags | **ARIMA(2,1,2) won**: 66.2 MAE vs. 72.0 (naive, ~8% worse) and 68.3 (XGBoost) |
| 💰 **Profitability** | Segmentation across category/region/discount | **18.71%** of orders are loss-making; profit falls from **$23.94**/order (0–5% discount) to **$18.41**/order (20–30% discount) |
| 🌱 **Sustainability** | Shipping-mode emissions proxy vs. late-delivery trade-off | Hypothesis returned **0 qualifying orders** — scheduling and shipping mode are structurally coupled, reinforcing that First Class isn't a flexible option, it's unreliable by design |

**🛡️ On target leakage:** `Days for shipping (real)` and `Delivery Status` directly determine the
late-delivery label and were excluded from the classifier — confirmed directly in `01_eda.py` (mean real
shipping days: 4.09 for late orders vs. 2.78 for on-time orders, nearly identical to the label itself).

---

## 📄 Full Write-Up

See **[`REPORT.md`](./REPORT.md)** for detailed findings, business recommendations, and honest
limitations for each of the four analyses.

---

## ✅ Status

- [x] All 6 pipeline scripts run end to end on real data
- [x] REPORT.md filled in with real results
- [x] Streamlit dashboard built and deployed
- [x] Live at [dataco-supply-chain-analytics.streamlit.app](https://dataco-supply-chain-analytics-vqushnhevif9daqrhmpa9z.streamlit.app/)
