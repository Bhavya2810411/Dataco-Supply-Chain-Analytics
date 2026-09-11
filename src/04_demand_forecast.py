"""
04_demand_forecast.py — Weekly demand forecasting at the category level.

Builds a simple, defensible forecasting baseline (ARIMA) and compares it to a
feature-based model (XGBoost on lag features), rather than jumping straight
to the fanciest model. In an interview, being able to say "I compared X and Y
and Y won because..." is worth more than a single black-box model.
"""

# %% Imports
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# %% Load cleaned data
df = pd.read_parquet(OUTPUT_DIR / "dataco_clean.parquet")
if "order_date" not in df.columns and "order date (DateOrders)" in df.columns:
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")

# %% Pick the top product category by volume to forecast
# (Forecasting every category individually is a natural "v2" extension —
# start with one to prove the approach works.)
top_category = df["Category Name"].value_counts().index[0]
print(f"Forecasting demand for top category: {top_category}")

cat_df = df[df["Category Name"] == top_category].copy()

# %% Aggregate to weekly order volume
weekly = (
    cat_df.set_index("order_date")
    .resample("W")["Order Item Quantity"]
    .sum()
    .rename("demand")
    .to_frame()
)
weekly = weekly[weekly.index.notnull()]
print(weekly.describe())

plt.figure(figsize=(10, 4))
weekly["demand"].plot()
plt.title(f"Weekly Demand — {top_category}")
plt.ylabel("Units ordered")
plt.savefig(OUTPUT_DIR / "weekly_demand_raw.png", bbox_inches="tight")
plt.close()

# %% Train/test split (last 12 weeks held out — forecasting needs a TIME split, not random)
n_test = 12
train, test = weekly.iloc[:-n_test], weekly.iloc[-n_test:]
print(f"Train weeks: {len(train)}, Test weeks: {len(test)}")

# %% Baseline: naive seasonal-ish forecast (last observed value repeated)
naive_pred = np.repeat(train["demand"].iloc[-1], n_test)
naive_mae = mean_absolute_error(test["demand"], naive_pred)
print(f"Naive baseline MAE: {naive_mae:.1f}")

# %% ARIMA model
# Order (p,d,q) kept simple and interpretable rather than auto-tuned — good
# enough for a portfolio project; mention in the writeup that auto_arima /
# SARIMA with seasonal terms is a natural extension.
arima_model = ARIMA(train["demand"], order=(2, 1, 2))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=n_test)
arima_mae = mean_absolute_error(test["demand"], arima_pred)
arima_rmse = np.sqrt(mean_squared_error(test["demand"], arima_pred))
print(f"ARIMA MAE: {arima_mae:.1f}, RMSE: {arima_rmse:.1f}")

# %% Feature-based model: XGBoost on lag features
def make_lag_features(series, n_lags=4):
    df_lag = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df_lag[f"lag_{lag}"] = series.shift(lag)
    df_lag["weekofyear"] = series.index.isocalendar().week.values
    return df_lag.dropna()

lag_df = make_lag_features(weekly["demand"])
lag_train = lag_df.iloc[:-n_test]
lag_test = lag_df.iloc[-n_test:]

xgb_reg = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
xgb_reg.fit(lag_train.drop(columns="y"), lag_train["y"])
xgb_pred = xgb_reg.predict(lag_test.drop(columns="y"))
xgb_mae = mean_absolute_error(lag_test["y"], xgb_pred)
print(f"XGBoost (lag features) MAE: {xgb_mae:.1f}")

# %% Compare all three
comparison = pd.DataFrame({
    "model": ["Naive baseline", "ARIMA(2,1,2)", "XGBoost (lags)"],
    "MAE": [naive_mae, arima_mae, xgb_mae],
})
print("\nModel comparison:\n", comparison)
best_model = comparison.loc[comparison["MAE"].idxmin(), "model"]
print(f"\n>>> Best performing model: {best_model}")

# %% Plot forecast vs actual (ARIMA, as the primary interpretable model)
plt.figure(figsize=(10, 4))
plt.plot(train.index[-20:], train["demand"].iloc[-20:], label="Train (recent)")
plt.plot(test.index, test["demand"], label="Actual", marker="o")
plt.plot(test.index, arima_pred, label="ARIMA forecast", marker="x")
plt.legend()
plt.title(f"Demand Forecast vs Actual — {top_category}")
plt.savefig(OUTPUT_DIR / "demand_forecast_comparison.png", bbox_inches="tight")
plt.close()

comparison.to_csv(OUTPUT_DIR / "forecast_model_comparison.csv", index=False)
print(f"\nSaved comparison table and charts to {OUTPUT_DIR}")
