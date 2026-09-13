"""
01_eda.py — Exploratory Data Analysis for the DataCo Smart Supply Chain dataset.

Goals:
1. Load the data safely (it ships in latin1 encoding, not UTF-8)
2. Get a first read on shape, missing values, and column types
3. Look at the target variable for the late-delivery model
4. Explicitly flag and explain the target-leakage columns
5. Save a lightly cleaned version for the next steps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 60)
sns.set_style("whitegrid")

# The DataCo CSV is encoded in latin1 / ISO-8859-1, not UTF-8 — reading it as
# UTF-8 will throw a UnicodeDecodeError partway through the file.
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")
print(f"Shape: {df.shape}")
df.head()

# %% Column overview
print(df.dtypes)
print("\nColumns:\n", list(df.columns))

# %% Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Columns with missing values:\n", missing)

# A couple of columns (e.g. 'Order Zipcode', 'Product Description') are often
# almost entirely null in this dataset — worth confirming and dropping if so.
mostly_null = missing[missing > 0.5 * len(df)]
print("\nColumns >50% null (candidates to drop):\n", mostly_null)

# %% Target variable: Late_delivery_risk
target_col = "Late_delivery_risk"
print(df[target_col].value_counts(normalize=True))

sns.countplot(data=df, x=target_col)
plt.title("Late Delivery Risk — Class Balance")
plt.savefig("outputs/target_balance.png", bbox_inches="tight")
plt.close()

# %% ⚠️ TARGET LEAKAGE CHECK — read this before modeling
# 'Late_delivery_risk' is essentially derived from comparing scheduled vs.
# actual shipping days. If these columns are left in as model features, the
# model will "predict" the label almost perfectly by cheating — this looks
# like a great accuracy score but is worthless in a real forecasting setting
# (you don't know the ACTUAL shipping days in advance, only the scheduled ones).
leakage_candidates = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Status",  # also derived from the same comparison
]
present_leakage_cols = [c for c in leakage_candidates if c in df.columns]
print("Leakage columns present in dataset:", present_leakage_cols)

for col in present_leakage_cols:
    print(f"\n--- {col} vs {target_col} ---")
    print(df.groupby(target_col)[col].describe() if df[col].dtype != object
          else pd.crosstab(df[target_col], df[col]))

# Decision: these columns are excluded from X in 02_feature_engineering.py.
# 'Days for shipment (scheduled)' is arguably safe to KEEP (it's known at
# order time), but 'Days for shipping (real)' and 'Delivery Status' are not
# known until after the fact and must be dropped.

# %% Categorical distributions relevant to the business question
for col in ["Shipping Mode", "Order Region", "Category Name", "Customer Segment"]:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

# %% Late delivery rate by shipping mode and region — first look at drivers
rate_by_mode = df.groupby("Shipping Mode")[target_col].mean().sort_values(ascending=False)
print("\nLate delivery rate by shipping mode:\n", rate_by_mode)

rate_by_region = df.groupby("Order Region")[target_col].mean().sort_values(ascending=False)
print("\nLate delivery rate by region:\n", rate_by_region)

fig, ax = plt.subplots(figsize=(8, 4))
rate_by_mode.plot(kind="bar", ax=ax)
ax.set_ylabel("Late delivery rate")
ax.set_title("Late Delivery Rate by Shipping Mode")
plt.savefig(BASE_DIR / "outputs" / "late_rate_by_shipping_mode.png", bbox_inches="tight")
plt.close()

# %% Profitability first look
if "Order Profit Per Order" in df.columns:
    print(df["Order Profit Per Order"].describe())
    loss_making = (df["Order Profit Per Order"] < 0).mean()
    print(f"\nShare of orders that are loss-making: {loss_making:.2%}")

# %% Date parsing (needed for demand forecasting later)
date_cols = [c for c in df.columns if "date" in c.lower()]
print("Date-like columns:", date_cols)
if "order date (DateOrders)" in df.columns:
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    print(df["order_date"].min(), "to", df["order_date"].max())

# %% Save a lightly cleaned copy for the next script
cols_to_drop = mostly_null.index.tolist()
df_clean = df.drop(columns=cols_to_drop)
df_clean.to_parquet(BASE_DIR / "outputs" / "dataco_clean.parquet", index=False)
print(f"\nSaved cleaned dataset: {df_clean.shape} -> {BASE_DIR / 'outputs' / 'dataco_clean.parquet'}")
print(f"Dropped columns: {cols_to_drop}")
