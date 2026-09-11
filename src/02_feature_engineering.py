"""
02_feature_engineering.py — Build the modeling-ready feature set for the
late-delivery risk classifier.

Loads the cleaned dataset from 01_eda.py, removes the leakage columns
identified there, encodes categoricals, engineers a few date-based features,
and saves a train/test split for 03_late_delivery_model.py.
"""

# %% Imports
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %% Load cleaned data from step 1
df = pd.read_parquet(OUTPUT_DIR / "dataco_clean.parquet")
print(f"Loaded: {df.shape}")

# %% Drop leakage columns
# 'Days for shipping (real)' and 'Delivery Status' are only known AFTER the
# order has shipped — they are direct proxies for the label and must not be
# used as features. 'Days for shipment (scheduled)' is known at order time
# and is safe to keep.
LEAKY_COLS = ["Days for shipping (real)", "Delivery Status"]
df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])

TARGET = "Late_delivery_risk"

# %% Date features
if "order_date" not in df.columns and "order date (DateOrders)" in df.columns:
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")

df["order_month"] = df["order_date"].dt.month
df["order_dayofweek"] = df["order_date"].dt.dayofweek
df["order_year"] = df["order_date"].dt.year

# %% Select feature columns
# Keep it focused: shipping/order attributes known at order time, plus the
# categorical dimensions the business cares about (region, category, segment).
categorical_features = [
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Category Name",
    "Customer Segment",
    "Market",
]
categorical_features = [c for c in categorical_features if c in df.columns]

numeric_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate",
    "order_month",
    "order_dayofweek",
    "order_year",
]
numeric_features = [c for c in numeric_features if c in df.columns]

feature_cols = categorical_features + numeric_features
print("Using features:", feature_cols)

model_df = df[feature_cols + [TARGET]].dropna()
print(f"Rows after dropping NA in modeling columns: {len(model_df)}")

# %% Encode categoricals
# Label encoding is sufficient for tree-based models like XGBoost, which
# don't assume ordinal relationships from the encoding.
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

# %% Train/test split (stratified — the target is imbalanced-ish)
X = model_df[feature_cols]
y = model_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Train target balance:\n", y_train.value_counts(normalize=True))

# %% Save everything for the next step
X_train.to_parquet(OUTPUT_DIR / "X_train.parquet")
X_test.to_parquet(OUTPUT_DIR / "X_test.parquet")
y_train.to_frame(name=TARGET).to_parquet(OUTPUT_DIR / "y_train.parquet", index=False)
y_test.to_frame(name=TARGET).to_parquet(OUTPUT_DIR / "y_test.parquet", index=False)

import joblib
joblib.dump(encoders, OUTPUT_DIR / "categorical_encoders.joblib")
joblib.dump(feature_cols, OUTPUT_DIR / "feature_cols.joblib")

print(f"Saved train/test splits and encoders to {OUTPUT_DIR}")
