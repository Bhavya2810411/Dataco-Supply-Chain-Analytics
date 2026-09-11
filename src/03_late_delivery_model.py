"""
03_late_delivery_model.py — Train and explain the late-delivery risk model.

Trains an XGBoost classifier on the leakage-free feature set from
02_feature_engineering.py, evaluates it properly (not just accuracy — this
dataset's target isn't wildly imbalanced, but precision/recall/ROC-AUC still
matter more than accuracy for a business decision), and uses SHAP to turn
"the model predicts X" into "here's WHY, and here's what to do about it."
"""

# %% Imports
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    RocCurveDisplay,
)

# %% Load train/test data
X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet").iloc[:, 0]
y_test = pd.read_parquet(OUTPUT_DIR / "y_test.parquet").iloc[:, 0]
feature_cols = joblib.load(OUTPUT_DIR / "feature_cols.joblib")

# %% Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

# %% Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification report:\n", classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title(f"Late Delivery Risk — ROC Curve (AUC={auc:.3f})")
plt.savefig(OUTPUT_DIR / "roc_curve.png", bbox_inches="tight")
plt.close()

# %% Feature importance (quick view before SHAP)
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nXGBoost feature importances:\n", importances)

# %% SHAP explainability — the "why", not just the "what"
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(OUTPUT_DIR / "shap_summary.png", bbox_inches="tight")
plt.close()

# %% Translate SHAP into a business-facing insight
# Identify the single most influential feature and, if it's categorical,
# which category value pushes risk up the most.
top_feature = importances.index[0]
print(f"\nTop driver of late-delivery risk: {top_feature}")

if top_feature in ["Shipping Mode", "Order Region", "Order Country", "Category Name", "Customer Segment", "Market"]:
    encoders = joblib.load(OUTPUT_DIR / "categorical_encoders.joblib")
    le = encoders[top_feature]
    # Map encoded values back to labels and compare mean late-delivery rate
    tmp = X_test.copy()
    tmp["actual"] = y_test.values
    tmp[top_feature + "_label"] = le.inverse_transform(tmp[top_feature].astype(int))
    rate_by_cat = tmp.groupby(top_feature + "_label")["actual"].mean().sort_values(ascending=False)
    print(f"\nLate delivery rate by {top_feature}:\n", rate_by_cat)
    worst = rate_by_cat.index[0]
    best = rate_by_cat.index[-1]
    print(
        f"\n>>> INSIGHT: '{worst}' has a late-delivery rate of {rate_by_cat.iloc[0]:.1%}, "
        f"vs. {rate_by_cat.iloc[-1]:.1%} for '{best}'. "
        f"This is the single largest driver the model identifies."
    )

# %% Save the model
joblib.dump(model, OUTPUT_DIR / "late_delivery_xgb_model.joblib")
print(f"\nModel saved to {OUTPUT_DIR / 'late_delivery_xgb_model.joblib'}")

# %% Business recommendation template (fill in with the printed numbers above)
print("""
--- BUSINESS RECOMMENDATION TEMPLATE ---
Problem: Late deliveries hurt customer satisfaction and violate SLAs.
Finding: [TOP FEATURE] is the strongest driver of late-delivery risk;
         specifically, [WORST CATEGORY] shows a late-delivery rate of [X%],
         roughly [Nx] higher than [BEST CATEGORY].
Recommendation: Prioritize [WORST CATEGORY] for either a shipping-mode change,
         carrier renegotiation, or SLA adjustment. Flagging orders that match
         this high-risk profile at order time (using this model's probability
         score) lets ops teams proactively expedite or communicate delays
         before they happen, rather than reacting after the fact.
""")
