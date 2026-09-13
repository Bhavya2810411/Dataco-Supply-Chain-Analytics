"""
app.py — Streamlit dashboard for the DataCo Supply Chain Analytics project.

Place this file in the ROOT of your `dataco-supply-chain-analytics` project
(next to `src/`, `outputs/`, `data/`) and run:

    streamlit run app.py

Requires that you've already run src/01_eda.py through src/06_sustainability_angle.py
at least once, since this dashboard reads their saved outputs (model, encoders,
charts) from the outputs/ folder rather than recomputing anything.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Supply Chain Analytics", layout="wide")

OUTPUTS = Path("outputs")

# ---------------------------------------------------------------------------
# Cached loaders — the model/encoders/charts don't change during a session,
# so load them once instead of on every interaction.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    model = joblib.load(OUTPUTS / "late_delivery_xgb_model.joblib")
    encoders = joblib.load(OUTPUTS / "categorical_encoders.joblib")
    feature_cols = joblib.load(OUTPUTS / "feature_cols.joblib")
    return model, encoders, feature_cols


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def missing_outputs_warning():
    st.error(
        "Couldn't find the saved model/encoders in `outputs/`. "
        "Run `src/01_eda.py` through `src/03_late_delivery_model.py` at least once first."
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Supply Chain Analytics")
page = st.sidebar.radio(
    "View",
    ["Predict a Shipment", "Late Delivery Insights", "Demand Forecast", "Profitability", "Sustainability"],
)

st.sidebar.markdown("---")
st.sidebar.caption("DataCo Smart Supply Chain Dataset — 180,519 orders")

# ---------------------------------------------------------------------------
# Page 1: Interactive prediction
# ---------------------------------------------------------------------------
if page == "Predict a Shipment":
    st.title("Predict Late Delivery Risk")
    st.write("Enter a hypothetical order's details to see the model's predicted risk of a late delivery, and why.")

    model_path = OUTPUTS / "late_delivery_xgb_model.joblib"
    if not model_path.exists():
        missing_outputs_warning()
        st.stop()

    model, encoders, feature_cols = load_model_bundle()

    col1, col2, col3 = st.columns(3)
    inputs = {}

    with col1:
        if "Shipping Mode" in encoders:
            inputs["Shipping Mode"] = st.selectbox("Shipping Mode", encoders["Shipping Mode"].classes_)
        if "Order Region" in encoders:
            inputs["Order Region"] = st.selectbox("Order Region", encoders["Order Region"].classes_)

    with col2:
        if "Category Name" in encoders:
            inputs["Category Name"] = st.selectbox("Category Name", encoders["Category Name"].classes_)
        if "Customer Segment" in encoders:
            inputs["Customer Segment"] = st.selectbox("Customer Segment", encoders["Customer Segment"].classes_)

    with col3:
        if "Order Country" in encoders:
            inputs["Order Country"] = st.selectbox("Order Country", encoders["Order Country"].classes_)
        if "Market" in encoders:
            inputs["Market"] = st.selectbox("Market", encoders["Market"].classes_)

    st.markdown("#### Order details")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scheduled_days = st.slider("Days for shipment (scheduled)", 0, 4, 2)
    with c2:
        quantity = st.number_input("Order Item Quantity", min_value=1, max_value=10, value=1)
    with c3:
        sales = st.number_input("Sales ($)", min_value=1.0, max_value=2000.0, value=200.0)
    with c4:
        discount_rate = st.slider("Discount Rate", 0.0, 0.5, 0.1)

    order_month = st.slider("Order Month", 1, 12, 6)
    order_dayofweek = st.slider("Order Day of Week (0=Mon)", 0, 6, 2)
    order_year = st.selectbox("Order Year", [2015, 2016, 2017, 2018], index=2)

    if st.button("Predict", type="primary"):
        row = {}
        for col in feature_cols:
            if col in encoders:
                le = encoders[col]
                row[col] = le.transform([inputs.get(col, le.classes_[0])])[0]
            elif col == "Days for shipment (scheduled)":
                row[col] = scheduled_days
            elif col == "Order Item Quantity":
                row[col] = quantity
            elif col == "Sales":
                row[col] = sales
            elif col == "Order Item Discount Rate":
                row[col] = discount_rate
            elif col == "order_month":
                row[col] = order_month
            elif col == "order_dayofweek":
                row[col] = order_dayofweek
            elif col == "order_year":
                row[col] = order_year
            else:
                row[col] = 0

        X_input = pd.DataFrame([row])[feature_cols]
        proba = model.predict_proba(X_input)[0, 1]

        st.markdown("---")
        risk_col, gauge_col = st.columns([1, 2])
        with risk_col:
            st.metric("Predicted Late Delivery Risk", f"{proba:.1%}")
            if proba >= 0.5:
                st.warning("High risk of late delivery")
            else:
                st.success("Lower risk of late delivery")

        with gauge_col:
            explainer = get_explainer(model)
            shap_values = explainer.shap_values(X_input)
            st.write("**Why the model predicted this:**")
            fig, ax = plt.subplots(figsize=(8, 3))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value, shap_values[0], feature_names=feature_cols, show=False
            )
            st.pyplot(fig)

# ---------------------------------------------------------------------------
# Page 2: Late delivery model insights (static charts from 03_late_delivery_model.py)
# ---------------------------------------------------------------------------
elif page == "Late Delivery Insights":
    st.title("Late Delivery Model — Results")
    st.metric("ROC-AUC", "0.75")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Late delivery rate by shipping mode**")
        img = OUTPUTS / "late_rate_by_shipping_mode.png"
        if img.exists():
            st.image(str(img))
        else:
            st.info("Chart not found — run 01_eda.py")

    with c2:
        st.write("**ROC curve**")
        img = OUTPUTS / "roc_curve.png"
        if img.exists():
            st.image(str(img))
        else:
            st.info("Chart not found — run 03_late_delivery_model.py")

    st.write("**SHAP summary — what drives the model's predictions**")
    img = OUTPUTS / "shap_summary.png"
    if img.exists():
        st.image(str(img))
    else:
        st.info("Chart not found — run 03_late_delivery_model.py")

    st.info(
        "Key insight: Shipping Mode accounts for ~70% of feature importance. "
        "First Class orders are late 95.2% of the time vs. 37.9% for Standard Class."
    )

# ---------------------------------------------------------------------------
# Page 3: Demand forecast
# ---------------------------------------------------------------------------
elif page == "Demand Forecast":
    st.title("Demand Forecasting")

    img = OUTPUTS / "weekly_demand_raw.png"
    if img.exists():
        st.write("**Weekly demand — top category**")
        st.image(str(img))

    img = OUTPUTS / "demand_forecast_comparison.png"
    if img.exists():
        st.write("**Forecast vs. actual (ARIMA)**")
        st.image(str(img))

    csv = OUTPUTS / "forecast_model_comparison.csv"
    if csv.exists():
        st.write("**Model comparison (MAE, lower is better)**")
        st.dataframe(pd.read_csv(csv), use_container_width=True)
    else:
        st.info("Comparison table not found — run 04_demand_forecast.py")

# ---------------------------------------------------------------------------
# Page 4: Profitability
# ---------------------------------------------------------------------------
elif page == "Profitability":
    st.title("Profitability Segmentation")
    st.metric("Share of loss-making orders", "18.71%")

    c1, c2 = st.columns(2)
    with c1:
        img = OUTPUTS / "profit_by_category.png"
        if img.exists():
            st.write("**Mean profit per order by category**")
            st.image(str(img))

    with c2:
        img = OUTPUTS / "profit_vs_discount.png"
        if img.exists():
            st.write("**Profit vs. discount level**")
            st.image(str(img))

    csv = OUTPUTS / "profitability_category_region.csv"
    if csv.exists():
        st.write("**Worst category x region combinations**")
        st.dataframe(pd.read_csv(csv).head(10), use_container_width=True)
    else:
        st.info("Table not found — run 05_profitability_analysis.py")

# ---------------------------------------------------------------------------
# Page 5: Sustainability
# ---------------------------------------------------------------------------
elif page == "Sustainability":
    st.title("Sustainability Trade-off")

    img = OUTPUTS / "sustainability_tradeoff.png"
    if img.exists():
        st.image(str(img))
    else:
        st.info("Chart not found — run 06_sustainability_angle.py")

    st.info(
        "The hypothesis (shift flexible-window, high-emission orders to a lower-emission mode) "
        "returned 0 qualifying orders — scheduled shipping days don't vary independently of "
        "shipping mode in this dataset. This reinforces the delivery-risk finding: First Class "
        "isn't a flexible premium option, it's tied to a near-guaranteed delay."
    )
