"""
05_profitability_analysis.py — Where is margin quietly leaking?

No predictive model here by design — this is a segmentation/exploratory
analysis, which is exactly the kind of "distill insights and propose
recommendations" work the AB InBev JD calls out. Not every business question
needs a model; sometimes a well-cut groupby is the right tool.
"""

# %% Imports
# %% Imports
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# %% Load cleaned data
df = pd.read_parquet(OUTPUT_DIR / "dataco_clean.parquet")

PROFIT_COL = "Order Profit Per Order"
assert PROFIT_COL in df.columns, "Profit column not found — check column name in your dataset version"

# %% Overall profitability snapshot
print(df[PROFIT_COL].describe())
loss_share = (df[PROFIT_COL] < 0).mean()
print(f"\nShare of orders that are loss-making: {loss_share:.2%}")

# %% Profitability by category
by_category = (
    df.groupby("Category Name")[PROFIT_COL]
    .agg(["mean", "sum", "count"])
    .sort_values("mean")
)
print("\nProfitability by category (worst first):\n", by_category.head(10))

fig, ax = plt.subplots(figsize=(10, 5))
by_category["mean"].sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("Mean profit per order")
ax.set_title("Mean Profit per Order by Category")
plt.savefig(OUTPUT_DIR / "profit_by_category.png", bbox_inches="tight")
plt.close()

# %% Profitability by region
by_region = (
    df.groupby("Order Region")[PROFIT_COL]
    .agg(["mean", "sum", "count"])
    .sort_values("mean")
)
print("\nProfitability by region (worst first):\n", by_region)

# %% Profitability by shipping mode
by_shipping = (
    df.groupby("Shipping Mode")[PROFIT_COL]
    .agg(["mean", "sum", "count"])
    .sort_values("mean")
)
print("\nProfitability by shipping mode:\n", by_shipping)

# %% Discount rate vs profitability — is discounting actually working?
if "Order Item Discount Rate" in df.columns:
    df["discount_bucket"] = pd.cut(
        df["Order Item Discount Rate"],
        bins=[-0.01, 0.05, 0.1, 0.2, 0.3, 1.0],
        labels=["0-5%", "5-10%", "10-20%", "20-30%", "30%+"],
    )
    by_discount = df.groupby("discount_bucket")[PROFIT_COL].mean()
    print("\nMean profit by discount bucket:\n", by_discount)

    fig, ax = plt.subplots(figsize=(7, 4))
    by_discount.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean profit per order")
    ax.set_title("Profit vs Discount Level")
    plt.savefig(OUTPUT_DIR / "profit_vs_discount.png", bbox_inches="tight")
    plt.close()

# %% Worst combination: category x region cross-cut
# This is usually the most interesting cut — a category might look fine
# overall but be badly loss-making in one specific region.
cross = (
    df.groupby(["Category Name", "Order Region"])[PROFIT_COL]
    .mean()
    .reset_index()
    .sort_values(PROFIT_COL)
)
print("\nWorst category x region combinations:\n", cross.head(10))

cross.to_csv(OUTPUT_DIR / "profitability_category_region.csv", index=False)

# %% Business recommendation template
worst_cat = by_category.index[0]
worst_cat_margin = by_category.loc[worst_cat, "mean"]
print(f"""
--- BUSINESS RECOMMENDATION TEMPLATE ---
Problem: Not all revenue is equally profitable — some category/region/discount
         combinations are quietly loss-making.
Finding: '{worst_cat}' has the lowest average profit per order (${worst_cat_margin:.2f}).
         [Check by_discount table — does heavier discounting correlate with
         lower or even negative average profit?]
Recommendation: Reassess discount policy or pricing for the worst-performing
         category/region combinations identified above, rather than applying
         a blanket discount strategy across all segments.
""")
