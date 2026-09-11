"""
06_sustainability_angle.py — Can we ship greener without hurting delivery SLA?

The AB InBev JD explicitly calls out "sustainability initiatives" as part of
the SET role. This script builds a simple, honest proxy analysis (not a
claim of real carbon accounting) to show that angle in the project.

Approach: shipping mode is a reasonable proxy for emissions intensity per
order (air/express modes are typically far more carbon-intensive per unit
than standard/ground). We look at whether some orders are being shipped via
higher-emission modes despite having flexible delivery windows, and whether a
mode shift is possible without increasing late-delivery risk.
"""

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

# %% Assign a relative emissions-intensity proxy per shipping mode
# NOTE: these are illustrative relative weights, not verified emissions
# factors — call this out explicitly in the write-up. The point of the
# analysis is the METHOD (trading off service level vs. emissions proxy),
# which is transferable once real emissions factors are available.
EMISSIONS_PROXY = {
    "Same Day": 5,
    "First Class": 4,
    "Second Class": 2,
    "Standard Class": 1,
}
df["emissions_proxy"] = df["Shipping Mode"].map(EMISSIONS_PROXY)

print(df.groupby("Shipping Mode")["emissions_proxy"].first())

# %% Volume and late-delivery rate by shipping mode (for the trade-off view)
mode_summary = df.groupby("Shipping Mode").agg(
    order_count=("Late_delivery_risk", "count"),
    late_rate=("Late_delivery_risk", "mean"),
    emissions_proxy=("emissions_proxy", "first"),
).sort_values("emissions_proxy", ascending=False)
print("\nShipping mode summary:\n", mode_summary)

# %% Estimate potential impact of shifting a share of high-emission,
# low-urgency orders to a lower-emission mode.
# "Low urgency" proxy: scheduled shipping days >= 3 (i.e., the order already
# has a flexible window, so there's less reason to use an express mode).
share = 0

if "Days for shipment (scheduled)" in df.columns:
    flexible_high_emission = df[
        (df["Shipping Mode"].isin(["Same Day", "First Class"]))
        & (df["Days for shipment (scheduled)"] >= 3)
    ]
    share = len(flexible_high_emission) / len(df)
    print(
        f"\n{len(flexible_high_emission):,} orders ({share:.1%} of all orders) "
        f"use a high-emission shipping mode despite having a >=3 day scheduled "
        f"window — these are the best candidates for a mode shift with minimal "
        f"customer impact."
    )

# %% Chart: emissions proxy vs late-delivery rate by mode (the trade-off)
fig, ax1 = plt.subplots(figsize=(8, 5))
x = range(len(mode_summary))
ax1.bar(x, mode_summary["emissions_proxy"], color="tab:red", alpha=0.6, label="Emissions proxy")
ax1.set_ylabel("Emissions proxy (relative)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(mode_summary.index, rotation=20)

ax2 = ax1.twinx()
ax2.plot(x, mode_summary["late_rate"], color="tab:blue", marker="o", label="Late delivery rate")
ax2.set_ylabel("Late delivery rate")

fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
plt.title("Shipping Mode: Emissions Proxy vs. Late Delivery Rate")
plt.savefig(OUTPUT_DIR / "sustainability_tradeoff.png", bbox_inches="tight")
plt.close()

# %% Business recommendation template
print(f"""
--- BUSINESS RECOMMENDATION TEMPLATE ---
Problem: Higher-emission shipping modes are sometimes used even when the
         order doesn't require express handling.
Finding: {share:.1%} of orders use a high-emission mode (Same Day/First
         Class) despite having a scheduled window of 3+ days — meaning a
         lower-emission mode could likely be used with no SLA impact.
Recommendation: Introduce an order-time rule (or a model flag) that defaults
         flexible-window orders to Standard/Second Class shipping, reducing
         emissions intensity without raising late-delivery risk. This
         directly supports a sustainability KPI without trading off service
         level — the trade-off chart makes this case visually.
Caveat to state upfront in interview: emissions weights here are illustrative
         proxies, not verified carbon-accounting figures — the framework is
         the contribution, and would use real per-mode emissions factors in
         production.
""")
