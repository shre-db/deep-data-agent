"""Generate the synthetic sample sales dataset (deterministic)."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_DAYS = 365
CHANNELS = ["Paid Search", "Organic", "Email", "Social", "Direct"]
CHANNEL_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
PRODUCTS = ["Widgets", "Gadgets", "Gizmos", "Sprockets"]
REGIONS = ["North", "South", "East", "West"]

dates = pd.date_range("2025-01-01", periods=N_DAYS, freq="D")

rows = []
for date in dates:
    for channel in CHANNELS:
        if rng.random() > 0.25:
            continue  # keep dataset small: ~1-2 rows per day
        month_factor = 1 + 0.6 * (date.dayofyear / N_DAYS)  # upward trend
        weekday = np.where(date.dayofweek < 5, 1.15, 0.8)
        base_orders = {"Paid Search": 60, "Organic": 90, "Email": 35,
                       "Social": 45, "Direct": 25}[channel]
        noise = rng.normal(1.0, 0.15)
        orders = max(0, int(base_orders * month_factor * weekday * noise))
        customers = max(1, int(orders * rng.uniform(0.55, 0.8)))
        conv = float(np.clip(rng.normal(0.045, 0.01) * {"Paid Search": 1.2,
                                                        "Organic": 1.0,
                                                        "Email": 1.3,
                                                        "Social": 0.8,
                                                        "Direct": 1.1}[channel], 0.005, 0.2))
        aov = float(rng.uniform(40, 120) * month_factor)
        revenue = round(orders * aov, 2)
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "channel": channel,
            "product": PRODUCTS[rng.integers(len(PRODUCTS))],
            "region": REGIONS[rng.integers(len(REGIONS))],
            "orders": orders,
            "revenue": revenue,
            "customers": customers,
            "conversion_rate": round(conv, 4),
        })

df = pd.DataFrame(rows)

# Inject a couple of extreme outliers
idx = rng.choice(df.index, size=3, replace=False)
df.loc[idx, "revenue"] = df.loc[idx, "revenue"] * 25

# Inject a few missing values
df.loc[rng.choice(df.index, 7, replace=False), "revenue"] = np.nan
df.loc[rng.choice(df.index, 5, replace=False), "conversion_rate"] = np.nan

# A few duplicate rows (for data-quality checks)
dupes = df.sample(4, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df = df.sample(frac=1.0, random_state=2).reset_index(drop=True)
df.to_csv("data/sample_sales.csv", index=False)
print(f"Wrote data/sample_sales.csv: {df.shape[0]} rows x {df.shape[1]} cols")
