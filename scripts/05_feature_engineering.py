"""
05_feature_engineering.py

Purpose:
    Add a few SIMPLE, easy-to-explain features derived from existing columns.
    We are NOT doing anything fancy here - just features a beginner can
    reason about and justify:
        - month        : captures seasonal effects beyond raw temperature
        - temp_squared  : we saw in EDA that consumption has a U-shape with
                          temperature (high at both hot & cold extremes).
                          A plain linear model can't capture a U-shape, but
                          if we ALSO give it temperature-squared, it can
                          (this is called a "polynomial feature").

Run:
    python 05_feature_engineering.py
"""

import pandas as pd

# ---------------------------------------------------------
# STEP 1: Load the cleaned dataset.
# ---------------------------------------------------------
df = pd.read_csv(
    "/home/claude/electricity_forecast/data/electricity_consumption_clean.csv",
    parse_dates=["date"],
)

# ---------------------------------------------------------
# STEP 2: Extract MONTH from the date.
# .dt is the "datetime accessor" - it unlocks date-specific properties.
# .dt.month gives an integer 1-12.
# ---------------------------------------------------------
df["month"] = df["date"].dt.month

# ---------------------------------------------------------
# STEP 3: Create TEMP_SQUARED feature.
# Why: EDA showed a U-shaped (quadratic) relationship between temperature
# and consumption. temperature_c alone can't express "far from 22°C in
# EITHER direction = more consumption" to a linear model. Squaring it
# does: e.g., (5-22)^2 and (39-22)^2 are both large positive numbers,
# while (22-22)^2 = 0. This directly encodes the U-shape as a feature.
# ---------------------------------------------------------
df["temp_squared"] = df["temperature_c"] ** 2

# ---------------------------------------------------------
# STEP 4: Create a SEASON feature from month (simple bucket/grouping).
# This gives models a coarser, human-intuitive grouping alongside month.
# We map each month number to one of 4 seasons using a dictionary.
# ---------------------------------------------------------
month_to_season = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}
df["season"] = df["month"].map(month_to_season)

# ---------------------------------------------------------
# STEP 5: Preview the new columns.
# ---------------------------------------------------------
print("New columns added: month, temp_squared, season")
print(df[["date", "temperature_c", "temp_squared", "month", "season"]].head())

# ---------------------------------------------------------
# STEP 6: Save the featured dataset for preprocessing.
# ---------------------------------------------------------
featured_path = "/home/claude/electricity_forecast/data/electricity_consumption_featured.csv"
df.to_csv(featured_path, index=False)
print(f"\nFeatured dataset saved to:\n{featured_path}")
print(f"Shape: {df.shape}")
