"""
03_data_cleaning.py

Purpose:
    Fix the 3 issues we found during inspection:
        1. 'date' column is text -> convert to real datetime
        2. 8 missing values in 'temperature_c' -> fill sensibly
        3. 3 duplicate rows -> remove
        4. 1 extreme outlier in 'consumption_kwh' (9999) -> fix

Run:
    python 03_data_cleaning.py
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------
# STEP 1: Load the raw data we inspected earlier.
# ---------------------------------------------------------
raw_path = "/home/claude/electricity_forecast/data/electricity_consumption.csv"
df = pd.read_csv(raw_path)

print(f"Rows before cleaning: {len(df)}")

# ---------------------------------------------------------
# STEP 2: Convert 'date' column from text to real datetime objects.
# pd.to_datetime() parses date-like strings into pandas Timestamp objects.
# This lets us later extract month, season, etc. and sort properly.
# ---------------------------------------------------------
df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------------
# STEP 3: Remove duplicate rows.
# df.drop_duplicates() removes rows that are IDENTICAL across all columns.
# We reset the index afterward so row numbers stay clean (0,1,2,...).
# ---------------------------------------------------------
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate row(s).")

# ---------------------------------------------------------
# STEP 4: Handle missing temperature values.
#
# Since temperature changes smoothly day-to-day (it doesn't jump randomly),
# a good beginner-friendly fix is to fill missing values using
# INTERPOLATION - estimating the missing value from neighboring days.
#
# df.sort_values() first ensures rows are in date order (important for
# interpolation to make sense).
# .interpolate(method="linear") draws a straight line between the
# nearest known values before and after the gap, and fills it in.
# ---------------------------------------------------------
df = df.sort_values("date").reset_index(drop=True)

missing_before = df["temperature_c"].isnull().sum()
df["temperature_c"] = df["temperature_c"].interpolate(method="linear")
missing_after = df["temperature_c"].isnull().sum()
print(f"Filled {missing_before - missing_after} missing temperature value(s) via interpolation.")

# ---------------------------------------------------------
# STEP 5: Fix the extreme outlier in 'consumption_kwh'.
#
# Approach: define a reasonable valid range using domain knowledge
# (we know from describe() that normal values are roughly 150-700).
# Any value far outside this is almost certainly a data entry error,
# not a real reading. We REPLACE such outliers with the median value
# of the column (a simple, robust beginner technique).
#
# df["col"].median() -> the middle value when sorted (robust to outliers,
# unlike mean, which outliers can drag up/down).
# ---------------------------------------------------------
median_consumption = df["consumption_kwh"].median()

# Boolean mask: True where the value is unrealistically high
outlier_mask = df["consumption_kwh"] > 1000  # normal daily values never reach this

n_outliers = outlier_mask.sum()
df.loc[outlier_mask, "consumption_kwh"] = median_consumption
print(f"Replaced {n_outliers} outlier value(s) in consumption_kwh with median ({median_consumption:.1f}).")

# ---------------------------------------------------------
# STEP 6: Final sanity check - confirm no missing values or duplicates remain.
# ---------------------------------------------------------
print("\nFinal missing values per column:")
print(df.isnull().sum())
print(f"\nFinal duplicate rows: {df.duplicated().sum()}")
print(f"Final row count: {len(df)}")

# ---------------------------------------------------------
# STEP 7: Save the cleaned dataset for the next step (EDA).
# ---------------------------------------------------------
clean_path = "/home/claude/electricity_forecast/data/electricity_consumption_clean.csv"
df.to_csv(clean_path, index=False)
print(f"\nCleaned dataset saved to:\n{clean_path}")
