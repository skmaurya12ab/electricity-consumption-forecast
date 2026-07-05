"""
06_preprocessing.py

Purpose:
    Prepare data for modeling:
        1. One-hot encode the 'season' text column into numeric columns
        2. Select our final feature set (X) and target (y)
        3. Split into train/test sets
        4. Scale numeric features (important for Linear Regression;
           harmless for tree models)
        5. Save everything (train/test arrays + the scaler) so the
           training script can just load and use them.

Run:
    python 06_preprocessing.py
"""

import pandas as pd
import joblib  # used to save/load Python objects (like scikit-learn models & scalers)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# STEP 1: Load the featured dataset.
# ---------------------------------------------------------
df = pd.read_csv(
    "/home/claude/electricity_forecast/data/electricity_consumption_featured.csv",
    parse_dates=["date"],
)

# ---------------------------------------------------------
# STEP 2: One-hot encode 'season'.
#
# ML models need NUMBERS, not text like "winter"/"summer". One-hot
# encoding turns one text column into several 0/1 columns, one per
# category. pd.get_dummies() does this automatically.
#
# drop_first=True drops one category's column (e.g., 'season_fall') to
# avoid redundant information - if the other 3 season columns are all 0,
# we already know it must be fall. This avoids a mild redundancy issue
# called "multicollinearity" in Linear Regression.
# ---------------------------------------------------------
df = pd.get_dummies(df, columns=["season"], drop_first=True)
# This creates columns like: season_spring, season_summer, season_winter
# (season_fall becomes the "baseline" - all-zero - category)

# ---------------------------------------------------------
# STEP 3: Define our FEATURE columns (X) and TARGET column (y).
#
# We exclude 'date' (not a number, and day-to-day date itself isn't
# predictive - the DERIVED features like month/day_of_week are what matter)
# and 'consumption_kwh' (that's our target, not an input).
# ---------------------------------------------------------
target_col = "consumption_kwh"

feature_cols = [
    "temperature_c",
    "temp_squared",
    "humidity_pct",
    "day_of_week",
    "is_weekend",
    "month",
] + [col for col in df.columns if col.startswith("season_")]  # adds the one-hot season columns

X = df[feature_cols]
y = df[target_col]

print("Feature columns used for modeling:")
print(feature_cols)

# ---------------------------------------------------------
# STEP 4: Train/Test split.
#
# train_test_split() randomly divides data into a training portion
# (used to fit the model) and a testing portion (used to check how well
# it generalizes to UNSEEN data).
#   test_size=0.2   -> 20% of data reserved for testing, 80% for training
#   random_state=42 -> fixes the randomness so the split is reproducible
#   shuffle=False    -> IMPORTANT for time-series-like data: we keep
#                       chronological order, training on the past and
#                       testing on the most recent period, which is more
#                       realistic for a forecasting problem than random
#                       shuffling.
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")

# ---------------------------------------------------------
# STEP 5: Scale numeric features using StandardScaler.
#
# StandardScaler transforms each feature to have mean=0 and std=1:
#     scaled_value = (value - mean) / standard_deviation
#
# Why scale: Linear Regression (and many algorithms) can be sensitive to
# features being on very different scales (e.g., humidity 20-90 vs
# temp_squared up to ~1600). Tree-based models (Decision Tree, Random
# Forest) don't NEED scaling, but scaling never hurts them, so we scale
# once and use the same data for all 3 models for a fair comparison.
#
# IMPORTANT: We fit the scaler ONLY on training data (.fit_transform),
# then apply the SAME transformation to test data (.transform only,
# no re-fitting). This prevents "data leakage" - the test set must
# remain unseen information during training.
# ---------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # learn mean/std from train, then transform
X_test_scaled = scaler.transform(X_test)        # apply the SAME mean/std to test data

# Convert back to DataFrames (with original column names) for readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

# ---------------------------------------------------------
# STEP 6: Save everything for the training script.
# joblib.dump(object, path) serializes a Python object to disk.
# We save: scaled train/test sets, targets, the scaler itself, and the
# feature column order (so predict.py can rebuild inputs correctly later).
# ---------------------------------------------------------
out_dir = "/home/claude/electricity_forecast/data"

joblib.dump(X_train_scaled, f"{out_dir}/X_train.pkl")
joblib.dump(X_test_scaled, f"{out_dir}/X_test.pkl")
joblib.dump(y_train, f"{out_dir}/y_train.pkl")
joblib.dump(y_test, f"{out_dir}/y_test.pkl")
joblib.dump(scaler, f"{out_dir}/scaler.pkl")
joblib.dump(feature_cols, f"{out_dir}/feature_cols.pkl")

print("\nSaved: X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl, scaler.pkl, feature_cols.pkl")
