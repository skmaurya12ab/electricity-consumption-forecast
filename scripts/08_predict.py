"""
08_predict.py

Purpose:
    Load the best trained model + scaler, ask the user for simple inputs
    (temperature, humidity, date info), build the exact same features
    used during training, and print a predicted electricity consumption.

Run:
    python 08_predict.py
"""

import joblib
import numpy as np
import pandas as pd

data_dir = "/home/claude/electricity_forecast/data"
models_dir = "/home/claude/electricity_forecast/outputs/models"

# ---------------------------------------------------------
# STEP 1: Load the saved scaler, feature column order, and best model name.
# We MUST use the same scaler and column order as training, otherwise
# predictions will be meaningless (the model expects inputs transformed
# and arranged exactly the same way it was trained on).
# ---------------------------------------------------------
scaler = joblib.load(f"{data_dir}/scaler.pkl")
feature_cols = joblib.load(f"{data_dir}/feature_cols.pkl")
best_model_name = joblib.load(f"{models_dir}/best_model_name.pkl")

model_filename = best_model_name.lower().replace(" ", "_") + ".pkl"
model = joblib.load(f"{models_dir}/{model_filename}")

print(f"Loaded model: {best_model_name}")
print("Enter the following details to get a predicted electricity consumption.\n")


# ---------------------------------------------------------
# STEP 2: A small helper function to safely read numeric input from the
# user, re-asking if they type something invalid (not a number).
# ---------------------------------------------------------
def get_float_input(prompt):
    while True:
        raw_value = input(prompt)
        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a valid number.")


def get_int_input(prompt, valid_range):
    while True:
        raw_value = input(prompt)
        try:
            value = int(raw_value)
            if value in valid_range:
                return value
            print(f"Please enter a value in range {valid_range.start}-{valid_range.stop - 1}.")
        except ValueError:
            print("Please enter a valid whole number.")


# ---------------------------------------------------------
# STEP 3: Collect raw inputs from the user (the simple, real-world values
# they'd actually know - NOT the engineered features like temp_squared,
# which we compute automatically).
# ---------------------------------------------------------
temperature_c = get_float_input("Temperature today (°C), e.g. 28.5: ")
humidity_pct = get_float_input("Humidity (%), e.g. 65: ")
month = get_int_input("Month (1-12): ", range(1, 13))
day_of_week = get_int_input("Day of week (0=Mon ... 6=Sun): ", range(0, 7))

is_weekend = 1 if day_of_week >= 5 else 0

month_to_season = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}
season = month_to_season[month]

# ---------------------------------------------------------
# STEP 4: Rebuild the SAME engineered features used in training.
# ---------------------------------------------------------
temp_squared = temperature_c ** 2

# Build a single-row dictionary matching the training feature set.
# We start all season_* columns at 0, then set the matching one to 1
# (mirroring the one-hot encoding done during preprocessing).
input_dict = {
    "temperature_c": temperature_c,
    "temp_squared": temp_squared,
    "humidity_pct": humidity_pct,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "month": month,
    "season_spring": 1 if season == "spring" else 0,
    "season_summer": 1 if season == "summer" else 0,
    "season_winter": 1 if season == "winter" else 0,
}

# ---------------------------------------------------------
# STEP 5: Arrange into a DataFrame with EXACTLY the same column order
# as feature_cols (loaded from training). This ordering matters because
# the scaler and model expect columns in a specific position.
# ---------------------------------------------------------
input_df = pd.DataFrame([input_dict])[feature_cols]

# ---------------------------------------------------------
# STEP 6: Scale the input using the SAME scaler fitted during training
# (scaler.transform, NOT fit_transform - we must not re-learn new
# mean/std values here).
# ---------------------------------------------------------
input_scaled = scaler.transform(input_df)

# Wrap back into a DataFrame with column names - the model was trained on
# a DataFrame (not a raw array), so this keeps scikit-learn's internal
# feature-name check happy and avoids a cosmetic warning.
input_scaled_df = pd.DataFrame(input_scaled, columns=feature_cols)

# ---------------------------------------------------------
# STEP 7: Predict!
# ---------------------------------------------------------
predicted_consumption = model.predict(input_scaled_df)[0]

print("\n" + "=" * 50)
print(f"Predicted Electricity Consumption: {predicted_consumption:.1f} kWh")
print("=" * 50)
