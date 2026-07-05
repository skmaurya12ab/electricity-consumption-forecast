"""
01_generate_dataset.py

Purpose:
    We don't have internet access to a public electricity dataset in this
    environment, so we GENERATE one ourselves. We make it "realistic" by
    building consumption out of real-world-like drivers: temperature,
    weekday/weekend patterns, seasonality, and random noise.

Run:
    python 01_generate_dataset.py
"""

# numpy: lets us create arrays of numbers and do fast math (like sine waves, random noise)
import numpy as np

# pandas: lets us organize our data into a table (DataFrame), like an Excel sheet in Python
import pandas as pd

# ---------------------------------------------------------
# STEP 1: Set a "seed" so that the random numbers generated
# are the SAME every time we run this script.
# This makes our project reproducible (very important in ML!).
# ---------------------------------------------------------
np.random.seed(42)  # 42 is just a common convention, any fixed number works

# ---------------------------------------------------------
# STEP 2: Decide how many days of data we want.
# We'll generate 2 full years of DAILY data.
# ---------------------------------------------------------
n_days = 365 * 2  # 730 rows total, one row = one day

# ---------------------------------------------------------
# STEP 3: Create a column of DATES.
# pd.date_range() generates a sequence of dates.
#   start       -> the first date
#   periods     -> how many dates to generate
#   freq="D"    -> frequency = Daily
# ---------------------------------------------------------
dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

# ---------------------------------------------------------
# STEP 4: Simulate TEMPERATURE (in Celsius) for each day.
#
# Real temperature rises and falls in a yearly cycle (summer/winter).
# We model this using a sine wave:
#     temp = mean_temp + amplitude * sin(2*pi*day_of_year / 365)
#
# np.arange(n_days) -> [0, 1, 2, ..., 729]  (day index)
# np.sin(...)       -> smooth wave between -1 and 1
# We add small random noise so it's not a "perfect" wave (real weather is noisy).
# ---------------------------------------------------------
day_index = np.arange(n_days)  # array: 0,1,2,...,729

# base yearly cycle: peaks in summer (~day 180), dips in winter (~day 0/365)
seasonal_cycle = np.sin((day_index - 80) * (2 * np.pi / 365))

# scale it to a believable temperature range (roughly 5°C to 35°C)
temperature = 20 + 15 * seasonal_cycle

# add random daily noise (real weather isn't a perfect sine wave)
# np.random.normal(loc, scale, size) draws `size` numbers from a
# normal (bell-curve) distribution centered at `loc` with spread `scale`
temperature_noise = np.random.normal(loc=0, scale=2, size=n_days)
temperature = temperature + temperature_noise

# ---------------------------------------------------------
# STEP 5: Simulate HUMIDITY (%) - just another weather feature.
# Roughly inversely related to temperature (hotter = drier, simplified assumption),
# clipped between 20% and 90% so it stays realistic.
# ---------------------------------------------------------
humidity = 70 - 0.8 * (temperature - 20) + np.random.normal(0, 5, n_days)
humidity = np.clip(humidity, 20, 90)  # np.clip forces values to stay in [20, 90]

# ---------------------------------------------------------
# STEP 6: Extract DAY OF WEEK from the dates.
# .dayofweek gives 0=Monday ... 6=Sunday
# ---------------------------------------------------------
day_of_week = dates.dayofweek

# is_weekend: True if Saturday(5) or Sunday(6)
is_weekend = (day_of_week >= 5).astype(int)  # convert True/False to 1/0

# ---------------------------------------------------------
# STEP 7: Build the actual ELECTRICITY CONSUMPTION (kWh) formula.
#
# Logic (all beginner-intuitive):
#   - Base load: every household/building uses some baseline power daily
#   - Temperature effect: consumption goes UP when it's very hot (AC) or
#     very cold (heating). We model this using (temperature - 22)^2, a U-shape.
#   - Weekend effect: slightly lower consumption on weekends (offices closed)
#   - Random noise: real-world randomness (appliance usage varies day to day)
# ---------------------------------------------------------
base_load = 300  # baseline kWh per day

# U-shaped temperature effect: minimum around 22°C (comfortable, less heating/cooling)
temp_effect = 0.8 * (temperature - 22) ** 2

weekend_effect = np.where(is_weekend == 1, -40, 0)  # -40 kWh on weekends

noise = np.random.normal(loc=0, scale=25, size=n_days)  # random daily variation

consumption = base_load + temp_effect + weekend_effect + noise

# Consumption can't be negative in real life, so clip at a sensible minimum
consumption = np.clip(consumption, 150, None)  # None means "no upper limit"

# ---------------------------------------------------------
# STEP 8: Assemble everything into a single pandas DataFrame (our data table).
# pd.DataFrame({column_name: column_data, ...}) builds a table column by column.
# ---------------------------------------------------------
df = pd.DataFrame({
    "date": dates,
    "temperature_c": np.round(temperature, 1),
    "humidity_pct": np.round(humidity, 1),
    "day_of_week": day_of_week,      # 0=Mon ... 6=Sun
    "is_weekend": is_weekend,        # 0 or 1
    "consumption_kwh": np.round(consumption, 1),
})

# ---------------------------------------------------------
# STEP 9: Intentionally introduce a FEW realistic messiness issues,
# because real datasets are never perfectly clean, and you asked to
# practice data cleaning too.
# ---------------------------------------------------------

# 9a: A few missing values in temperature (sensor failure simulation)
missing_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[missing_idx, "temperature_c"] = np.nan

# 9b: A few duplicate rows (data entry error simulation)
duplicate_rows = df.sample(3, random_state=1)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 9c: One obvious outlier/typo in consumption (e.g., a data entry mistake)
df.loc[10, "consumption_kwh"] = 9999.0

# ---------------------------------------------------------
# STEP 10: Save to CSV so our next script can load it.
# index=False means "don't write pandas' internal row numbers as a column"
# ---------------------------------------------------------
output_path = "/home/claude/electricity_forecast/data/electricity_consumption.csv"
df.to_csv(output_path, index=False)

print(f"Dataset generated with {len(df)} rows and saved to:\n{output_path}")
