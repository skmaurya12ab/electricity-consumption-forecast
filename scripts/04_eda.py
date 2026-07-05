"""
04_eda.py

Purpose:
    Explore the CLEANED data visually to understand patterns before
    building features/models. We save each plot as a PNG file so you
    can view them even outside a notebook.

Run:
    python 04_eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # "Agg" backend renders to files instead of a screen window
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# STEP 1: Load the cleaned dataset.
# ---------------------------------------------------------
df = pd.read_csv(
    "/home/claude/electricity_forecast/data/electricity_consumption_clean.csv",
    parse_dates=["date"],  # tells pandas to parse this column as dates directly
)

plots_dir = "/home/claude/electricity_forecast/outputs/plots"

# seaborn.set_style() changes the visual theme of all following plots
sns.set_style("whitegrid")

# ---------------------------------------------------------
# PLOT 1: Consumption over time (line plot).
# This shows us the overall trend and any obvious seasonal wave pattern.
# ---------------------------------------------------------
plt.figure(figsize=(12, 4))  # figsize=(width_inches, height_inches)
plt.plot(df["date"], df["consumption_kwh"], color="steelblue", linewidth=1)
plt.title("Daily Electricity Consumption Over Time")
plt.xlabel("Date")
plt.ylabel("Consumption (kWh)")
plt.tight_layout()  # prevents labels from being cut off
plt.savefig(f"{plots_dir}/01_consumption_over_time.png")  # save to file
plt.close()  # close the figure to free memory before making the next one

# ---------------------------------------------------------
# PLOT 2: Distribution of consumption (histogram).
# Shows us the overall shape (is it normal/bell-shaped, skewed, etc.)
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.histplot(df["consumption_kwh"], bins=30, kde=True, color="steelblue")
plt.title("Distribution of Daily Electricity Consumption")
plt.xlabel("Consumption (kWh)")
plt.tight_layout()
plt.savefig(f"{plots_dir}/02_consumption_distribution.png")
plt.close()

# ---------------------------------------------------------
# PLOT 3: Consumption vs Temperature (scatter plot).
# We expect a U-shape: high consumption at both cold and hot extremes.
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="temperature_c", y="consumption_kwh", alpha=0.5, color="darkorange")
plt.title("Electricity Consumption vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Consumption (kWh)")
plt.tight_layout()
plt.savefig(f"{plots_dir}/03_consumption_vs_temperature.png")
plt.close()

# ---------------------------------------------------------
# PLOT 4: Average consumption by day of week (bar plot).
# day_of_week: 0=Monday ... 6=Sunday
# ---------------------------------------------------------
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
avg_by_day = df.groupby("day_of_week")["consumption_kwh"].mean()

plt.figure(figsize=(7, 5))
sns.barplot(x=day_names, y=avg_by_day.values, color="mediumseagreen")
plt.title("Average Consumption by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Consumption (kWh)")
plt.tight_layout()
plt.savefig(f"{plots_dir}/04_avg_consumption_by_day.png")
plt.close()

# ---------------------------------------------------------
# PLOT 5: Correlation heatmap.
# df.corr() computes the Pearson correlation coefficient between
# every pair of numeric columns (ranges from -1 to +1).
# sns.heatmap() visualizes this matrix with colors + numbers.
# ---------------------------------------------------------
numeric_cols = ["temperature_c", "humidity_pct", "day_of_week", "is_weekend", "consumption_kwh"]
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{plots_dir}/05_correlation_heatmap.png")
plt.close()

print("EDA complete. 5 plots saved to:", plots_dir)
print("\nCorrelation of each feature with consumption_kwh:")
print(corr_matrix["consumption_kwh"].sort_values(ascending=False))
