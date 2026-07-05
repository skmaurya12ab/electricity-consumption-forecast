"""
02_load_and_inspect.py

Purpose:
    Load the CSV we generated and look at it from every basic angle:
    shape, column types, sample rows, missing values, and summary stats.
    This is ALWAYS the first real step in any ML project - you must
    understand your raw data before touching it.

Run:
    python 02_load_and_inspect.py
"""

# pandas: our main tool for tabular data (rows & columns)
import pandas as pd

# ---------------------------------------------------------
# STEP 1: Load the CSV file into a pandas DataFrame.
# pd.read_csv(path) reads a comma-separated-values file and
# returns a DataFrame (an in-memory table).
# ---------------------------------------------------------
data_path = "/home/claude/electricity_forecast/data/electricity_consumption.csv"
df = pd.read_csv(data_path)

# ---------------------------------------------------------
# STEP 2: Check the SHAPE of the data.
# df.shape returns a tuple: (number_of_rows, number_of_columns)
# This is the very first thing to check - how big is our dataset?
# ---------------------------------------------------------
print("=" * 60)
print("SHAPE OF THE DATASET (rows, columns):")
print(df.shape)

# ---------------------------------------------------------
# STEP 3: Look at the first 5 rows.
# df.head(n) shows the first n rows. Default n=5.
# This gives us a visual "feel" for what the data looks like.
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("FIRST 5 ROWS:")
print(df.head())

# ---------------------------------------------------------
# STEP 4: Check column DATA TYPES.
# df.dtypes shows what type each column is (number, text, etc.)
# This matters because ML models need numeric input eventually,
# and wrong types (e.g., date read as text) cause bugs later.
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("COLUMN DATA TYPES:")
print(df.dtypes)

# ---------------------------------------------------------
# STEP 5: df.info() gives a compact summary:
# column names, non-null counts, and dtypes all in one view.
# The "non-null count" is a quick way to spot missing values.
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("DATAFRAME INFO:")
df.info()

# ---------------------------------------------------------
# STEP 6: Count MISSING VALUES per column.
# df.isnull() returns True/False for every cell (True = missing).
# .sum() adds up the True values (True counts as 1) per column.
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# ---------------------------------------------------------
# STEP 7: Count DUPLICATE rows.
# df.duplicated() returns True for rows that are exact repeats
# of an earlier row. .sum() counts how many True values exist.
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("NUMBER OF DUPLICATE ROWS:", df.duplicated().sum())

# ---------------------------------------------------------
# STEP 8: df.describe() gives summary statistics
# (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.
# This helps us spot outliers - e.g., a max value that's way too high.
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY STATISTICS (numeric columns):")
print(df.describe())
