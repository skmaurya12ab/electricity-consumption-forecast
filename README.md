# Electricity Consumption Forecast

A beginner-friendly, end-to-end machine learning project that predicts
daily electricity consumption (in kWh) from weather and calendar features,
using only NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

## Why a synthetic dataset?

This project generates its own realistic 2-year daily dataset (`01_generate_dataset.py`)
instead of downloading one, so that:
- You know exactly what patterns exist in the data (temperature effect,
  weekend effect, seasonality) and can verify the model actually learns them.
- The project runs fully offline, with no dataset download step to break.
- It includes realistic messiness on purpose (missing values, duplicate
  rows, one outlier) so you get real data-cleaning practice.

The consumption formula (see comments in `01_generate_dataset.py`) is:
`consumption = base_load + temperature_effect (U-shaped) + weekend_effect + random_noise`

## Project structure

```
electricity_forecast/
├── data/                                  # generated at runtime
│   ├── electricity_consumption.csv               (raw, messy)
│   ├── electricity_consumption_clean.csv         (after cleaning)
│   ├── electricity_consumption_featured.csv      (after feature engineering)
│   ├── X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
├── outputs/                               # generated at runtime
│   ├── plots/                             # all PNG charts from EDA + evaluation
│   ├── models/                            # trained model .pkl files
│   └── model_comparison.csv               # MAE / RMSE / R2 for each model
├── scripts/
│   ├── 01_generate_dataset.py             # builds the synthetic dataset
│   ├── 02_load_and_inspect.py             # shape, dtypes, missing values, describe()
│   ├── 03_data_cleaning.py                # fix dtypes, missing values, duplicates, outlier
│   ├── 04_eda.py                          # 5 exploratory plots + correlation check
│   ├── 05_feature_engineering.py          # month, temp_squared, season
│   ├── 06_preprocessing.py                # one-hot encoding, scaling, train/test split
│   ├── 07_train_and_evaluate.py           # trains 3 models, compares, saves everything
│   ├── 08_predict.py                      # interactive CLI prediction
│   └── run_all.py                         # runs steps 01-07 in order
└── requirements.txt
```

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the full pipeline (from the `scripts/` folder):
   ```
   cd scripts
   python run_all.py
   ```
   This regenerates the dataset, cleans it, runs EDA, engineers features,
   preprocesses, trains all 3 models, and saves all outputs and plots.

3. Try the interactive predictor:
   ```
   python 08_predict.py
   ```
   You'll be asked for temperature, humidity, month, and day of week, and
   it will print a predicted electricity consumption in kWh.

   You can also run any individual step script on its own at any time
   (e.g., `python 04_eda.py`) as long as the earlier steps have been run
   at least once, since each step reads the file(s) saved by the one
   before it.

## Models compared

| Model              | What it does (in plain terms) |
|---------------------|-------------------------------|
| Linear Regression   | Fits a weighted straight-line relationship between inputs and consumption |
| Decision Tree       | Learns a series of if/else questions (e.g., "is temperature > 25?") to split the data into groups with similar consumption |
| Random Forest       | Trains many different Decision Trees on random subsets of data/features, then averages their predictions for a more robust result |

Each is evaluated using:
- **MAE (Mean Absolute Error)** — average error in kWh, easy to interpret directly
- **RMSE (Root Mean Squared Error)** — like MAE but penalizes large errors more
- **R² (R-squared)** — fraction of variance in consumption explained by the model (closer to 1 = better)

In this project, **Linear Regression** tends to win, because the engineered
`temp_squared` feature already captures the true U-shaped relationship
between temperature and consumption — a good example of how good feature
engineering can let a simple model match or beat more complex ones.

## Key learning points baked into this project

- Real-world data is messy: wrong dtypes, missing values, duplicates, and
  outliers all appear here on purpose, with beginner-appropriate fixes.
- Feature engineering doesn't need to be complicated — `temp_squared` and
  `month` are simple, human-explainable features that meaningfully help.
- Scaling only really matters for Linear Regression, but we scale
  everything once so the comparison across models is on equal footing.
- We split train/test **without shuffling**, because this is a
  forecasting problem — we train on the past and test on the most recent
  period, which is closer to how the model would be used in real life.
