"""
07_train_and_evaluate.py

Purpose:
    Train 3 regression models, evaluate each with MAE/RMSE/R^2,
    compare them, plot predictions vs actuals for the best one,
    and plot feature importance.

Run:
    python 07_train_and_evaluate.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_dir = "/home/claude/electricity_forecast/data"
models_dir = "/home/claude/electricity_forecast/outputs/models"
plots_dir = "/home/claude/electricity_forecast/outputs/plots"

import os
os.makedirs(models_dir, exist_ok=True)

# ---------------------------------------------------------
# STEP 1: Load the preprocessed train/test data saved in the last step.
# joblib.load(path) reads back a Python object we previously saved.
# ---------------------------------------------------------
X_train = joblib.load(f"{data_dir}/X_train.pkl")
X_test = joblib.load(f"{data_dir}/X_test.pkl")
y_train = joblib.load(f"{data_dir}/y_train.pkl")
y_test = joblib.load(f"{data_dir}/y_test.pkl")
feature_cols = joblib.load(f"{data_dir}/feature_cols.pkl")

# ---------------------------------------------------------
# STEP 2: Define the 3 models we want to compare.
# We store them in a dictionary: {name: model_object}, so we can loop
# over them cleanly instead of repeating code 3 times.
#
#   LinearRegression()          -> fits a straight-line (weighted sum) relationship
#   DecisionTreeRegressor()     -> learns if/else splits on feature values
#   RandomForestRegressor()     -> trains MANY decision trees on random
#                                  subsets of data/features, then averages
#                                  their predictions (usually more accurate
#                                  and less prone to overfitting than a
#                                  single tree)
#
#   random_state=42 -> makes the tree-based models' internal randomness
#                       reproducible (same result every run)
#   n_estimators=200 -> Random Forest builds 200 individual trees
#   max_depth=6      -> limits how deep the Decision Tree can grow, to
#                       avoid it memorizing the training data (overfitting)
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
}

# ---------------------------------------------------------
# STEP 3: Train each model and collect evaluation metrics.
#
#   .fit(X_train, y_train)  -> the model LEARNS patterns from training data
#   .predict(X_test)        -> the model makes predictions on UNSEEN test data
#
# Metrics explained:
#   MAE  (Mean Absolute Error)     -> average absolute difference between
#                                     predicted and actual values, in the
#                                     SAME units as consumption (kWh).
#                                     Easy to interpret: "on average, we're
#                                     off by X kWh".
#   RMSE (Root Mean Squared Error) -> similar to MAE but squares errors
#                                     first (penalizing big mistakes more
#                                     heavily), then takes the square root
#                                     to return to kWh units.
#   R^2  (R-squared)               -> fraction of variance in consumption
#                                     that the model explains, from 0 to 1
#                                     (1.0 = perfect, 0 = no better than
#                                     always predicting the average).
# ---------------------------------------------------------
results = []
trained_models = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)                # train the model
    y_pred = model.predict(X_test)              # predict on test data

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # sqrt of MSE = RMSE
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    trained_models[name] = model
    predictions[name] = y_pred

    print(f"{name}: MAE={mae:.2f} kWh | RMSE={rmse:.2f} kWh | R2={r2:.3f}")

# ---------------------------------------------------------
# STEP 4: Put results in a DataFrame for a clean comparison table,
# sorted by R^2 (higher = better) so the best model appears first.
# ---------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
print("\n=== Model Comparison (sorted by R^2) ===")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------
# STEP 5: Save the comparison table and all trained models + best model name.
# ---------------------------------------------------------
results_df.to_csv("/home/claude/electricity_forecast/outputs/model_comparison.csv", index=False)

for name, model in trained_models.items():
    # Turn "Linear Regression" -> "linear_regression.pkl" for a clean filename
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"{models_dir}/{filename}")

joblib.dump(best_model_name, f"{models_dir}/best_model_name.pkl")

# ---------------------------------------------------------
# STEP 6: Plot predicted vs actual values for the BEST model.
# A perfect model would have all points lie exactly on the diagonal line.
# ---------------------------------------------------------
best_preds = predictions[best_model_name]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds, alpha=0.5, color="steelblue")
# Draw a diagonal reference line from min to max value
min_val, max_val = y_test.min(), y_test.max()
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title(f"Predicted vs Actual - {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{plots_dir}/06_predicted_vs_actual.png")
plt.close()

# ---------------------------------------------------------
# STEP 7: Plot model comparison bar chart (R^2 for each model).
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.bar(results_df["Model"], results_df["R2"], color="mediumseagreen")
plt.ylabel("R² Score")
plt.title("Model Comparison (R² - higher is better)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{plots_dir}/07_model_comparison_r2.png")
plt.close()

# ---------------------------------------------------------
# STEP 8: Feature importance.
#
# Tree-based models (Decision Tree, Random Forest) expose
# `.feature_importances_`: how much each feature reduced prediction
# error across all the tree's splits (higher = more important).
#
# Linear Regression exposes `.coef_`: the weight/slope applied to each
# (scaled) feature. We use the ABSOLUTE value of coefficients to rank
# importance, since a large negative weight is just as influential as a
# large positive one.
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, trained_models.items()):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_)

    importance_series = pd.Series(importances, index=feature_cols).sort_values()
    importance_series.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title(f"Feature Importance - {name}")
    ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig(f"{plots_dir}/08_feature_importance.png")
plt.close()

print(f"\nAll plots saved to: {plots_dir}")
print(f"All trained models saved to: {models_dir}")
