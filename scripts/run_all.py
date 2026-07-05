"""
run_all.py

Purpose:
    Runs every step of the pipeline in the correct order, end-to-end.
    Useful the first time you set up the project, or after changing
    the dataset-generation logic.

Run:
    python run_all.py
"""

import subprocess  # lets us run other .py scripts as separate processes
import sys

# List of scripts to run IN ORDER. Order matters - each step depends on
# files saved by the previous one.
steps = [
    "01_generate_dataset.py",
    "02_load_and_inspect.py",
    "03_data_cleaning.py",
    "04_eda.py",
    "05_feature_engineering.py",
    "06_preprocessing.py",
    "07_train_and_evaluate.py",
]

for step in steps:
    print("\n" + "#" * 70)
    print(f"# RUNNING: {step}")
    print("#" * 70)
    # sys.executable = path to the current Python interpreter (so we use
    # the same environment/venv that's running this script)
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"\nStep {step} failed. Stopping pipeline.")
        break
else:
    print("\nAll steps completed successfully!")
    print("Run '08_predict.py' separately to make an interactive prediction.")
