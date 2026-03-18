# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5 # Adjust if needed
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Notebook 3: Feature Engineering & Basic Models
#
# **Phases 3, 4, 5:** This notebook covers:
# 1. Initial Feature Engineering (time-based, lags, rolling stats).
# 2. Training baseline models (Linear Regression, Random Forest).
# 3. Evaluating these baseline models.
# It loads the cleaned data from Notebook 1 and uses functions from `src/`.

# ## 1. Imports and Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import joblib # For saving models/scalers if needed

# Add src directory to path
# Assuming notebook is run from project root
try:
    from src.feature_engineering import (add_time_based_features, add_lagged_features,
                                         add_rolling_features)
    from src.model_training import (split_data_chronological, train_linear_regression,
                                    train_random_forest)
    from src.evaluate import evaluate_model
    from src.utils import setup_plot_style, display_dataframe_info, save_object, load_object
except ModuleNotFoundError as e:
    print(f"ERROR: {e}")
    print("Ensure you are running this notebook from the project's root directory.")
    raise

# Apply plot style
setup_plot_style()

# Define constants
TARGET_POLLUTANT = 'CO(GT)'
FIGURES_DIR = 'reports/figures/'
MODELS_DIR = 'models/' # Directory to save models
REPORTS_DIR = 'reports/' # Directory for saving results csv
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# Path to cleaned data
CLEANED_DATA_PATH = 'data/processed/01_air_quality_cleaned.pkl'

# Control flags
PERFORM_RF_TUNING = True # Set to False to speed up by skipping RF GridSearchCV

# ## 2. Load Cleaned Data

if os.path.exists(CLEANED_DATA_PATH):
    print(f"Loading cleaned data from: {CLEANED_DATA_PATH}")
    try:
        df_clean = pd.read_pickle(CLEANED_DATA_PATH)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data from {CLEANED_DATA_PATH}: {e}")
        df_clean = None
else:
    print(f"Error: Cleaned data file not found at {CLEANED_DATA_PATH}")
    print("Please run Notebook 01 first.")
    df_clean = None

if df_clean is None or df_clean.empty:
     raise SystemExit("Stopping notebook execution: Cleaned data not available.")
if TARGET_POLLUTANT not in df_clean.columns:
     raise SystemExit(f"Stopping notebook execution: Target '{TARGET_POLLUTANT}' not found.")

# ## 3. Initial Feature Engineering (Phase 3)

print("\n--- Phase 3: Initial Feature Engineering ---")
df_fe_basic = df_clean.copy()

# Add basic time features
df_fe_basic = add_time_based_features(df_fe_basic)

# Add lagged features (Target + Example Sensor)
# Determine the correct sensor column name ('PT08 S1(CO)' or 'PT08.S1(CO)')
pt08s1_col = None
if 'PT08 S1(CO)' in df_fe_basic.columns: pt08s1_col = 'PT08 S1(CO)'
elif 'PT08.S1(CO)' in df_fe_basic.columns: pt08s1_col = 'PT08.S1(CO)'
other_lag_cols = [pt08s1_col] if pt08s1_col else []

df_fe_basic = add_lagged_features(df_fe_basic, TARGET_POLLUTANT, other_cols_to_lag=other_lag_cols)

# Add rolling features (Target)
df_fe_basic = add_rolling_features(df_fe_basic, TARGET_POLLUTANT, windows=[6, 12, 24])

# Drop NaNs introduced by lags/rolling features
print("\nDropping rows with NaNs introduced by feature engineering...")
rows_before = len(df_fe_basic)
df_fe_basic.dropna(inplace=True)
rows_after = len(df_fe_basic)
print(f"Dropped {rows_before - rows_after} rows.")
print(f"Shape after basic feature engineering: {df_fe_basic.shape}")

if df_fe_basic.empty:
    raise SystemExit("DataFrame became empty after basic feature engineering. Check steps.")

# Display info about the feature-engineered DataFrame
display_dataframe_info(df_fe_basic, "DataFrame with Basic Features")

# Optional: Save this intermediate DataFrame
# BASIC_FE_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, '03_air_quality_basic_features.pkl')
# df_fe_basic.to_pickle(BASIC_FE_DATA_PATH)
# print(f"DataFrame with basic features saved to {BASIC_FE_DATA_PATH}")

# ## 4. Data Splitting (Phase 4 Prep)

print("\n--- Preparing Data for Basic Models ---")

# Separate features (X) and target (y)
if TARGET_POLLUTANT not in df_fe_basic.columns:
     raise SystemExit(f"Stopping notebook execution: Target '{TARGET_POLLUTANT}' not found in df_fe_basic.")

X_basic = df_fe_basic.drop(columns=[TARGET_POLLUTANT], errors='ignore')
y_basic = df_fe_basic[TARGET_POLLUTANT]

# Select only numeric features for modeling
X_basic = X_basic.select_dtypes(include=np.number)
print(f"Shape of X_basic (features): {X_basic.shape}")
print(f"Shape of y_basic (target): {y_basic.shape}")

if X_basic.empty or y_basic.empty or X_basic.shape[0] != y_basic.shape[0]:
    raise SystemExit("X_basic or y_basic is empty or shapes mismatch. Stopping.")

# Split data chronologically
X_train_b, X_test_b, y_train_b, y_test_b = split_data_chronological(X_basic, y_basic, test_size=0.2)

if X_train_b is None: # Check if split failed
     raise SystemExit("Stopping notebook execution: Data splitting failed.")

# Note: Scaling is typically applied *after* splitting. For these basic models (LR, RF),
# scaling isn't strictly necessary, especially for RF. We'll apply scaling in Notebook 04
# before training models like XGBoost, LightGBM, and LSTM.

# ## 5. Model Training (Phase 4 - Basic Models)

print("\n--- Phase 4: Training Basic Models ---")
basic_models = {} # Dictionary to store trained models

# 5.1 Linear Regression
lr_model = train_linear_regression(X_train_b, y_train_b)
if lr_model:
    basic_models['Linear Regression'] = lr_model

# 5.2 Random Forest
# train_random_forest returns (model, model_name)
rf_model, rf_model_name = train_random_forest(X_train_b, y_train_b, perform_tuning=PERFORM_RF_TUNING)
if rf_model:
    basic_models[rf_model_name] = rf_model # Use the dynamic name

# Optional: Save the trained basic models
for name, model in basic_models.items():
    if model:
        filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.joblib'
        save_object(model, os.path.join(MODELS_DIR, filename))


# ## 6. Model Evaluation (Phase 5 - Basic Models)

print("\n--- Phase 5: Evaluating Basic Models ---")
basic_eval_results = {}

# Evaluate models stored in the dictionary
for model_name, model_instance in basic_models.items():
    # Note: Basic models are evaluated on unscaled data here
    metrics = evaluate_model(model_instance, model_name, X_test_b, y_test_b, TARGET_POLLUTANT)
    if metrics: basic_eval_results[model_name] = metrics

# Display summary of results
print("\n--- Summary of Basic Model Performance ---")
if basic_eval_results:
    results_df_basic = pd.DataFrame(basic_eval_results).T
    print(results_df_basic)
    # Save results
    basic_results_path = os.path.join(REPORTS_DIR, '03_basic_model_results.csv')
    results_df_basic.to_csv(basic_results_path)
    print(f"Basic model results saved to {basic_results_path}")
else:
    print("No basic model evaluation results to display.")


# ## 7. Next Steps
#
# Baseline models have been trained and evaluated.
# The next notebook (`04_Advanced_Features_Models.ipynb`) will:
# 1. Implement more sophisticated feature engineering (interactions, Fourier).
# 2. Apply advanced imputation (`IterativeImputer`).
# 3. Train and evaluate advanced models (XGBoost, LightGBM, LSTM).
# 4. Compare performance against these baseline models.

