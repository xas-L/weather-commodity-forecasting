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

# # Notebook 1: Data Loading and Preprocessing
#
# **Phase 1:** This notebook covers loading the Air Quality dataset from the UCI repository,
# cleaning it (handling placeholders, missing values), and preparing the DateTime index.
# It utilizes functions defined in `src/preprocessing.py`.

# ## 1. Imports and Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Import custom functions
# If this import fails, it means the kernel's working directory
# is not the project root where the 'src' folder resides.
try:
    from src.preprocessing import run_phase1_preprocessing
    from src.utils import setup_plot_style, display_dataframe_info
except ModuleNotFoundError as e:
    print(f"ERROR: {e}")
    print("Ensure you are running this notebook from the project's root directory (e.g., 'air-quality-time-series-prediction/'),")
    print("or that the 'src' directory is correctly added to the Python path.")
    # Example manual path addition (if needed, uncomment and adjust):
    # module_path = os.path.abspath(os.path.join('..')) # If running from 'notebooks/' dir
    # if module_path not in sys.path:
    #     sys.path.append(module_path)
    # from src.preprocessing import run_phase1_preprocessing
    # from src.utils import setup_plot_style, display_dataframe_info
    raise # Re-raise the error to stop execution if import fails


# Apply plot style
setup_plot_style()

# Define constants if needed early (e.g., target variable)
TARGET_POLLUTANT = 'CO(GT)'
print(f"Target pollutant set to: {TARGET_POLLUTANT}")

# Define output directory for saving intermediate results
# Create the directory if it doesn't exist
# Use relative path from project root
PROCESSED_DATA_DIR = 'data/processed/'
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, '01_air_quality_cleaned.pkl') # Using pickle


# ## 2. Run Preprocessing Pipeline

# The `run_phase1_preprocessing` function encapsulates all steps from Phase 1.
# It handles data loading, cleaning, datetime indexing, initial imputation, and final cleanup.
# It will print status messages during execution.
# It returns the cleaned DataFrame or None if a critical step fails.
df_clean = run_phase1_preprocessing(dataset_id=360)

# ## 3. Exploration of Cleaned Data

if df_clean is not None and not df_clean.empty:
    print("\n--- Post-Preprocessing Data Exploration ---")
    # Use the utility function for display
    display_dataframe_info(df_clean, "Cleaned DataFrame")

    # Check if target pollutant is present
    if TARGET_POLLUTANT not in df_clean.columns:
         print(f"\nWARNING: Target pollutant '{TARGET_POLLUTANT}' not found in the cleaned DataFrame!")
         print("Available columns:", df_clean.columns.tolist())
         # Depending on the project needs, you might exit or redefine the target here.
         # For now, we just print a warning.

    # Save the cleaned data for subsequent notebooks
    try:
        df_clean.to_pickle(CLEANED_DATA_PATH)
        print(f"\nCleaned data saved successfully to: {CLEANED_DATA_PATH}")
    except Exception as e:
        print(f"\nError saving cleaned data to {CLEANED_DATA_PATH}: {e}")

else:
    print("\nPreprocessing failed or resulted in an empty DataFrame. Cannot proceed.")
    # Consider raising an error or exiting if df_clean is required for subsequent cells
    # raise RuntimeError("Preprocessing failed, cannot continue notebook execution.")


# ## 4. Next Steps
#
# The cleaned DataFrame (`df_clean`) is now ready for Exploratory Data Analysis (EDA)
# in the next notebook (`02_Exploratory_Data_Analysis.ipynb`). The cleaned data has been
# saved to `data/processed/01_air_quality_cleaned.pkl` for easy loading in the next stage.

