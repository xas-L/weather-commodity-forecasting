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

# # Notebook 2: Exploratory Data Analysis (EDA)
#
# **Phase 2:** This notebook focuses on exploring the cleaned Air Quality dataset.
# We will visualize time series, distributions, correlations, and temporal patterns.
# It loads the cleaned data saved from Notebook 1.

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
    from src.utils import setup_plot_style, show_plot, display_dataframe_info
except ModuleNotFoundError as e:
    print(f"ERROR: {e}")
    print("Ensure you are running this notebook from the project's root directory (e.g., 'air-quality-time-series-prediction/'),")
    print("or that the 'src' directory is correctly added to the Python path.")
    raise # Re-raise the error to stop execution if import fails


# Apply plot style
setup_plot_style()

# Define constants
TARGET_POLLUTANT = 'CO(GT)'
FIGURES_DIR = 'reports/figures/' # Relative path from project root
os.makedirs(FIGURES_DIR, exist_ok=True) # Ensure figure directory exists

# Define path to cleaned data from Notebook 1
CLEANED_DATA_PATH = 'data/processed/01_air_quality_cleaned.pkl' # Relative path from project root

# ## 2. Load Cleaned Data

if os.path.exists(CLEANED_DATA_PATH):
    print(f"Loading cleaned data from: {CLEANED_DATA_PATH}")
    try:
        df_clean = pd.read_pickle(CLEANED_DATA_PATH)
        print("Data loaded successfully.")
        # display_dataframe_info(df_clean, "Loaded Cleaned Data") # Optional: Display full info
        print("\nLoaded DataFrame head:")
        print(df_clean.head())
    except Exception as e:
        print(f"Error loading data from {CLEANED_DATA_PATH}: {e}")
        df_clean = None
else:
    print(f"Error: Cleaned data file not found at {CLEANED_DATA_PATH}")
    print("Please run Notebook 01 first to generate the cleaned data.")
    df_clean = None # Set to None if loading failed

# Exit if data loading failed
if df_clean is None or df_clean.empty:
     raise SystemExit("Stopping notebook execution: Cleaned data not available.")

# Check if target is present
if TARGET_POLLUTANT not in df_clean.columns:
     raise SystemExit(f"Stopping notebook execution: Target '{TARGET_POLLUTANT}' not found in loaded data.")

# Define features list (excluding target)
features = df_clean.columns.drop(TARGET_POLLUTANT, errors='ignore').tolist()
print(f"\nTarget: {TARGET_POLLUTANT}")
print(f"Features ({len(features)}): {features[:5]}...") # Print first few features

# ## 3. Time Series Visualization

print("\n--- 3. Time Series Visualization ---")

# Plot Target Variable
try:
    plt.figure(figsize=(15, 6))
    df_clean[TARGET_POLLUTANT].plot(alpha=0.9, title=f'Time Series of {TARGET_POLLUTANT} Concentration')
    show_plot(save_path=os.path.join(FIGURES_DIR, '02_target_timeseries.png'))
except Exception as e:
    print(f"Error plotting target time series: {e}")

# Plot Selected Features
if features:
    try:
        plt.figure(figsize=(15, 8))
        num_features_to_plot = min(len(features), 4)
        # Select features known to be relevant or diverse
        features_to_plot_ts = [f for f in ['PT08 S1(CO)', 'PT08.S1(CO)', 'T', 'RH', 'C6H6(GT)'] if f in features][:num_features_to_plot]

        print(f"Plotting time series for features: {features_to_plot_ts}")
        plotted_count = 0
        if features_to_plot_ts: # Check if list is not empty
            for i, feature_name in enumerate(features_to_plot_ts):
                if feature_name in df_clean.columns:
                    plt.subplot(len(features_to_plot_ts), 1, plotted_count + 1)
                    df_clean[feature_name].plot(label=feature_name, alpha=0.8)
                    plt.legend(loc='upper right')
                    plt.title(f'Time Series of {feature_name}', fontsize=10) # Add subplot titles
                    plotted_count += 1

            if plotted_count > 0:
                plt.suptitle('Time Series of Selected Features', y=1.02, fontsize=14) # Add overall title
                plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
                show_plot(save_path=os.path.join(FIGURES_DIR, '02_features_timeseries.png'))
            else:
                print("No selected features found to plot time series.")
        else:
            print("Could not find any of the pre-selected features to plot.")
    except Exception as e:
        print(f"Error plotting feature time series: {e}")
else:
    print("No features available for time series plots.")


# ## 4. Distribution Analysis

print("\n--- 4. Distribution Analysis ---")

plot_cols_dist = [TARGET_POLLUTANT] + [f for f in features[:min(len(features), 5)] if f in df_clean.columns]
plot_cols_dist = [col for col in plot_cols_dist if col in df_clean.columns] # Final check

if plot_cols_dist:
    print(f"Analyzing distributions for: {plot_cols_dist}")
    num_plot_cols_dist = 2
    subplot_rows_dist = (len(plot_cols_dist) + num_plot_cols_dist - 1) // num_plot_cols_dist

    # Histograms
    try:
        plt.figure(figsize=(14, 4 * subplot_rows_dist))
        for i, col in enumerate(plot_cols_dist):
            plt.subplot(subplot_rows_dist, num_plot_cols_dist, i + 1)
            sns.histplot(df_clean[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
        plt.suptitle('Feature Distributions (Histograms)', y=1.02, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        show_plot(save_path=os.path.join(FIGURES_DIR, '02_distributions_hist.png'))
    except Exception as e:
        print(f"Error plotting histograms: {e}")

    # Box Plots
    try:
        plt.figure(figsize=(14, 4 * subplot_rows_dist))
        for i, col in enumerate(plot_cols_dist):
            plt.subplot(subplot_rows_dist, num_plot_cols_dist, i + 1)
            sns.boxplot(y=df_clean[col])
            plt.title(f'Box Plot of {col}')
        plt.suptitle('Feature Distributions (Box Plots)', y=1.02, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        show_plot(save_path=os.path.join(FIGURES_DIR, '02_distributions_box.png'))
    except Exception as e:
        print(f"Error plotting box plots: {e}")
else:
    print("No columns selected/available for distribution plots.")

# ## 5. Relationship Analysis

print("\n--- 5. Relationship Analysis ---")

# Correlation Matrix
numeric_df = df_clean.select_dtypes(include=np.number)
if not numeric_df.empty:
    print("Calculating correlation matrix...")
    try:
        correlation_matrix = numeric_df.corr()
        plt.figure(figsize=(14, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5, annot_kws={"size": 7})
        show_plot('Correlation Matrix of Numerical Features', save_path=os.path.join(FIGURES_DIR, '02_correlation_matrix.png'))

        # Scatter Plots vs Target
        if TARGET_POLLUTANT in numeric_df.columns and not correlation_matrix.empty:
            target_correlations = correlation_matrix[TARGET_POLLUTANT].abs().sort_values(ascending=False)
            # Select top 3-4 correlated features (excluding self)
            top_correlated_features = target_correlations.drop(TARGET_POLLUTANT, errors='ignore').head(4).index.tolist()
            top_correlated_features = [f for f in top_correlated_features if f in df_clean.columns] # Ensure they exist

            if top_correlated_features:
                print(f"\nScatter plots for {TARGET_POLLUTANT} vs. {top_correlated_features}...")
                num_scatter_plots = len(top_correlated_features)
                plt.figure(figsize=(14 , 6 * ((num_scatter_plots + 1)//2) )) # Adjust layout based on rows
                for i, feature_scat in enumerate(top_correlated_features):
                    plt.subplot( (num_scatter_plots + 1)//2, 2, i + 1) # Arrange in 2 columns
                    sns.scatterplot(x=df_clean[feature_scat], y=df_clean[TARGET_POLLUTANT], alpha=0.5, s=10)
                    plt.title(f'{TARGET_POLLUTANT} vs {feature_scat}')
                plt.suptitle('Scatter Plots: Target vs. Top Correlated Features', y=1.02, fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                show_plot(save_path=os.path.join(FIGURES_DIR, '02_scatter_plots.png'))
            else:
                print(f"Not enough other features correlated with {TARGET_POLLUTANT} for scatter plots.")
        else:
             print(f"Target pollutant {TARGET_POLLUTANT} not numeric or correlation matrix empty for scatter plots.")
    except Exception as e:
        print(f"Error during correlation/scatter plot analysis: {e}")
else:
    print("No numeric columns found for correlation matrix.")

# ## 6. Seasonality and Temporal Patterns

print("\n--- 6. Seasonality and Temporal Patterns ---")

if isinstance(df_clean.index, pd.DatetimeIndex) and TARGET_POLLUTANT in df_clean.columns and not df_clean.empty:
    print("Analyzing long-term trends (Daily, Weekly, Monthly averages)...")
    try:
        # Resampling
        daily_mean = df_clean[TARGET_POLLUTANT].resample('D').mean()
        weekly_mean = df_clean[TARGET_POLLUTANT].resample('W').mean()
        monthly_mean = df_clean[TARGET_POLLUTANT].resample('ME').mean() # Use MonthEnd

        # Plotting Resampled Data
        plt.figure(figsize=(15, 10))
        plt.subplot(3,1,1); daily_mean.plot(alpha=0.8); plt.title(f'Daily Average {TARGET_POLLUTANT}')
        plt.subplot(3,1,2); weekly_mean.plot(alpha=0.8); plt.title(f'Weekly Average {TARGET_POLLUTANT}')
        plt.subplot(3,1,3); monthly_mean.plot(alpha=0.8); plt.title(f'Monthly Average {TARGET_POLLUTANT}')
        plt.suptitle("Seasonality and Trends", y=1.0, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        show_plot(save_path=os.path.join(FIGURES_DIR, '02_seasonality_trends.png'))

        # Diurnal and Weekly Patterns
        print("Analyzing diurnal (hourly) and weekly patterns...")
        df_eda = df_clean.copy()
        if isinstance(df_eda.index, pd.DatetimeIndex):
            df_eda['Hour'] = df_eda.index.hour
            df_eda['DayOfWeek_Name'] = df_eda.index.day_name()

            plt.figure(figsize=(16, 6)) # Wider figure
            # Diurnal Pattern
            plt.subplot(1, 2, 1)
            hourly_avg = df_eda.groupby('Hour')[TARGET_POLLUTANT].mean()
            hourly_avg.plot(kind='bar', color=sns.color_palette("viridis", 24)) # Use palette
            plt.title(f'Average {TARGET_POLLUTANT} by Hour of Day')
            plt.ylabel(f'Average {TARGET_POLLUTANT}')
            plt.xlabel('Hour of Day')

            # Weekly Pattern
            plt.subplot(1, 2, 2)
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if 'DayOfWeek_Name' in df_eda.columns:
                 # Ensure DayOfWeek_Name is categorical with the specified order
                 df_eda['DayOfWeek_Name'] = pd.Categorical(df_eda['DayOfWeek_Name'], categories=days_order, ordered=True)
                 # Use observed=False if using pandas >= 1.1.0 to include all categories even if some have no data
                 weekly_avg = df_eda.groupby('DayOfWeek_Name', observed=False)[TARGET_POLLUTANT].mean()
                 weekly_avg.plot(kind='bar', color=sns.color_palette("viridis", 7))
                 plt.title(f'Average {TARGET_POLLUTANT} by Day of Week')
                 plt.ylabel(f'Average {TARGET_POLLUTANT}')
                 plt.xlabel('Day of Week')
                 plt.xticks(rotation=45)
            else:
                 print("Warning: 'DayOfWeek_Name' column not found for weekly pattern plot.")

            plt.suptitle("Diurnal and Weekly Patterns", y=1.02, fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            show_plot(save_path=os.path.join(FIGURES_DIR, '02_diurnal_weekly_patterns.png'))
    except Exception as e:
        print(f"Error during seasonality analysis: {e}")
else:
    print("Skipping seasonality analysis: Index is not DatetimeIndex, target missing, or DataFrame empty.")


# ## 7. Next Steps
#
# The EDA has provided valuable insights into the data's characteristics and patterns.
# The next notebook (`03_Feature_Eng_Basic_Models.ipynb`) will focus on:
# 1. Implementing initial feature engineering based on these insights (lags, rolling stats, basic time features).
# 2. Training and evaluating baseline models (Linear Regression, Random Forest).

