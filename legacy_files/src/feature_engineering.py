# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer

def add_time_based_features(df):
    """Adds basic time-based features from the DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame index is not DatetimeIndex.")
        return df
    
    print("\nAdding basic time-based features (Hour, DayOfWeek, Month, etc.)...")
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek # Monday=0, Sunday=6
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    df['WeekOfYear'] = df.index.isocalendar().week.astype(int)
    df['Year'] = df.index.year # Added Year as well
    return df

def add_interaction_features(df):
    """Adds example interaction features."""
    print("\nAdding interaction features...")
    if 'T' in df.columns and 'RH' in df.columns:
        df['T_x_RH'] = df['T'] * df['RH']
        print("Created interaction feature: T_x_RH")
    
    # Handle potential naming variations for PT08 S1(CO)
    pt08s1_col = None
    if 'PT08 S1(CO)' in df.columns:
        pt08s1_col = 'PT08 S1(CO)'
    elif 'PT08.S1(CO)' in df.columns:
        pt08s1_col = 'PT08.S1(CO)'
        
    if pt08s1_col and 'Hour' in df.columns:
        df[f'{pt08s1_col}_x_Hour'] = df[pt08s1_col] * df['Hour']
        print(f"Created interaction feature: {pt08s1_col}_x_Hour")
    elif 'Hour' not in df.columns:
         print("Warning: 'Hour' column not found, cannot create interaction with sensor.")
    else:
         print("Warning: PT08 S1(CO) sensor column not found, cannot create interaction feature.")
         
    return df

def add_fourier_features(df):
    """Adds Fourier features for seasonality."""
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame index is not DatetimeIndex.")
        return df
        
    print("\nAdding Fourier features for seasonality...")
    # Annual seasonality
    day_of_year_signal = df.index.dayofyear
    df['annual_sin'] = np.sin(2 * np.pi * day_of_year_signal / 365.25)
    df['annual_cos'] = np.cos(2 * np.pi * day_of_year_signal / 365.25)

    # Diurnal seasonality
    hour_signal = df.index.hour # Assumes 'Hour' column might not exist yet
    df['diurnal_sin'] = np.sin(2 * np.pi * hour_signal / 24)
    df['diurnal_cos'] = np.cos(2 * np.pi * hour_signal / 24)
    print("Created Fourier features for annual and diurnal seasonality.")
    return df

def add_lagged_features(df, target_column, lags=[1, 3, 6, 12, 24], other_cols_to_lag=None, other_lags=[1, 6, 12]):
    """Adds lagged features for the target and optionally other columns."""
    print(f"\nAdding lagged features for '{target_column}' and potentially others...")
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found. Skipping its lag features.")
    else:
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        print(f"Added lags {lags} for '{target_column}'.")

    if other_cols_to_lag:
        for col in other_cols_to_lag:
            if col in df.columns and col != target_column:
                for lag in other_lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                print(f"Added lags {other_lags} for '{col}'.")
            else:
                 print(f"Warning: Column '{col}' for lagging not found or is the target column. Skipping.")
    return df

def add_rolling_features(df, target_column, windows=[6, 12, 24]):
    """Adds rolling mean and std deviation features for the target column."""
    print(f"\nAdding rolling features for '{target_column}'...")
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found. Skipping rolling features.")
        return df
        
    for rw in windows:
        df[f'{target_column}_roll_mean_{rw}'] = df[target_column].rolling(window=rw, min_periods=1).mean()
        df[f'{target_column}_roll_std_{rw}'] = df[target_column].rolling(window=rw, min_periods=1).std()
    print(f"Added rolling mean/std features with windows {windows} for '{target_column}'.")
    return df

def impute_missing_advanced(df, target_column):
    """Fills missing values using IterativeImputer, excluding the target column."""
    print("\nAttempting advanced imputation using IterativeImputer...")
    
    features_for_imputation = df.select_dtypes(include=np.number).columns.tolist()
    
    # Ensure target column exists and remove it from imputation features
    target_present = target_column in features_for_imputation
    if target_present:
        features_for_imputation.remove(target_column)
        
    if not features_for_imputation:
        print("No numeric features found (excluding target) for imputation. Skipping.")
        return df

    # Separate target before imputation
    y_target_temp = None
    if target_present and target_column in df.columns:
        y_target_temp = df[target_column].copy()
        df_features_only = df[features_for_imputation].copy() # Impute only on feature columns
    else:
        # If target wasn't numeric or wasn't present, impute on all numeric columns found
        df_features_only = df[features_for_imputation].copy() 

    # Check if there are NaNs to impute in the selected feature columns
    if not df_features_only.empty and df_features_only.isnull().sum().sum() > 0:
        print(f"NaNs before IterativeImputer in feature set: {df_features_only.isnull().sum().sum()}")
        imputer = IterativeImputer(max_iter=10, random_state=42, verbose=0) 
        
        # Fit and transform
        df_features_imputed_np = imputer.fit_transform(df_features_only)
        
        # Convert back to DataFrame
        df_features_imputed = pd.DataFrame(df_features_imputed_np, columns=features_for_imputation, index=df_features_only.index)
        
        # Update the original DataFrame with imputed values
        df[features_for_imputation] = df_features_imputed[features_for_imputation]
            
        print("IterativeImputer applied.")
        print(f"NaNs after IterativeImputer in feature set: {df[features_for_imputation].isnull().sum().sum()}")
    else:
        print("No NaNs found in the feature set requiring IterativeImputer.")

    # Note: This function modifies the DataFrame 'df' in place for the imputed columns.
    # It doesn't explicitly recombine with the target here, assuming 'df' is the main object being passed around.
    # Final dropna should happen after all feature engineering.
    return df

def run_feature_engineering(df, target_column, add_interactions=True, add_fourier=True, advanced_impute=True):
    """Runs the feature engineering pipeline."""
    if df is None or df.empty:
        print("Input DataFrame for feature engineering is empty. Skipping.")
        return None
        
    df_fe = df.copy()
    
    # Basic time features
    df_fe = add_time_based_features(df_fe)
    
    # Optional advanced features
    if add_interactions:
        df_fe = add_interaction_features(df_fe)
    if add_fourier:
        df_fe = add_fourier_features(df_fe)
        
    # Lagged and Rolling features (these introduce NaNs)
    # Define which other columns to lag (example)
    pt08s1_col = 'PT08 S1(CO)' if 'PT08 S1(CO)' in df_fe.columns else 'PT08.S1(CO)' if 'PT08.S1(CO)' in df_fe.columns else None
    other_cols = [pt08s1_col] if pt08s1_col else []
    
    df_fe = add_lagged_features(df_fe, target_column, other_cols_to_lag=other_cols)
    df_fe = add_rolling_features(df_fe, target_column)
    
    # Advanced Imputation (optional, applied before final dropna)
    if advanced_impute:
        df_fe = impute_missing_advanced(df_fe, target_column)

    # Final dropna after all features created and potential imputation
    print("\nDropping rows with NaNs introduced during feature engineering...")
    rows_before = len(df_fe)
    df_fe.dropna(inplace=True)
    rows_after = len(df_fe)
    print(f"Dropped {rows_before - rows_after} rows.")
    
    print(f"\nShape after all feature engineering: {df_fe.shape}")
    if df_fe.empty:
        print("DataFrame became empty after feature engineering. Check steps.")
        return None
        
    print("\n--- Feature Engineering Complete ---")
    return df_fe

