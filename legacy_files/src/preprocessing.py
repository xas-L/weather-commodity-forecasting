# src/preprocessing.py
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

def load_data_from_uci(dataset_id=360):
    """Fetches dataset from UCI repository and concatenates features/targets."""
    print(f"\nFetching dataset ID {dataset_id} from UCI ML Repository...")
    try:
        repo_data = fetch_ucirepo(id=dataset_id)
        X_repo = repo_data.data.features
        y_repo = repo_data.data.targets
        df = pd.concat([X_repo, y_repo], axis=1)
        print("Dataset fetched and concatenated successfully.")
        return df
    except Exception as e:
        print(f"An error occurred while fetching data from ucimlrepo: {e}")
        return None

def handle_missing_placeholders(df, placeholder=-200):
    """Replaces placeholder values (like -200) with NaN."""
    print(f"\nReplacing placeholder value {placeholder} with NaN...")
    initial_nan_count = df.isnull().sum().sum()
    df.replace(to_replace=placeholder, value=np.nan, inplace=True)
    new_nan_count = df.isnull().sum().sum()
    print(f"NaN count increased by {new_nan_count - initial_nan_count} after replacement.")
    # print(f"Missing values count after replacement:\n{df.isnull().sum()}") # Optional: verbose
    return df

def drop_fully_empty_columns(df):
    """Drops columns that contain only NaN values."""
    print("\nDropping fully empty columns...")
    cols_before = set(df.columns)
    df.dropna(axis=1, how='all', inplace=True)
    cols_after = set(df.columns)
    dropped_cols = cols_before - cols_after
    if dropped_cols:
        print(f"Dropped empty columns: {list(dropped_cols)}")
    else:
        print("No fully empty columns found to drop.")
    print(f"Shape after dropping empty columns: {df.shape}")
    return df

def create_datetime_index(df, date_col='Date', time_col='Time', format_str='%m/%d/%Y %H:%M:%S'):
    """Combines Date and Time columns, sets as index, and drops original cols."""
    print(f"\nAttempting to parse DateTime using format: {format_str}...")
    if date_col not in df.columns or time_col not in df.columns:
        raise KeyError(f"'{date_col}' or '{time_col}' column missing.")

    # Ensure columns are string type for concatenation and parsing
    df[date_col] = df[date_col].astype(str)
    df[time_col] = df[time_col].astype(str)

    df['DateTime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], format=format_str, errors='coerce')

    failed_count = df['DateTime'].isnull().sum()
    if failed_count > 0:
        print(f"Warning: {failed_count} DateTime values failed to parse with format '{format_str}'.")
        # Attempt alternative format if necessary (e.g., DD/MM/YYYY)
        # For simplicity here, we'll just drop rows that failed parsing
        print(f"Dropping {failed_count} rows with unparseable DateTime.")
        df.dropna(subset=['DateTime'], inplace=True)
        if df.empty:
             print("DataFrame became empty after dropping rows with invalid DateTime.")
             return None # Indicate failure

    df.set_index('DateTime', inplace=True)
    df.drop([date_col, time_col], axis=1, inplace=True, errors='ignore')
    print("Successfully created and set DateTime index.")
    return df

def convert_columns_to_numeric(df):
    """Converts all non-datetime columns to numeric, coercing errors."""
    print("\nConverting columns to numeric types...")
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or col == df.index.name:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # print("Data types after conversion:\n", df.dtypes) # Optional: verbose
    return df

def impute_missing_simple(df):
    """Fills remaining NaNs using ffill and bfill."""
    print("\nImputing remaining missing values using ffill then bfill...")
    nan_before = df.isnull().sum().sum()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    nan_after = df.isnull().sum().sum()
    print(f"NaNs filled: {nan_before - nan_after}. Remaining NaNs: {nan_after}")
    return df

def final_cleanup(df):
    """Drops any remaining rows with NaNs and any columns that are all NaN."""
    print("\nPerforming final cleanup (dropping rows/cols with any remaining NaNs)...")
    rows_before = len(df)
    cols_before = set(df.columns)
    
    df.dropna(axis=0, how='any', inplace=True) # Drop rows with any NaNs
    df.dropna(axis=1, how='all', inplace=True) # Drop columns that are all NaN
    
    rows_after = len(df)
    cols_after = set(df.columns)
    
    print(f"Rows dropped: {rows_before - rows_after}. Columns dropped: {list(cols_before - cols_after)}")
    print(f"Final shape after cleanup: {df.shape}")
    return df

def run_phase1_preprocessing(dataset_id=360):
    """Runs the complete Phase 1 preprocessing pipeline."""
    df = load_data_from_uci(dataset_id)
    if df is None: return None
    
    df = handle_missing_placeholders(df)
    df = drop_fully_empty_columns(df)
    df = create_datetime_index(df)
    if df is None: return None # Exit if DateTime parsing failed critically
    
    df = convert_columns_to_numeric(df)
    df = impute_missing_simple(df)
    df = final_cleanup(df)
    
    if df.empty:
        print("Preprocessing resulted in an empty DataFrame.")
        return None
        
    print("\n--- Phase 1 Preprocessing Complete ---")
    print("Cleaned dataset info:")
    df.info()
    return df

