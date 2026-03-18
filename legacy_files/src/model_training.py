# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility in this module if needed elsewhere
# np.random.seed(42)
# tf.random.set_seed(42)

def split_data_chronological(X, y, test_size=0.2):
    """Performs a chronological train-test split."""
    print(f"\nSplitting data chronologically (test_size={test_size})...")
    if len(X) < 2:
        print("Error: Not enough data to split.")
        return None, None, None, None
        
    split_index = int(len(X) * (1 - test_size))
    if split_index <= 0 or split_index >= len(X):
        print(f"Error: Invalid split index {split_index} for data length {len(X)}.")
        return None, None, None, None
        
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    if X_train.empty or X_test.empty:
        print("Error: Training or testing set is empty after split.")
        return None, None, None, None
        
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scales features using StandardScaler and MinMaxScaler."""
    print("\nScaling features...")
    scaler_standard = StandardScaler()
    X_train_scaled_std = scaler_standard.fit_transform(X_train)
    X_test_scaled_std = scaler_standard.transform(X_test)

    scaler_minmax = MinMaxScaler()
    X_train_scaled_mm = scaler_minmax.fit_transform(X_train)
    X_test_scaled_mm = scaler_minmax.transform(X_test)
    print("Features scaled with StandardScaler and MinMaxScaler.")
    
    # Return scalers as well, needed for inverse transforms later if applicable
    # and for scaling new data for prediction
    scalers = {'standard': scaler_standard, 'minmax': scaler_minmax}
    
    return X_train_scaled_std, X_test_scaled_std, X_train_scaled_mm, X_test_scaled_mm, scalers

def train_linear_regression(X_train, y_train):
    """Trains a simple Linear Regression model."""
    print("\nTraining Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("Linear Regression training complete.")
    return lr_model

def train_random_forest(X_train, y_train, perform_tuning=True, n_jobs=-1):
    """Trains a RandomForestRegressor, optionally with GridSearchCV."""
    print("\nTraining Random Forest model...")
    
    # Initial model
    rf_model_initial = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs,
                                             max_depth=15, min_samples_split=5, min_samples_leaf=3,
                                             oob_score=True)
    rf_model_initial.fit(X_train, y_train)
    print("Initial Random Forest trained.")
    if hasattr(rf_model_initial, 'oob_score_') and rf_model_initial.oob_score_ is not None:
        print(f"  Initial RF OOB Score: {rf_model_initial.oob_score_:.4f}")
        
    best_rf_model = rf_model_initial
    model_name = "Random Forest (Initial)"

    if perform_tuning:
        print("Performing Hyperparameter Tuning for Random Forest (using simplified grid)...")
        # Simplified grid for faster execution
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 15],
            'min_samples_split': [10],
            'min_samples_leaf': [5]
        }
        n_cv_splits = 2 # Reduced splits
        
        if X_train.shape[0] >= n_cv_splits + 1 and X_train.shape[0] // (n_cv_splits + 1) > 0:
            tscv = TimeSeriesSplit(n_splits=n_cv_splits)
            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42, n_jobs=n_jobs),
                                       param_grid=param_grid, cv=tscv,
                                       scoring='neg_mean_squared_error', verbose=1)
            try:
                grid_search.fit(X_train, y_train)
                print("Best hyperparameters found:", grid_search.best_params_)
                best_rf_model = grid_search.best_estimator_
                model_name = "Random Forest (Optimized)"
                print("Optimized Random Forest trained.")
            except Exception as e:
                print(f"Error during GridSearchCV: {e}. Using initial model.")
        else:
            print(f"Warning: Too few samples ({X_train.shape[0]}) for TimeSeriesSplit with {n_cv_splits} splits. Skipping tuning.")
    else:
        print("Skipping hyperparameter tuning.")
        
    return best_rf_model, model_name


def train_xgboost(X_train_scaled, y_train, n_jobs=-1):
    """Trains an XGBoost Regressor model."""
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
                                 max_depth=5, random_state=42, n_jobs=n_jobs)
    xgb_model.fit(X_train_scaled, y_train)
    print("XGBoost training complete.")
    return xgb_model

def train_lightgbm(X_train_scaled, y_train, n_jobs=-1):
    """Trains a LightGBM Regressor model."""
    print("\nTraining LightGBM model...")
    lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1,
                                   num_leaves=31, random_state=42, n_jobs=n_jobs, verbose=-1)
    lgbm_model.fit(X_train_scaled, y_train)
    print("LightGBM training complete.")
    return lgbm_model

# --- LSTM Specific Functions ---
def create_lstm_sequences(X, y, time_steps=24):
    """Reshapes data into sequences for LSTM."""
    print(f"\nCreating LSTM sequences with time_steps={time_steps}...")
    Xs, ys = [], []
    if len(X) <= time_steps:
        print(f"Error: Data length ({len(X)}) is not sufficient for time_steps ({time_steps}).")
        return None, None
        
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps]) # y value corresponds to the time step *after* the sequence X
        
    print(f"Generated {len(Xs)} sequences.")
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    """Builds a simple LSTM model."""
    print("\nBuilding LSTM model architecture...")
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1) # Output layer
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    model.summary()
    return model

def train_lstm(X_train_scaled_mm, y_train, X_test_scaled_mm, y_test, time_steps=24, epochs=20, batch_size=32):
    """Creates sequences, builds, trains, and evaluates an LSTM model."""
    
    # Create sequences for training and testing
    X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled_mm, y_train.values, time_steps)
    X_test_seq, y_test_seq = create_lstm_sequences(X_test_scaled_mm, y_test.values, time_steps)

    if X_train_seq is None or X_test_seq is None or X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
        print("Skipping LSTM training due to insufficient data for sequences.")
        return None, None # Return None for model and history

    print(f"LSTM training sequences shape: {X_train_seq.shape}")
    print(f"LSTM test sequences shape: {X_test_seq.shape}")

    # Build the model
    lstm_model = build_lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    print("\nTraining LSTM model...")
    history = lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use 10% of training sequences for validation
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("LSTM training complete.")
    # Note: Evaluation and inverse scaling of predictions happen in the evaluation phase/script
    # We return the trained model and history object
    return lstm_model, history

