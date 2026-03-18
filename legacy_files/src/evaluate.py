# src/evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler # Needed for potential LSTM inverse transform
from .model_training import create_lstm_sequences # Import helper function if needed here
from .utils import show_plot # Import plotting utility

def calculate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, and R2 score."""
    if y_true is None or y_pred is None: return {}
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) != len(y_pred):
        print(f"Error: Length mismatch between y_true ({len(y_true)}) and y_pred ({len(y_pred)}). Cannot calculate metrics.")
        return {}
    if len(y_true) == 0:
        print("Error: y_true is empty. Cannot calculate metrics.")
        return {}
        
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}


def plot_predictions_vs_actual(y_true_series, y_pred_array, model_name, target_pollutant):
    """Plots actual vs predicted values over time."""
    if y_true_series is None or y_pred_array is None: return
    # Ensure y_pred_array is flattened
    y_pred_array = np.asarray(y_pred_array).flatten()
    if len(y_true_series) != len(y_pred_array):
         print(f"Warning: Length mismatch in plot_predictions_vs_actual ({len(y_true_series)} vs {len(y_pred_array)}). Skipping plot.")
         return
         
    plt.figure(figsize=(15, 6))
    plt.plot(y_true_series.index, y_true_series.values, label='Actual', alpha=0.8, color='blue')
    # Use the same index for predictions
    plt.plot(y_true_series.index, y_pred_array, label='Predicted', linestyle='--', alpha=0.8, color='red')
    plt.title(f'{model_name}: Actual vs. Predicted {target_pollutant}')
    plt.xlabel("Time")
    plt.ylabel(target_pollutant)
    plt.legend()
    show_plot() # Use the utility function

def plot_residuals(y_true, y_pred, model_name):
    """Plots residual distribution and residuals vs. predicted values."""
    if y_true is None or y_pred is None: return
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) != len(y_pred):
        print(f"Warning: Length mismatch in plot_residuals ({len(y_true)} vs {len(y_pred)}). Skipping plot.")
        return
    if len(y_true) == 0:
        print("Warning: Empty arrays in plot_residuals. Skipping plot.")
        return
        
    residuals = y_true - y_pred
    plt.figure(figsize=(15, 6))
    
    # Residual Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, bins=30, color='purple')
    plt.title(f'Residuals Distribution ({model_name})')
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")

    # Residuals vs. Predicted
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5, color='green', s=10) 
    # Calculate bounds safely
    pred_min = np.min(y_pred) if len(y_pred) > 0 else 0
    pred_max = np.max(y_pred) if len(y_pred) > 0 else 1
    plt.hlines(0, xmin=pred_min, xmax=pred_max, colors='black', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs. Predicted Values ({model_name})')
    
    plt.tight_layout()
    show_plot() # Use the utility function

def evaluate_model(model, model_name, X_test, y_test, target_pollutant, 
                   is_lstm=False, scaler_y=None, X_test_scaled_mm=None, time_steps=24):
    """Evaluates a trained model, handling LSTM specifics."""
    print(f"\nEvaluating {model_name}...")
    
    if model is None:
        print(f"Model '{model_name}' is None. Skipping evaluation.")
        return None
        
    y_pred = None
    y_true = None
    y_true_series_for_plot = None # For non-LSTM time series plot

    if is_lstm:
        # LSTM evaluation requires sequences and inverse scaling
        if scaler_y is None or X_test_scaled_mm is None or y_test is None:
            print(f"Error: Missing scaler_y, X_test_scaled_mm, or y_test for LSTM '{model_name}'. Skipping.")
            return None
            
        # Create test sequences
        X_test_seq, y_test_seq = create_lstm_sequences(X_test_scaled_mm, y_test.values, time_steps)
        
        if X_test_seq is None or y_test_seq is None or X_test_seq.shape[0] == 0:
            print(f"Could not create test sequences for LSTM '{model_name}'. Skipping evaluation.")
            return None
            
        try:
            print("Predicting with LSTM model...")
            y_pred_scaled = model.predict(X_test_seq)
            print("Inverse transforming LSTM predictions...")
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            y_true = y_test_seq.flatten() # Use the unscaled y sequence, flattened
        except Exception as e:
            print(f"Error during LSTM prediction or inverse transform for {model_name}: {e}")
            return None
    else:
        # For non-LSTM models (X_test should be appropriately scaled *before* calling this function)
        if X_test is None or y_test is None:
             print(f"Error: Missing X_test or y_test for {model_name}. Skipping.")
             return None
        try:
            print(f"Predicting with {model_name} model...")
            y_pred = model.predict(X_test)
            y_true = y_test.values # Use the original y_test series values
            y_true_series_for_plot = y_test # Keep series for plotting index
        except Exception as e:
             print(f"Error during prediction for {model_name}: {e}")
             return None

    # Calculate Metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    if not metrics: # If metrics calculation failed
        print(f"Failed to calculate metrics for {model_name}.")
        return None
        
    print(f"  MAE: {metrics['MAE']:.4f}, MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")

    # Plotting
    try:
        if not is_lstm and y_true_series_for_plot is not None:
            print("Plotting predictions vs actual...")
            plot_predictions_vs_actual(y_true_series_for_plot, y_pred, model_name, target_pollutant)
            print("Plotting residuals...")
            plot_residuals(y_true, y_pred, model_name)
        elif is_lstm:
            print("Plotting residuals for LSTM...")
            plot_residuals(y_true, y_pred, model_name)
            print("(Time series plot skipped for LSTM due to sequence format)")
        else:
             print("Skipping plots due to missing data.")
             
    except Exception as e:
        print(f"Error during plotting for {model_name}: {e}")

    return metrics

def plot_lstm_history(history, model_name="LSTM"):
    """Plots training and validation loss from LSTM history."""
    if history and history.history: # Check if history object and its history attribute exist
        plt.figure(figsize=(10, 6))
        if 'loss' in history.history:
            plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        if 'loss' in history.history or 'val_loss' in history.history: 
            plt.legend()
        show_plot() # Use utility function
    else:
        print(f"{model_name} training history not available for plotting loss.")

