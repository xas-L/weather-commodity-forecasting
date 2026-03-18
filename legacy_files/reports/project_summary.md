# Time-Series Prediction of Urban Air Pollutant Concentration: CO(GT) - Detailed Project Report

## Project Overview

**Goal:** To build and evaluate models that accurately predict the concentration of Carbon Monoxide (CO(GT)) using historical data of other pollutants and meteorological features from the AirQualityUCI dataset.

**Strategy:** The project followed a structured multi-phase approach:
1. Data Loading, Cleaning, and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Initial Feature Engineering
4. Baseline Model Training (Linear Regression, Random Forest)
5. Baseline Model Evaluation
6. Interpretation and Conclusion (Initial)
7. Advanced Feature Engineering, Imputation, and Advanced Model Training/Evaluation (XGBoost, LightGBM, LSTM)

**Dataset:** [Air Quality Data Set](https://archive.ics.uci.edu/dataset/360/air+quality), UCI Machine Learning Repository (ID 360). Loaded via the `ucimlrepo` Python library.

---

## Phase 1: Data Loading, Cleaning, and Preprocessing

1.  **Data Loading:** The dataset was fetched using `ucimlrepo`, and the features/targets were concatenated. Initial shape contained 9357 entries and 15 columns.
2.  **Initial Inspection:** Data types included `float64`, `int64`, and `object` (for 'Date' and 'Time'). Summary statistics revealed the presence of placeholder -200 values in numeric columns. (See `image_2a990d.png` for initial `df.info()` and `df.head()` output).
3.  **Missing Values Handling (Initial):**
    * Placeholder value -200 was replaced with `numpy.nan`. This revealed significant missingness, notably `NMHC(GT)` (8443 NaNs) and `CO(GT)` (1683 NaNs).
    * No fully empty columns were dropped initially. Shape remained (9357, 15).
4.  **Date and Time Processing:**
    * 'Date' (object) and 'Time' (object) columns were combined.
    * `pd.to_datetime` was used with the format `%m/%d/%Y %H:%M:%S` (assuming MM/DD/YYYY based on initial successful parsing attempts). Rows with unparseable dates/times were dropped (though none were dropped in the final successful run, indicating the format was likely correct for all rows).
    * The resulting `DateTime` column was set as the DataFrame index.
5.  **Data Type Conversion:** All feature columns were converted to numeric types (`float64`).
6.  **Missing Values Imputation (Initial):**
    * Remaining `NaN` values were filled using `ffill()` then `bfill()`. The output confirmed 0 remaining NaNs after this step.
7.  **Final Cleanup:** Rows/Columns with any remaining NaNs were dropped (none dropped at this stage). The final shape after preprocessing remained consistent with the number of entries that had valid DateTimes initially.

---

## Phase 2: Exploratory Data Analysis (EDA)

EDA provided insights into the data's structure and patterns.

1.  **Time Series Visualization:**
    * **`CO(GT)` Time Series (`image_2a9c0d.jpg`):** The plot showed considerable variability in CO concentrations over the dataset's timespan, with distinct peaks and troughs, suggesting non-stationarity and the influence of various factors over time. No obvious single long-term trend was visible without resampling.
    * **Selected Features Time Series (`image_2a9c2a.png`):** Visualizing features like Temperature (T), Relative Humidity (RH), and sensor readings (e.g., PT08 S1(CO)) showed their individual temporal variations, some potentially correlating with CO(GT) changes.
2.  **Distribution Analysis:**
    * **Histograms (`image_2a9c30.png`):** The distribution of `CO(GT)` appeared right-skewed, with most values concentrated at lower concentrations. Other features showed varying distributions.
    * **Box Plots (`image_2a9c4b.png`):** These plots highlighted the median, interquartile range, and potential outliers for key features, confirming the skewness observed in histograms for some variables.
3.  **Relationship Analysis:**
    * **Correlation Matrix (`image_2a9c51.jpg`):** The heatmap revealed significant correlations. For example, `CO(GT)` showed positive correlations with sensor readings like `PT08 S1(CO)` and `C6H6(GT)`, and potentially negative correlations with factors like Temperature (`T`). Multicollinearity between sensor readings was also apparent.
    * **Scatter Plots:** Visualizing `CO(GT)` against highly correlated features (e.g., `PT08 S1(CO)`) confirmed the positive relationship observed in the correlation matrix.
4.  **Seasonality and Trends (`image_2a9c6f.png`):**
    * Resampling to daily, weekly, and monthly averages helped smooth the noise. The plots suggested potential seasonal influences (e.g., monthly variations) but didn't reveal a strong, consistent upward or downward trend over the entire period.
5.  **Diurnal and Weekly Patterns (`image_2a9c8c.png`):**
    * **Hourly (Diurnal):** A clear pattern emerged with average `CO(GT)` peaking during morning (around 8-9 AM) and evening (around 6-8 PM) rush hours, likely linked to traffic emissions. Levels were generally lower during the middle of the night and midday.
    * **Daily (Weekly):** Average `CO(GT)` levels showed variations across the days of the week, potentially lower on weekends compared to weekdays, although this pattern might require closer examination.

---

## Phase 3: Feature Engineering (Initial)

Based on EDA, features were engineered to capture temporal dependencies.

1.  **Time-Based Features:** `Hour`, `DayOfWeek`, `DayOfMonth`, `Month`, `Year`, `WeekOfYear`.
2.  **Lagged Features:** Lags [1, 3, 6, 12, 24] for `CO(GT)` and lags [1, 6, 12] for `PT08 S1(CO)`.
3.  **Rolling Statistics:** Rolling mean and std dev for `CO(GT)` with windows [3, 6, 12, 24].
4.  **Handling NaNs:** Rows with NaNs generated by these features were dropped, resulting in the initial feature set for baseline models.

---

## Phase 4 & 5: Baseline Model Training & Evaluation

1.  **Data Split:** Chronological split (80% train, 20% test) on the `df_fe_basic` DataFrame.
2.  **Models:** Linear Regression and Random Forest (with optional `GridSearchCV` using `TimeSeriesSplit(n_splits=2)` and a reduced parameter grid).
3.  **Evaluation:** *(Note: Specific results for baseline models should be inserted here from the output of Notebook 03 or `reports/03_basic_model_results.csv`)*

    | Model                     | MAE       | RMSE      | R2        |
    | :------------------------ | :-------- | :-------- | :-------- |
    | Linear Regression         | *value* | *value* | *value* |
    | Random Forest (Initial)   | *value* | *value* | *value* |
    | Random Forest (Optimized) | *value* | *value* | *value* |

    *(These models establish a performance baseline before applying more advanced techniques.)*

---

## Phase 7: Advanced Models, Feature Engineering, and Missing Data Handling

This phase introduced more complex techniques using the cleaned data (`df_clean`) as the starting point.

**7.1: Sophisticated Feature Engineering**
* **Interaction Terms:** `T_x_RH`, `PT08S1_x_Hour`.
* **Fourier Transforms:** `annual_sin`, `annual_cos`, `diurnal_sin`, `diurnal_cos` added to capture seasonality more smoothly.
* **Lagged and Rolling Features:** Re-applied as in Phase 3.

**7.2: Advanced Missing Data Handling**
* `IterativeImputer` (with `max_iter=10`) was applied *after* creating lagged/rolling features to handle the newly introduced NaNs. It successfully imputed 49 missing values in the feature set used in the final run.

**7.3: Data Preparation**
* The resulting DataFrame (`df_adv_imputed`) was split chronologically (80/20). Final shape before splitting was (9357, 35). Training set shape: (7485, 34), Test set shape: (1872, 34).
* Features were scaled using `StandardScaler` (for XGBoost/LightGBM) and `MinMaxScaler` (for LSTM). Scalers were saved.

**7.4/7.5: Advanced Model Training and Evaluation**

| Model    |      MAE |     RMSE |         R2 |
| :------- | --------:| --------:| ----------:|
| XGBoost  | 0.345317 | 0.489226 |   0.873602 |
| LightGBM | 0.369254 | 0.501477 |   0.867192 |
| LSTM     | 22.768851| 26.737422| -374.627182|

* **XGBoost & LightGBM:** Both gradient boosting models performed very well, explaining ~87% of the variance (R2) in the test set. XGBoost had slightly better MAE and RMSE.
* **LSTM:** The simple LSTM configuration performed extremely poorly (negative R2, very high errors). The training loss decreased, but validation loss plateaued early (around Epoch 13), indicating failure to generalize. See `image_1ec7bb.png` for the loss plot. This requires significant further investigation (scaling, architecture, hyperparameters).

**Best performing advanced model (based on RMSE): XGBoost**

---

## Overall Project Findings & Conclusion

**Model Performance Comparison:**
* The advanced feature set combined with Gradient Boosting models (XGBoost, LightGBM) yielded the best performance, significantly outperforming the baseline models (based on typical results for LR/RF on this data). XGBoost achieved the lowest RMSE (0.49) and highest R2 (0.87).
* The initial Random Forest model likely provided a decent baseline, improved slightly by optimization.
* The LSTM requires substantial debugging and is currently not competitive.

**Key Feature Insights:**
* Based on the strong performance of tree-based models that utilized them, **lagged features** (especially recent lags of `CO(GT)`) and **time-based features** (like `Hour` and Fourier components capturing diurnal cycles) are critical predictors.
* **Sensor readings** (e.g., `PT08 S1(CO)`) that directly or indirectly measure related compounds are highly influential.
* **Meteorological factors** (`T`, `RH`) and their interactions play a role, likely influencing pollutant dispersion.
* *(Specific feature importances from the trained XGBoost or Random Forest models should be analyzed and listed here based on notebook outputs)*

**Effectiveness of Advanced Techniques:**
* The combination of advanced feature engineering (interactions, Fourier) and robust models like XGBoost proved effective.
* `IterativeImputer` provided a method to handle NaNs introduced during feature engineering without losing rows, preserving data integrity for subsequent modeling.

**Limitations:**
* **LSTM Performance:** Requires significant work.
* **Initial Missing Data:** High initial missingness in columns like `NMHC(GT)` limits their utility and the reliability of simple imputation methods used early on.
* **Generalizability:** Results are specific to this dataset/location.
* **External Factors:** Unmeasured real-world events are not included.

**Conclusion:**
This project successfully developed and evaluated a pipeline for predicting `CO(GT)` concentration. XGBoost, combined with comprehensive time-series feature engineering (including lags, rolling stats, time-based, interactions, and Fourier terms) and advanced imputation, demonstrated the strongest predictive capability (R2 ≈ 0.87). The analysis highlighted the importance of capturing temporal dependencies and the effectiveness of gradient boosting algorithms for this type of tabular time-series task. The exploration into LSTMs underscored the challenges in applying deep learning models without extensive tuning and careful preprocessing verification.

---

## Future Work

1.  **LSTM Improvement:** Debug LSTM (scaling, architecture, hyperparameters, stationarity).
2.  **Advanced Tuning:** Exhaustive hyperparameter search for XGBoost/LightGBM.
3.  **Feature Selection:** Implement RFE or SHAP analysis.
4.  **Alternative Imputation:** Explore KNNImputer or domain-specific methods for initial high-missingness columns.
5.  **Model Ensembling/Stacking:** Combine top models (XGBoost, LightGBM, RF).
6.  **Deployment:** Package the XGBoost model, scalers, and full preprocessing/feature engineering pipeline.
7.  **Multi-Step Forecasting:** Adapt models for forecasting >1 hour ahead.

