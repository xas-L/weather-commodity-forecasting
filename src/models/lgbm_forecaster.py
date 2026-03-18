"""
Implementation for the short-range temperature and HDD forecasting model using
LightGBM gradient boosting, with walk-forward expanding-window cross-validation,
Optuna hyperparameter optimisation, and SHAP-based feature importance.

LightGBM is chosen over XGBoost here for three reasons. First, it trains faster
on the tabular weather feature matrix, which matters when running walk-forward
CV with 8-10 refits. Second, its histogram-based splitting handles the dense
continuous features from ERA5 efficiently. Third, the native early stopping API
is cleaner to use with a time-series validation set. XGBoost is retained in
regime_classifier.py because it tends to produce slightly more stable probability
calibration for the ternary classification task.

Walk-forward cross-validation is non-negotiable for this project. A single
train/test split is insufficient because:
    - It gives a single point estimate of performance rather than a distribution.
    - It does not reflect how the model would behave in live deployment, where
      it is retrained periodically on an expanding history.
    - It can be misleadingly optimistic or pessimistic depending on where the
      split falls relative to unusual weather regimes.

SHAP values are used to explain model predictions to non-meteorologist PMs.
The key output is a plain-language attribution: "the model is forecasting a
cold anomaly because the NAO has been persistently negative for 10 days and
HDD accumulations are 8 degree-days above seasonal norms."

Setup required before use:
    pip install lightgbm shap optuna pandas numpy scikit-learn
"""

import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.models.baseline import compute_skill_score, ClimatologyBaseline

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Default hyperparameter ranges for Optuna search. These bounds are informed
# by experience with tabular weather data and are intentionally conservative
# to reduce overfitting risk on a dataset of roughly 3,000 daily observations.
OPTUNA_PARAM_SPACE: dict = {
    "n_estimators":      (300, 1500),
    "learning_rate":     (0.005, 0.05),
    "num_leaves":        (20, 100),
    "max_depth":         (3, 8),
    "min_child_samples": (10, 60),
    "subsample":         (0.6, 1.0),
    "colsample_bytree":  (0.6, 1.0),
    "reg_alpha":         (0.0, 2.0),
    "reg_lambda":        (0.0, 2.0),
}

# Default LightGBM parameters used when Optuna tuning is skipped.
DEFAULT_PARAMS: dict = {
    "n_estimators":      800,
    "learning_rate":     0.02,
    "num_leaves":        50,
    "max_depth":         6,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "objective":         "regression_l1",
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}


class WalkForwardResult:
    """
    Container for the outputs of a walk-forward cross-validation run.

    Attributes
    ----------
    fold_metrics:
        List of per-fold metric dictionaries containing MAE, RMSE, and
        skill score.
    predictions:
        Concatenated out-of-sample predictions across all folds, as a
        UTC-indexed Series.
    actuals:
        Corresponding actual values aligned with predictions.
    mean_mae:
        Mean MAE across all folds.
    mean_rmse:
        Mean RMSE across all folds.
    mean_skill:
        Mean skill score across all folds, relative to the climatology
        baseline fitted on each fold's training set.
    std_rmse:
        Standard deviation of per-fold RMSE, which measures forecast
        consistency rather than mean performance alone.
    """

    def __init__(
        self,
        fold_metrics: list[dict],
        predictions:  pd.Series,
        actuals:      pd.Series,
    ) -> None:
        self.fold_metrics = fold_metrics
        self.predictions  = predictions
        self.actuals      = actuals

        rmse_values         = [m["RMSE"]        for m in fold_metrics]
        self.mean_mae       = float(np.mean([m["MAE"]   for m in fold_metrics]))
        self.mean_rmse      = float(np.mean(rmse_values))
        self.mean_skill     = float(np.mean([m["skill_score"] for m in fold_metrics]))
        self.std_rmse       = float(np.std(rmse_values))

    def summary(self) -> pd.DataFrame:
        """
        Returns a per-fold metrics DataFrame suitable for display in a notebook.

        Returns
        -------
        pd.DataFrame
            One row per fold with period, MAE, RMSE, and skill score.
        """
        return pd.DataFrame(self.fold_metrics)

    def __repr__(self) -> str:
        return (
            f"WalkForwardResult("
            f"folds={len(self.fold_metrics)}, "
            f"mean_RMSE={self.mean_rmse:.4f}, "
            f"mean_skill={self.mean_skill:.4f}, "
            f"std_RMSE={self.std_rmse:.4f})"
        )


def build_lgbm_model(params: Optional[dict] = None) -> lgb.LGBMRegressor:
    """
    Constructs a LightGBM regressor with the provided or default parameters.
    The objective is set to L1 (mean absolute error) loss because temperature
    forecast errors are approximately Laplace-distributed with heavier tails
    than a Gaussian, making L1 a more appropriate loss than L2.

    Parameters
    ----------
    params:
        Hyperparameter dictionary. If None, DEFAULT_PARAMS is used.

    Returns
    -------
    lgb.LGBMRegressor
        Configured but unfitted LightGBM model.
    """
    effective_params = DEFAULT_PARAMS.copy()
    if params is not None:
        effective_params.update(params)
    return lgb.LGBMRegressor(**effective_params)


def run_walk_forward_cv(
    df:            pd.DataFrame,
    target_col:    str,
    feature_cols:  list[str],
    n_splits:      int = 8,
    initial_train_frac: float = 0.5,
    params:        Optional[dict] = None,
    early_stopping_rounds: int = 50,
) -> WalkForwardResult:
    """
    Runs walk-forward expanding-window cross-validation on the full dataset.

    Each split trains on all data up to the split point and tests on the
    next contiguous block. The training window expands with each split.
    This mirrors how a live forecasting system would operate: always trained
    on everything seen so far, always predicting the future.

    A ClimatologyBaseline is fitted on each fold's training set independently
    to compute fold-level skill scores that are not contaminated by future
    climatological information.

    Parameters
    ----------
    df:
        Full feature and target DataFrame, UTC-indexed, sorted chronologically.
    target_col:
        Name of the target column.
    feature_cols:
        List of feature column names to use as model inputs.
    n_splits:
        Number of cross-validation folds.
    initial_train_frac:
        Fraction of the dataset used as the minimum training window for the
        first fold. 0.5 means the first fold trains on the first 50% of data
        and tests on the following block.
    params:
        LightGBM hyperparameters. If None, DEFAULT_PARAMS is used.
    early_stopping_rounds:
        Number of rounds without validation improvement before stopping.

    Returns
    -------
    WalkForwardResult
        Container holding per-fold metrics and concatenated out-of-sample
        predictions.
    """
    n = len(df)
    initial_size = int(n * initial_train_frac)
    block_size   = int((n - initial_size) / n_splits)

    if block_size < 10:
        raise ValueError(
            f"Block size of {block_size} rows is too small for meaningful evaluation. "
            f"Reduce n_splits or increase initial_train_frac."
        )

    fold_metrics    = []
    all_predictions = []
    all_actuals     = []

    for fold in range(n_splits):
        train_end = initial_size + fold * block_size
        test_start = train_end
        test_end   = min(test_start + block_size, n)

        if test_start >= n:
            break

        train_slice = df.iloc[:train_end]
        test_slice  = df.iloc[test_start:test_end]

        X_tr = train_slice[feature_cols]
        y_tr = train_slice[target_col]
        X_te = test_slice[feature_cols]
        y_te = test_slice[target_col]

        model = build_lgbm_model(params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        preds = model.predict(X_te)
        mae   = float(mean_absolute_error(y_te, preds))
        rmse  = float(np.sqrt(mean_squared_error(y_te, preds)))

        clim = ClimatologyBaseline()
        clim.fit(pd.DataFrame(index=y_tr.index), y_tr)
        clim_preds = clim.predict(pd.DataFrame(index=y_te.index))
        clim_rmse  = float(np.sqrt(mean_squared_error(y_te, clim_preds)))
        skill      = compute_skill_score(rmse, clim_rmse)

        fold_metrics.append({
            "fold":        fold + 1,
            "train_rows":  len(X_tr),
            "test_rows":   len(X_te),
            "period_start": str(test_slice.index[0].date()),
            "period_end":   str(test_slice.index[-1].date()),
            "MAE":          round(mae,   4),
            "RMSE":         round(rmse,  4),
            "skill_score":  round(skill, 4),
        })

        all_predictions.append(pd.Series(preds, index=y_te.index))
        all_actuals.append(y_te)

        logger.info(
            "Fold %d/%d: RMSE=%.4f, skill=%.4f, period=%s to %s.",
            fold + 1, n_splits, rmse, skill,
            test_slice.index[0].date(), test_slice.index[-1].date(),
        )

    predictions_combined = pd.concat(all_predictions)
    actuals_combined     = pd.concat(all_actuals)

    result = WalkForwardResult(fold_metrics, predictions_combined, actuals_combined)
    logger.info("Walk-forward CV complete: %s", result)
    return result


def tune_hyperparameters(
    df:            pd.DataFrame,
    target_col:    str,
    feature_cols:  list[str],
    n_trials:      int = 50,
    n_cv_splits:   int = 5,
    timeout_secs:  Optional[int] = 600,
) -> dict:
    """
    Uses Optuna to find the hyperparameter configuration that minimises
    mean walk-forward RMSE. Each trial runs a condensed walk-forward CV
    with n_cv_splits folds to speed up the search.

    Optuna uses Tree-structured Parzen Estimator (TPE) by default, which is
    a Bayesian optimisation approach that is substantially more efficient than
    grid or random search for this number of hyperparameters.

    Parameters
    ----------
    df:
        Full feature and target DataFrame, sorted chronologically.
    target_col:
        Name of the target column.
    feature_cols:
        Feature column names.
    n_trials:
        Number of Optuna trials to run. 50 gives a reasonable exploration of
        the space in under 10 minutes on a standard laptop.
    n_cv_splits:
        Number of walk-forward folds used within each trial's evaluation.
        Fewer folds than the final CV run for speed.
    timeout_secs:
        Maximum number of seconds to allow for the full optimisation run.
        The search stops at whichever limit is reached first.

    Returns
    -------
    dict
        Best hyperparameter configuration found. Can be passed directly to
        build_lgbm_model or run_walk_forward_cv.
    """
    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "n_estimators":      trial.suggest_int("n_estimators",
                                     *OPTUNA_PARAM_SPACE["n_estimators"]),
            "learning_rate":     trial.suggest_float("learning_rate",
                                     *OPTUNA_PARAM_SPACE["learning_rate"], log=True),
            "num_leaves":        trial.suggest_int("num_leaves",
                                     *OPTUNA_PARAM_SPACE["num_leaves"]),
            "max_depth":         trial.suggest_int("max_depth",
                                     *OPTUNA_PARAM_SPACE["max_depth"]),
            "min_child_samples": trial.suggest_int("min_child_samples",
                                     *OPTUNA_PARAM_SPACE["min_child_samples"]),
            "subsample":         trial.suggest_float("subsample",
                                     *OPTUNA_PARAM_SPACE["subsample"]),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",
                                     *OPTUNA_PARAM_SPACE["colsample_bytree"]),
            "reg_alpha":         trial.suggest_float("reg_alpha",
                                     *OPTUNA_PARAM_SPACE["reg_alpha"]),
            "reg_lambda":        trial.suggest_float("reg_lambda",
                                     *OPTUNA_PARAM_SPACE["reg_lambda"]),
            "objective":         "regression_l1",
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }

        result = run_walk_forward_cv(
            df, target_col, feature_cols,
            n_splits=n_cv_splits,
            params=trial_params,
            early_stopping_rounds=30,
        )
        return result.mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_secs)

    best_params = study.best_params
    best_params.update({
        "objective":  "regression_l1",
        "random_state": 42,
        "n_jobs":     -1,
        "verbose":    -1,
    })

    logger.info(
        "Optuna tuning complete. Best RMSE=%.4f with params: %s.",
        study.best_value, best_params,
    )
    return best_params


def fit_final_model(
    X_train:  pd.DataFrame,
    y_train:  pd.Series,
    X_val:    pd.DataFrame,
    y_val:    pd.Series,
    params:   Optional[dict] = None,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMRegressor:
    """
    Fits a final LightGBM model on the full training set with a held-out
    validation set for early stopping. This is the production model used
    for generating forecasts and SHAP explanations.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Training target Series.
    X_val:
        Validation feature matrix for early stopping.
    y_val:
        Validation target Series.
    params:
        Hyperparameter dictionary. If None, DEFAULT_PARAMS is used.
    early_stopping_rounds:
        Rounds without validation improvement before stopping.

    Returns
    -------
    lgb.LGBMRegressor
        Fitted model instance.
    """
    model = build_lgbm_model(params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    logger.info(
        "Final model fitted. Best iteration: %d.",
        model.best_iteration_,
    )
    return model


def compute_shap_values(
    model:    lgb.LGBMRegressor,
    X:        pd.DataFrame,
) -> tuple[np.ndarray, shap.TreeExplainer]:
    """
    Computes SHAP values for the provided feature matrix using a
    TreeExplainer, which is exact and fast for tree-based models.

    SHAP values provide the explanation layer that makes this model legible
    to a non-meteorologist PM. Rather than saying "feature X has importance
    score 0.12", SHAP allows statements such as "on this particular day, the
    NAO index at lag 7 days contributed minus 0.8 degrees to the forecast
    relative to the model's baseline prediction, because it was strongly
    negative."

    Parameters
    ----------
    model:
        Fitted LightGBM model.
    X:
        Feature matrix for which SHAP values are to be computed.

    Returns
    -------
    tuple[np.ndarray, shap.TreeExplainer]
        SHAP value array (shape: n_samples x n_features) and the explainer
        object, which holds the expected value (model baseline prediction).
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


def build_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Summarises SHAP values as mean absolute feature importances, ranked
    from most to least important.

    This table is the primary deliverable for the PM explainability section
    of the commodity linkage notebook. The most important features are
    expected to be HDD accumulation anomalies, NAO lag features, and
    blocking persistence indices.

    Parameters
    ----------
    shap_values:
        SHAP value array from compute_shap_values.
    feature_names:
        List of feature names corresponding to columns of the SHAP array.
    top_n:
        Number of top features to include in the summary.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'feature' and 'mean_abs_shap', sorted
        by importance descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    summary  = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return summary


def build_meteorological_attribution(
    shap_values:   np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
    index:         pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Aggregates SHAP values by meteorological feature group to produce a
    higher-level attribution that a PM can read without needing to
    understand the individual features.

    For example, grouping all NAO-related features into a single "NAO regime"
    attribution and all HDD features into a "thermal demand" attribution
    produces a table that directly answers the question: "what is driving
    this forecast?"

    Parameters
    ----------
    shap_values:
        SHAP value array, shape (n_samples, n_features).
    feature_names:
        List of feature names.
    feature_groups:
        Dictionary mapping group names to lists of feature names.
        Features not assigned to any group are placed in an 'Other' group.
        Example: {"NAO regime": ["nao", "nao_7d_mean", "nao_lag14d"]}
    index:
        DatetimeIndex for the output DataFrame rows.

    Returns
    -------
    pd.DataFrame
        Attribution DataFrame with one column per group, indexed by date.
        Values are summed SHAP contributions within each group, representing
        the total directional contribution of that meteorological category
        to the model's deviation from its baseline prediction.
    """
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=index)

    attributed_features: set[str] = set()
    group_contributions: dict[str, pd.Series] = {}

    for group_name, group_features in feature_groups.items():
        present = [f for f in group_features if f in shap_df.columns]
        if present:
            group_contributions[group_name] = shap_df[present].sum(axis=1)
            attributed_features.update(present)

    remaining = [f for f in feature_names if f not in attributed_features]
    if remaining:
        group_contributions["Other"] = shap_df[remaining].sum(axis=1)

    return pd.DataFrame(group_contributions, index=index)


def scale_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Applies StandardScaler to the feature matrices. LightGBM does not require
    scaling since it uses rank-based splitting, but scaling is applied here
    for consistency with the NGBoost model in ngboost_prob.py, which shares
    the same feature matrix.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    X_test:
        Test feature matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, StandardScaler]
        Scaled training matrix, scaled test matrix, and the fitted scaler.
        The scaler should be persisted alongside the model for use at
        inference time.
    """
    scaler           = StandardScaler()
    X_train_scaled   = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled    = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def save_model(
    model:    lgb.LGBMRegressor,
    path:     Path,
) -> None:
    """
    Saves a fitted LightGBM model to disk using the native LightGBM text format.

    Parameters
    ----------
    model:
        Fitted LightGBM model.
    path:
        File path at which to save the model. A '.txt' extension is conventional.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))
    logger.info("LightGBM model saved to %s.", path)


def load_model(path: Path) -> lgb.LGBMRegressor:
    """
    Loads a LightGBM model from disk.

    Parameters
    ----------
    path:
        Path to the saved model text file.

    Returns
    -------
    lgb.LGBMRegressor
        Model instance with the loaded booster. Note that sklearn wrapper
        attributes such as best_iteration_ will not be available on a loaded
        model; only the booster's predict method is restored.
    """
    path  = Path(path)
    model = lgb.LGBMRegressor()
    model._Booster = lgb.Booster(model_file=str(path))
    logger.info("LightGBM model loaded from %s.", path)
    return model
