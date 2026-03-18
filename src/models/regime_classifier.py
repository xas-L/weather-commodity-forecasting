"""
Implementation of a subseasonal temperature regime classifier that predicts whether
NW Europe will be in a cold, neutral, or warm regime at week-2 to week-4
lead times, based on current teleconnection indices and atmospheric circulation
state.

The classifier answers a specific question asked by a gas trading desk: given
today's atmospheric state, what is the probability that temperatures will be
below seasonal norms in Germany and the Netherlands in two to four weeks?

This is a classification problem, not a regression problem. At subseasonal
lead times (10-28 days) the atmosphere has lost most of its deterministic
predictability. The appropriate output is a probability distribution over
regime categories, not a temperature point forecast. The commercial value of
this model is in the probability of the cold regime specifically, because that
probability translates directly into an expected HDD anomaly and a corresponding
gas demand signal.

XGBoost is used here rather than LightGBM because it produces marginally better
probability calibration for multi-class problems on this dataset size, and
because using a different library from the short-term model creates an
architectural distinction that is easier to explain to a reviewer.

The three regime classes are:
    cold      Weekly temperature anomaly more than 0.8 standard deviations
              below the seasonal mean. Implies HDD above seasonal norms.
              Associated with negative NAO, potential blocking, elevated
              TTF gas demand.
    neutral   Weekly temperature anomaly within 0.8 standard deviations
              of the seasonal mean.
    warm      Weekly temperature anomaly more than 0.8 standard deviations
              above the seasonal mean. Implies HDD below seasonal norms.

Calibration is assessed using the Brier score for the cold regime probability,
and via the reliability diagram from ngboost_prob.py adapted for classification
outputs. A well-calibrated classifier with Brier skill score above 0.1 relative
to the naive climatological prior provides useful subseasonal information.

Setup required before use:
    pip install xgboost scikit-learn pandas numpy optuna
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Regime class labels and their integer encodings.
REGIME_LABELS:   list[str] = ["cold", "neutral", "warm"]
REGIME_ENCODING: dict[str, int] = {"cold": 0, "neutral": 1, "warm": 2}
REGIME_DECODING: dict[int, str] = {v: k for k, v in REGIME_ENCODING.items()}

# The cold regime is the commercially relevant class for a gas desk.
# All Brier scores and AUC computations target this class unless specified.
COLD_CLASS_IDX: int = 0

# Default XGBoost parameters for the regime classifier.
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators":       400,
    "learning_rate":      0.03,
    "max_depth":          4,
    "min_child_weight":   5,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "gamma":              0.1,
    "reg_alpha":          0.5,
    "reg_lambda":         1.0,
    "objective":          "multi:softprob",
    "num_class":          3,
    "eval_metric":        "mlogloss",
    "use_label_encoder":  False,
    "random_state":       42,
    "n_jobs":             -1,
    "verbosity":          0,
}


class RegimeClassifier:
    """
    Wraps an XGBoost multi-class classifier with probability calibration and
    convenience methods for extracting cold-regime probabilities.

    Isotonic calibration is applied post-fit using cross-validation on the
    training set to correct the known tendency of gradient boosting models to
    produce overconfident probability estimates. For the subseasonal forecasting
    use case, calibration matters more than raw classification accuracy because
    the output probability is used directly for position sizing.

    Parameters
    ----------
    params:
        XGBoost hyperparameters. If None, DEFAULT_XGB_PARAMS is used.
    calibrate:
        If True, apply isotonic probability calibration on the training set.
    calibration_cv:
        Number of cross-validation folds used for isotonic calibration.
    """

    def __init__(
        self,
        params:         Optional[dict] = None,
        calibrate:      bool = True,
        calibration_cv: int  = 5,
    ) -> None:
        self.params         = params or DEFAULT_XGB_PARAMS.copy()
        self.calibrate      = calibrate
        self.calibration_cv = calibration_cv
        self._model:        Optional[xgb.XGBClassifier]         = None
        self._calibrated:   Optional[CalibratedClassifierCV]    = None
        self._scaler:       Optional[StandardScaler]            = None
        self._label_enc:    Optional[LabelEncoder]              = None
        self._classes:      Optional[list[str]]                 = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[pd.Series]    = None,
    ) -> "RegimeClassifier":
        """
        Fits the classifier on the training data, with optional early stopping
        on a validation set. If calibrate=True, isotonic calibration is applied
        using TimeSeriesSplit CV on the training set after the initial fit.

        Parameters
        ----------
        X_train:
            Training feature matrix.
        y_train:
            Training target Series with values in {'cold', 'neutral', 'warm'}.
        X_val:
            Optional validation feature matrix for early stopping.
        y_val:
            Optional validation target Series.

        Returns
        -------
        RegimeClassifier
            Fitted instance.
        """
        self._scaler = StandardScaler()
        X_tr_sc = pd.DataFrame(
            self._scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        self._label_enc = LabelEncoder()
        y_encoded = self._label_enc.fit_transform(y_train.map(REGIME_ENCODING))
        self._classes = [REGIME_DECODING[i] for i in range(len(REGIME_LABELS))]

        effective_params = self.params.copy()
        model = xgb.XGBClassifier(**effective_params)

        if X_val is not None and y_val is not None:
            X_val_sc = pd.DataFrame(
                self._scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index,
            )
            y_val_enc = self._label_enc.transform(y_val.map(REGIME_ENCODING))
            model.fit(
                X_tr_sc, y_encoded,
                eval_set=[(X_val_sc, y_val_enc)],
                verbose=False,
            )
        else:
            model.fit(X_tr_sc, y_encoded)

        self._model = model

        if self.calibrate:
            tscv = TimeSeriesSplit(n_splits=self.calibration_cv)
            self._calibrated = CalibratedClassifierCV(
                estimator=self._model,
                method="isotonic",
                cv=tscv,
            )
            self._calibrated.fit(X_tr_sc, y_encoded)
            logger.info("RegimeClassifier fitted with isotonic calibration.")
        else:
            logger.info("RegimeClassifier fitted without calibration.")

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns class probabilities for each observation in X.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'cold', 'neutral', 'warm' and probabilities
            summing to 1 for each row, indexed by X.index.
        """
        if self._scaler is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X_sc = pd.DataFrame(
            self._scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )

        predictor = self._calibrated if self._calibrated is not None else self._model
        proba     = predictor.predict_proba(X_sc)

        return pd.DataFrame(proba, columns=REGIME_LABELS, index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns the most likely regime class for each observation.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        pd.Series
            Predicted regime labels ('cold', 'neutral', or 'warm').
        """
        proba        = self.predict_proba(X)
        predicted    = proba.idxmax(axis=1)
        predicted.name = "predicted_regime"
        return predicted

    def cold_probability(self, X: pd.DataFrame) -> pd.Series:
        """
        Extracts the cold-regime probability from the full probability matrix.
        This is the primary output used by the commodity linkage notebook.

        A cold probability of 0.65 at a two-week lead means that the model
        assigns 65% probability to NW European temperatures being more than
        0.8 standard deviations below seasonal norms in that week, which
        corresponds to an expected positive HDD anomaly and elevated gas demand.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        pd.Series
            Cold regime probability for each row of X, between 0 and 1.
        """
        return self.predict_proba(X)["cold"].rename("cold_regime_probability")

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Returns XGBoost feature importances sorted by gain.

        Parameters
        ----------
        feature_names:
            List of feature names corresponding to the model's input columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'feature' and 'importance_gain' columns, sorted
            descending by gain.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before feature_importance().")

        scores = self._model.get_booster().get_score(importance_type="gain")
        df = pd.DataFrame(
            list(scores.items()), columns=["feature", "importance_gain"]
        ).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        return df


def encode_regime_target(y: pd.Series) -> pd.Series:
    """
    Converts string regime labels to integer codes for XGBoost training.

    Parameters
    ----------
    y:
        Target Series with values in {'cold', 'neutral', 'warm'}.

    Returns
    -------
    pd.Series
        Integer-encoded target (0=cold, 1=neutral, 2=warm).
    """
    return y.map(REGIME_ENCODING)


def compute_brier_score(
    y_actual:        pd.Series,
    cold_probability: pd.Series,
) -> float:
    """
    Computes the Brier score for the cold regime probability forecast.

    The Brier score is the mean squared error between the predicted probability
    and the binary outcome (1 if cold, 0 otherwise). Lower is better.

    A Brier score of 0.25 corresponds to always predicting 50% probability,
    i.e. climatological uncertainty. A Brier score approaching 0 indicates
    near-perfect probabilistic predictions.

    Parameters
    ----------
    y_actual:
        Observed regime labels ('cold', 'neutral', 'warm').
    cold_probability:
        Predicted probability of the cold regime, between 0 and 1.

    Returns
    -------
    float
        Brier score for the cold regime.
    """
    binary_cold = (y_actual == "cold").astype(int)
    aligned     = cold_probability.reindex(binary_cold.index)
    return float(brier_score_loss(binary_cold, aligned))


def compute_brier_skill_score(
    brier_model:       float,
    y_train:           pd.Series,
) -> float:
    """
    Computes the Brier Skill Score relative to a climatological reference.

    The reference Brier score is computed from the naive forecast of always
    predicting the training-set base rate for the cold regime. A BSS above
    zero indicates that the model beats this climatological prior.

    BSS = 1 - (Brier_model / Brier_reference)

    Parameters
    ----------
    brier_model:
        Brier score of the regime classifier.
    y_train:
        Training target Series used to estimate the cold base rate.

    Returns
    -------
    float
        Brier Skill Score. Positive values indicate improvement over the prior.
    """
    cold_base_rate = float((y_train == "cold").mean())

    # Reference: always predicting the base rate probability
    n_test         = 1   # relative measure, actual n cancels out
    brier_ref      = cold_base_rate * (1 - cold_base_rate) ** 2 + \
                     (1 - cold_base_rate) * cold_base_rate ** 2

    if brier_ref == 0.0:
        logger.warning("Reference Brier score is zero. Returning BSS of 0.")
        return 0.0

    return float(1.0 - (brier_model / brier_ref))


def evaluate_regime_classifier(
    y_actual:         pd.Series,
    y_predicted:      pd.Series,
    y_proba:          pd.DataFrame,
    y_train:          pd.Series,
) -> dict:
    """
    Computes a full suite of evaluation metrics for the regime classifier.

    Metrics computed:
        Classification report (precision, recall, F1 per class)
        Brier score for cold regime probability
        Brier Skill Score relative to climatological prior
        Multi-class log loss
        AUC for cold regime (one-vs-rest)

    Parameters
    ----------
    y_actual:
        Observed regime labels.
    y_predicted:
        Predicted regime labels (from RegimeClassifier.predict).
    y_proba:
        Probability DataFrame (from RegimeClassifier.predict_proba).
    y_train:
        Training labels used for reference metrics.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    report     = classification_report(y_actual, y_predicted, output_dict=True)
    brier      = compute_brier_score(y_actual, y_proba["cold"])
    bss        = compute_brier_skill_score(brier, y_train)
    logloss    = float(log_loss(y_actual, y_proba[REGIME_LABELS].to_numpy()))

    binary_cold = (y_actual == "cold").astype(int)
    auc_cold    = float(roc_auc_score(binary_cold, y_proba["cold"]))

    metrics = {
        "brier_score_cold":        round(brier,   4),
        "brier_skill_score":       round(bss,     4),
        "log_loss":                round(logloss, 4),
        "auc_cold_regime":         round(auc_cold, 4),
        "cold_f1":                 round(report.get("cold", {}).get("f1-score", 0.0), 4),
        "cold_precision":          round(report.get("cold", {}).get("precision", 0.0), 4),
        "cold_recall":             round(report.get("cold", {}).get("recall", 0.0), 4),
        "macro_f1":                round(report.get("macro avg", {}).get("f1-score", 0.0), 4),
    }

    logger.info("Regime classifier evaluation: %s", metrics)
    return metrics


def run_regime_walk_forward_cv(
    df:            pd.DataFrame,
    target_col:    str,
    feature_cols:  list[str],
    n_splits:      int = 6,
    initial_train_frac: float = 0.5,
    params:        Optional[dict] = None,
) -> dict:
    """
    Runs walk-forward cross-validation for the regime classifier.

    Because the target is a weekly regime label, the dataset here is
    weekly-indexed and substantially smaller than the daily datasets
    used by the other models. With 8 years of weekly data the full dataset
    is approximately 420 observations, so the walk-forward blocks are
    correspondingly smaller.

    Parameters
    ----------
    df:
        Weekly-indexed feature and target DataFrame.
    target_col:
        Name of the regime label column.
    feature_cols:
        Feature column names.
    n_splits:
        Number of walk-forward folds.
    initial_train_frac:
        Fraction of data in the minimum training window.
    params:
        XGBoost hyperparameters. If None, DEFAULT_XGB_PARAMS is used.

    Returns
    -------
    dict
        Dictionary with keys 'fold_metrics', 'all_predictions',
        'all_probabilities', and 'all_actuals'.
    """
    n            = len(df)
    initial_size = int(n * initial_train_frac)
    block_size   = max(4, int((n - initial_size) / n_splits))

    fold_metrics:     list[dict]       = []
    all_predictions:  list[pd.Series]  = []
    all_probabilities: list[pd.DataFrame] = []
    all_actuals:      list[pd.Series]  = []

    for fold in range(n_splits):
        train_end  = initial_size + fold * block_size
        test_start = train_end
        test_end   = min(test_start + block_size, n)

        if test_start >= n or (test_end - test_start) < 2:
            break

        train_slice = df.iloc[:train_end]
        test_slice  = df.iloc[test_start:test_end]

        X_tr = train_slice[feature_cols].dropna()
        y_tr = train_slice.loc[X_tr.index, target_col]
        X_te = test_slice[feature_cols].dropna()
        y_te = test_slice.loc[X_te.index, target_col]

        if len(y_tr) < 20 or len(y_te) < 2:
            logger.warning("Fold %d skipped due to insufficient data.", fold + 1)
            continue

        clf = RegimeClassifier(params=params)
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_te)
        proba = clf.predict_proba(X_te)
        metrics = evaluate_regime_classifier(y_te, preds, proba, y_tr)
        metrics["fold"]         = fold + 1
        metrics["period_start"] = str(test_slice.index[0].date())
        metrics["period_end"]   = str(test_slice.index[-1].date())
        metrics["train_weeks"]  = len(X_tr)
        metrics["test_weeks"]   = len(X_te)

        fold_metrics.append(metrics)
        all_predictions.append(preds)
        all_probabilities.append(proba)
        all_actuals.append(y_te)

        logger.info(
            "Regime fold %d/%d: BSS=%.4f, AUC_cold=%.4f, cold_F1=%.4f.",
            fold + 1, n_splits,
            metrics["brier_skill_score"],
            metrics["auc_cold_regime"],
            metrics["cold_f1"],
        )

    return {
        "fold_metrics":      fold_metrics,
        "all_predictions":   pd.concat(all_predictions) if all_predictions else pd.Series(),
        "all_probabilities": pd.concat(all_probabilities) if all_probabilities else pd.DataFrame(),
        "all_actuals":       pd.concat(all_actuals) if all_actuals else pd.Series(),
    }


def tune_regime_hyperparameters(
    df:            pd.DataFrame,
    target_col:    str,
    feature_cols:  list[str],
    n_trials:      int = 40,
    n_cv_splits:   int = 4,
    timeout_secs:  Optional[int] = 300,
) -> dict:
    """
    Uses Optuna to minimise the mean walk-forward Brier score for the cold
    regime probability. The Brier score is used as the optimisation objective
    rather than accuracy or log loss because it directly measures calibrated
    probability quality for the regime class that matters commercially.

    Parameters
    ----------
    df:
        Weekly-indexed feature and target DataFrame.
    target_col:
        Regime label column name.
    feature_cols:
        Feature column names.
    n_trials:
        Number of Optuna trials.
    n_cv_splits:
        Walk-forward folds per trial (fewer than the final CV for speed).
    timeout_secs:
        Maximum time budget for the full search.

    Returns
    -------
    dict
        Best XGBoost hyperparameter configuration.
    """
    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 2, 7),
            "min_child_weight":  trial.suggest_int("min_child_weight", 3, 20),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma":             trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3.0),
            "objective":         "multi:softprob",
            "num_class":         3,
            "eval_metric":       "mlogloss",
            "use_label_encoder": False,
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         0,
        }

        cv_result = run_regime_walk_forward_cv(
            df, target_col, feature_cols,
            n_splits=n_cv_splits,
            params=trial_params,
        )

        fold_brier_scores = [
            m["brier_score_cold"] for m in cv_result["fold_metrics"]
        ]
        return float(np.mean(fold_brier_scores)) if fold_brier_scores else 1.0

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_secs)

    best_params = study.best_params
    best_params.update({
        "objective":         "multi:softprob",
        "num_class":         3,
        "eval_metric":       "mlogloss",
        "use_label_encoder": False,
        "random_state":      42,
        "n_jobs":            -1,
        "verbosity":         0,
    })

    logger.info(
        "Regime hyperparameter tuning complete. Best Brier=%.4f.",
        study.best_value,
    )
    return best_params
