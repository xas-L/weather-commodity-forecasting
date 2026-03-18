"""
Implementation for the climatological and NWP-direct baselines against which all
machine learning models in this project are benchmarked.

Having a credible baseline is not optional. Any model evaluation that does not
include a skill score relative to climatology is incomplete, and a PhD reviewer
or a trading PM will ask for it immediately. A model with R2 of 0.85 against
a baseline that also scores 0.83 has added almost no value; the same R2 against
a baseline that scores 0.60 represents genuine skill.

Three baselines are provided here:

    ClimatologyBaseline
        Predicts the historical mean for each calendar day-of-year, computed
        from the training window. This is the simplest possible forecast and
        represents the floor of useful performance. Any model that cannot beat
        it should not be deployed.

    PersistenceBaseline
        Predicts tomorrow's value as equal to today's observed value. This is
        a strong baseline for short-range temperature forecasting because
        temperatures are autocorrelated at lag 1, especially in winter. A model
        must beat persistence at all horizons to demonstrate that it adds value
        beyond the most naive extrapolation.

    NWPDirectBaseline
        Wraps the Open-Meteo NWP forecast as a baseline model. This represents
        the skill of an uncorrected operational NWP system, which is the correct
        benchmark for a post-processing model. A post-processing model that
        cannot improve on raw NWP output has not justified its complexity.

All baselines implement the same fit / predict interface so they can be used
interchangeably with the sklearn-compatible models in this project.

Setup required before use:
    pip install pandas numpy scikit-learn
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class ClimatologyBaseline(BaseEstimator, RegressorMixin):
    """
    Predicts the training-set mean for each calendar day-of-year.

    This is the weakest credible forecast. The skill score of any other model
    is defined relative to this baseline: skill = 1 - (RMSE_model / RMSE_clim).
    A positive skill score means the model outperforms climatology. A negative
    skill score means it is worse than ignoring all dynamics and predicting the
    seasonal mean.

    The day-of-year climatology is fit on training data only. Any use of the
    full dataset to compute the climatology would introduce forward-looking bias
    in the evaluation, since future observations would influence the baseline
    for past dates.

    Parameters
    ----------
    smoothing_window:
        Number of days over which to smooth the day-of-year climatology using
        a centred rolling mean. A window of 15 removes synoptic-scale noise
        from the climatological profile while preserving seasonal structure.
        Set to 1 to disable smoothing.
    """

    def __init__(self, smoothing_window: int = 15) -> None:
        self.smoothing_window = smoothing_window
        self._climatology: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClimatologyBaseline":
        """
        Computes the day-of-year mean from the training labels.

        Parameters
        ----------
        X:
            Feature matrix. Not used by this model; only the index is required
            to extract day-of-year values. Included for sklearn compatibility.
        y:
            Training target Series with a DatetimeIndex.

        Returns
        -------
        ClimatologyBaseline
            Fitted instance.
        """
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("Target Series must have a DatetimeIndex.")

        doy = y.index.dayofyear
        clim = y.groupby(doy).mean()

        if self.smoothing_window > 1:
            # Wrap the series at both ends before smoothing to avoid edge artefacts
            # at the start and end of the calendar year.
            repeated = pd.concat([clim, clim, clim])
            smoothed = repeated.rolling(
                self.smoothing_window, center=True, min_periods=1
            ).mean()
            clim = smoothed.iloc[len(clim):2 * len(clim)]
            clim.index = range(1, len(clim) + 1)

        self._climatology = clim
        logger.info(
            "ClimatologyBaseline fitted on %d observations, smoothing_window=%d.",
            len(y), self.smoothing_window,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the climatological mean for each day-of-year in the index of X.

        Parameters
        ----------
        X:
            Feature matrix with a DatetimeIndex.

        Returns
        -------
        np.ndarray
            Array of climatological predictions, one per row of X.
        """
        if self._climatology is None:
            raise RuntimeError("Call fit() before predict().")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("X must have a DatetimeIndex.")

        doy_values  = X.index.dayofyear
        predictions = np.array([
            self._climatology.get(d, self._climatology.mean())
            for d in doy_values
        ])
        return predictions

    def get_climatology(self) -> pd.Series:
        """
        Returns the fitted day-of-year climatology Series for inspection
        or for use in anomaly calculations downstream.

        Returns
        -------
        pd.Series
            Day-of-year mean values, indexed from 1 to 366.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self._climatology is None:
            raise RuntimeError("Call fit() before get_climatology().")
        return self._climatology.copy()


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """
    Predicts the target at horizon h as the last observed value before the
    forecast is made.

    At a lag-1 horizon this is simply yesterday's temperature. At longer
    horizons the same last-observed value is repeated. This baseline is
    particularly competitive for short-range temperature forecasting in
    winter due to the strong day-to-day autocorrelation of cold air masses.

    Parameters
    ----------
    lag:
        Number of time steps by which to shift the observed values.
        A lag of 1 corresponds to a naive day-ahead forecast.
        A lag of 7 corresponds to a week-ahead persistence forecast.
    """

    def __init__(self, lag: int = 1) -> None:
        self.lag = lag
        self._last_values: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceBaseline":
        """
        Stores the lagged target values from the training set for use
        at prediction time.

        Parameters
        ----------
        X:
            Feature matrix. Not used; included for sklearn compatibility.
        y:
            Training target Series.

        Returns
        -------
        PersistenceBaseline
            Fitted instance.
        """
        self._last_values = y.copy()
        logger.info("PersistenceBaseline fitted with lag=%d.", self.lag)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the lagged observed value for each row in X.

        If X has a DatetimeIndex that aligns with the training labels, the
        lag is applied relative to the training series. For out-of-sample
        prediction where the test index follows the training index, this
        effectively predicts each test value as the training observation
        lag steps before the test timestamp.

        Parameters
        ----------
        X:
            Feature matrix with a DatetimeIndex.

        Returns
        -------
        np.ndarray
            Persistence predictions for each row of X.
        """
        if self._last_values is None:
            raise RuntimeError("Call fit() before predict().")

        lagged = self._last_values.shift(self.lag)
        predictions = lagged.reindex(X.index)

        if predictions.isna().any():
            n_missing = predictions.isna().sum()
            logger.warning(
                "PersistenceBaseline: %d predictions could not be aligned "
                "to the training series and will be filled with the training mean.",
                n_missing,
            )
            predictions = predictions.fillna(self._last_values.mean())

        return predictions.to_numpy()


class NWPDirectBaseline(BaseEstimator, RegressorMixin):
    """
    Wraps a pre-computed NWP forecast Series as a baseline model.

    This represents the skill of the raw operational NWP output before any
    statistical post-processing. It is the most demanding baseline for a
    post-processing model because it already incorporates the physical dynamics
    of the atmosphere on the relevant forecast horizon.

    In this project, the NWP forecast comes from Open-Meteo (GFS or ECMWF
    open data). The ML models in lgbm_forecaster.py and ngboost_prob.py are
    evaluated against this baseline as well as against climatology.

    Parameters
    ----------
    forecast_col:
        Name of the column in X that contains the NWP forecast values.
        If None, the first column of X is used.
    """

    def __init__(self, forecast_col: Optional[str] = None) -> None:
        self.forecast_col = forecast_col

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NWPDirectBaseline":
        """
        No fitting is required for the NWP baseline; the forecast is read
        directly from the feature matrix at prediction time.

        Parameters
        ----------
        X:
            Feature matrix containing the NWP forecast column.
        y:
            Training target Series. Not used.

        Returns
        -------
        NWPDirectBaseline
            Fitted instance.
        """
        col = self.forecast_col or X.columns[0]
        if col not in X.columns:
            raise KeyError(
                f"NWP forecast column '{col}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )
        self._col = col
        logger.info("NWPDirectBaseline using column '%s'.", self._col)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the NWP forecast column from X as the prediction array.

        Parameters
        ----------
        X:
            Feature matrix containing the NWP forecast column.

        Returns
        -------
        np.ndarray
            NWP forecast values for each row of X.
        """
        if not hasattr(self, "_col"):
            raise RuntimeError("Call fit() before predict().")
        return X[self._col].to_numpy()


def compute_skill_score(
    rmse_model: float,
    rmse_baseline: float,
) -> float:
    """
    Computes the deterministic skill score of a model relative to a baseline.

    Skill = 1 - (RMSE_model / RMSE_baseline)

    Interpretation:
        1.0   perfect forecast
        0.0   no improvement over the baseline
        < 0   worse than the baseline

    This is the standard metric used in NWP verification and in operational
    weather forecast evaluation. Including it in the results table is what
    distinguishes a project built with meteorological understanding from one
    built purely as a machine learning exercise.

    Parameters
    ----------
    rmse_model:
        RMSE of the model being evaluated.
    rmse_baseline:
        RMSE of the baseline (typically climatology).

    Returns
    -------
    float
        Skill score.

    Raises
    ------
    ValueError
        If rmse_baseline is zero, which would produce a division error.
    """
    if rmse_baseline == 0.0:
        raise ValueError(
            "Baseline RMSE is zero. Cannot compute a meaningful skill score."
        )
    return 1.0 - (rmse_model / rmse_baseline)


def evaluate_baselines(
    y_train: pd.Series,
    y_test: pd.Series,
    X_test: pd.DataFrame,
    nwp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fits all available baselines on the training set and evaluates them on
    the test set. Returns a results DataFrame that serves as the benchmark
    table in the model comparison notebook.

    Parameters
    ----------
    y_train:
        Training target Series with DatetimeIndex.
    y_test:
        Test target Series with DatetimeIndex.
    X_test:
        Test feature matrix. Used by PersistenceBaseline and NWPDirectBaseline.
    nwp_col:
        Column name in X_test containing the NWP forecast. If None, the NWP
        baseline is omitted from the results.

    Returns
    -------
    pd.DataFrame
        Results table with one row per baseline and columns for MAE, RMSE,
        R2, and skill score relative to climatology.
    """
    X_train_dummy = pd.DataFrame(index=y_train.index)

    clim        = ClimatologyBaseline().fit(X_train_dummy, y_train)
    persistence = PersistenceBaseline(lag=1).fit(X_train_dummy, y_train)

    baselines = {
        "Climatology":   clim,
        "Persistence":   persistence,
    }

    if nwp_col is not None and nwp_col in X_test.columns:
        nwp = NWPDirectBaseline(forecast_col=nwp_col).fit(X_test, y_test)
        baselines["NWP Direct"] = nwp

    records = []
    clim_pred = clim.predict(X_test if isinstance(X_test, pd.DataFrame)
                             else pd.DataFrame(index=y_test.index))
    clim_rmse = float(np.sqrt(mean_squared_error(y_test, clim_pred)))

    for name, model in baselines.items():
        preds = model.predict(X_test)
        mae   = float(mean_absolute_error(y_test, preds))
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        ss    = compute_skill_score(rmse, clim_rmse)

        ss_display = round(ss, 4) if name != "Climatology" else 0.0

        records.append({
            "model":       name,
            "MAE":         round(mae,  4),
            "RMSE":        round(rmse, 4),
            "skill_score": ss_display,
        })

    results = pd.DataFrame(records).set_index("model")
    logger.info("Baseline evaluation complete:\n%s", results.to_string())
    return results
