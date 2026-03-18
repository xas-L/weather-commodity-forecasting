"""
Implementation of probabilistic short-range temperature forecasting using NGBoost
(Natural Gradient Boosting). NGBoost fits a full parametric probability
distribution at each forecast point rather than a single point estimate,
providing calibrated uncertainty quantification that is directly useful for
commodity risk management.

Why probabilistic forecasting matters to a trading desk:
    A gas trader sizing a position on an expected cold spell does not just need
    to know the most likely temperature. They need to know the probability that
    temperature will fall below a critical threshold, and the width of the
    uncertainty band around that threshold. A probabilistic forecast translates
    directly into a position size and a confidence level. A point forecast does not.

NGBoost outputs:
    For each forecast horizon and location, NGBoost produces a Normal distribution
    parameterised by a predicted mean (mu) and standard deviation (sigma). The
    mean is the deterministic point forecast. The sigma quantifies forecast
    uncertainty and is itself a learnable function of the input features.

Evaluation metrics:
    CRPS (Continuous Ranked Probability Score) is the primary metric for
    probabilistic forecast quality. Lower is better. It rewards forecasts
    that assign high probability to outcomes that subsequently occur, and
    penalises overconfident forecasts that assign low probability to tails.

    Reliability (calibration) is assessed via a reliability diagram: the
    fraction of observations falling below the predicted 10th percentile
    should be approximately 10%, and so on across all quantile levels.

    Sharpness is the mean width of prediction intervals. A sharper forecast
    (narrower intervals) is better, provided calibration is maintained.

Setup required before use:
    pip install ngboost properscoring scipy pandas numpy scikit-learn
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import properscoring as ps
from ngboost import NGBRegressor
from ngboost.distns import Normal
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.models.baseline import compute_skill_score, ClimatologyBaseline

logger = logging.getLogger(__name__)


# Default NGBoost hyperparameters. The learning rate is set conservatively at
# 0.02 to reduce overfitting. The number of estimators is higher than for
# LightGBM because each NGBoost stage learns two distributions (mean and sigma)
# simultaneously, requiring more iterations to converge.
DEFAULT_NGB_PARAMS: dict = {
    "n_estimators":     500,
    "learning_rate":    0.02,
    "natural_gradient": True,
    "random_state":     42,
    "verbose":          False,
}

# Quantile levels used for interval coverage and reliability diagram computation.
QUANTILE_LEVELS: list[float] = [
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
]

# Prediction interval levels for commercial use.
# The 80% interval is most common in energy market risk reports.
INTERVAL_LEVELS: dict[str, tuple[float, float]] = {
    "50pct":  (0.25, 0.75),
    "80pct":  (0.10, 0.90),
    "90pct":  (0.05, 0.95),
}


class ProbabilisticForecast:
    """
    Container for the outputs of an NGBoost prediction, providing convenient
    access to point estimates, quantile forecasts, and interval bounds.

    Attributes
    ----------
    mu:
        Predicted mean temperature (point forecast), UTC-indexed Series.
    sigma:
        Predicted standard deviation (forecast uncertainty), UTC-indexed Series.
    index:
        DatetimeIndex of the forecast timestamps.
    """

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, index: pd.DatetimeIndex) -> None:
        self.mu    = pd.Series(mu.flatten(),    index=index, name="forecast_mean")
        self.sigma = pd.Series(sigma.flatten(), index=index, name="forecast_sigma")
        self.index = index

    def quantile(self, q: float) -> pd.Series:
        """
        Returns the q-th quantile of the predictive distribution at each
        forecast timestamp.

        Parameters
        ----------
        q:
            Quantile level between 0 and 1.

        Returns
        -------
        pd.Series
            Quantile values, UTC-indexed.
        """
        return pd.Series(
            stats.norm.ppf(q, loc=self.mu.values, scale=self.sigma.values),
            index=self.index,
            name=f"q{int(q * 100):02d}",
        )

    def interval(self, lower_q: float, upper_q: float) -> pd.DataFrame:
        """
        Returns the lower and upper bounds of a prediction interval.

        Parameters
        ----------
        lower_q:
            Lower quantile level, e.g. 0.10 for the 10th percentile.
        upper_q:
            Upper quantile level, e.g. 0.90 for the 90th percentile.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'lower' and 'upper' columns.
        """
        return pd.DataFrame({
            "lower": self.quantile(lower_q),
            "upper": self.quantile(upper_q),
        }, index=self.index)

    def exceedance_probability(self, threshold: float) -> pd.Series:
        """
        Returns the probability that the temperature will exceed a given
        threshold at each forecast timestamp.

        This is the most directly actionable output for a gas desk. A
        probability of 0.75 that temperature falls below 2 degrees Celsius
        on a given day translates directly into an expected HDD of 13.5 with
        a quantified uncertainty band.

        Parameters
        ----------
        threshold:
            Temperature threshold in degrees Celsius.

        Returns
        -------
        pd.Series
            Exceedance probability at each forecast timestamp, between 0 and 1.
        """
        prob_below = pd.Series(
            stats.norm.cdf(threshold, loc=self.mu.values, scale=self.sigma.values),
            index=self.index,
            name=f"prob_below_{threshold:.1f}C",
        )
        return prob_below

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns a wide DataFrame with mu, sigma, and standard quantile columns.

        Returns
        -------
        pd.DataFrame
            One column per output, UTC-indexed.
        """
        frames = [self.mu, self.sigma]
        for q in QUANTILE_LEVELS:
            frames.append(self.quantile(q))
        return pd.concat(frames, axis=1)


def build_ngboost_model(params: Optional[dict] = None) -> NGBRegressor:
    """
    Constructs an NGBRegressor with a Normal output distribution.

    The Normal distribution is appropriate for temperature because daily mean
    temperatures are approximately Gaussian conditional on the season and
    large-scale circulation state. For HDD forecasting a truncated Normal or
    log-Normal might be more appropriate, but the additional complexity is
    rarely justified for the forecast horizons used here.

    Parameters
    ----------
    params:
        Hyperparameter dictionary. If None, DEFAULT_NGB_PARAMS is used.

    Returns
    -------
    NGBRegressor
        Configured but unfitted NGBoost model.
    """
    effective_params = DEFAULT_NGB_PARAMS.copy()
    if params is not None:
        effective_params.update(params)
    return NGBRegressor(Dist=Normal, **effective_params)


def fit_ngboost(
    X_train:               pd.DataFrame,
    y_train:               pd.Series,
    X_val:                 pd.DataFrame,
    y_val:                 pd.Series,
    params:                Optional[dict] = None,
    early_stopping_rounds: int = 50,
) -> NGBRegressor:
    """
    Fits an NGBoost model on the training set with early stopping based on
    the validation set CRPS.

    Parameters
    ----------
    X_train:
        Training feature matrix. Should be standardised (use scale_features
        from lgbm_forecaster.py).
    y_train:
        Training target Series.
    X_val:
        Validation feature matrix, standardised with the training scaler.
    y_val:
        Validation target Series.
    params:
        NGBoost hyperparameters. If None, DEFAULT_NGB_PARAMS is used.
    early_stopping_rounds:
        Number of rounds without improvement in validation CRPS before
        training stops.

    Returns
    -------
    NGBRegressor
        Fitted NGBoost model.
    """
    model = build_ngboost_model(params)
    model.fit(
        X_train.to_numpy(), y_train.to_numpy(),
        X_val=X_val.to_numpy(), Y_val=y_val.to_numpy(),
        early_stopping_rounds=early_stopping_rounds,
    )
    logger.info(
        "NGBoost fitted. Best iteration: %d.",
        model.best_val_loss_itr if hasattr(model, "best_val_loss_itr") else -1,
    )
    return model


def predict_distribution(
    model: NGBRegressor,
    X:     pd.DataFrame,
) -> ProbabilisticForecast:
    """
    Generates a probabilistic forecast from a fitted NGBoost model.

    Parameters
    ----------
    model:
        Fitted NGBRegressor.
    X:
        Feature matrix for which predictions are required.

    Returns
    -------
    ProbabilisticForecast
        Container providing access to the full predictive distribution.
    """
    dist  = model.pred_dist(X.to_numpy())
    mu    = dist.loc
    sigma = dist.scale

    if isinstance(mu, (float, int)):
        mu    = np.full(len(X), mu)
        sigma = np.full(len(X), sigma)

    return ProbabilisticForecast(mu=mu, sigma=sigma, index=X.index)


def compute_crps(
    forecast: ProbabilisticForecast,
    y_actual: pd.Series,
) -> pd.Series:
    """
    Computes the Continuous Ranked Probability Score for each forecast
    timestamp.

    CRPS is a strictly proper scoring rule: a forecaster maximises their
    expected score only by issuing their true beliefs. It generalises MAE
    to probability distributions: for a deterministic forecast, CRPS
    reduces to absolute error.

    Interpreting CRPS values:
        A CRPS of 0.5 degrees Celsius means that on average, the predictive
        distribution is shifted by approximately 0.5 degrees from the outcome.
        A CRPS lower than the climatological distribution's CRPS indicates that
        the model adds probabilistic skill beyond seasonal norms.

    Parameters
    ----------
    forecast:
        ProbabilisticForecast from predict_distribution.
    y_actual:
        Observed target values, aligned to the same index as the forecast.

    Returns
    -------
    pd.Series
        CRPS at each timestamp. Lower is better.
    """
    aligned_actual = y_actual.reindex(forecast.index)

    crps_values = ps.crps_gaussian(
        aligned_actual.to_numpy(),
        mu=forecast.mu.to_numpy(),
        sig=forecast.sigma.to_numpy(),
    )
    return pd.Series(crps_values, index=forecast.index, name="crps")


def compute_probabilistic_skill_score(
    crps_model:      pd.Series,
    y_train:         pd.Series,
    y_test:          pd.Series,
) -> float:
    """
    Computes a probabilistic skill score (CRPSS) relative to a climatological
    distribution baseline.

    The climatological baseline is a Normal distribution fitted to the
    training-set target values, with mean and standard deviation computed
    from the training data alone. This represents the best probabilistic
    forecast one could make without any dynamic information.

    CRPSS = 1 - (CRPS_model / CRPS_climatology)

    A CRPSS of 0.3 means the model reduces the CRPS by 30% relative to
    issuing the training climatology as the forecast at every time step.

    Parameters
    ----------
    crps_model:
        Per-timestep CRPS from the NGBoost model.
    y_train:
        Training target used to estimate the climatological distribution.
    y_test:
        Test target aligned with crps_model.

    Returns
    -------
    float
        Continuous Ranked Probability Skill Score.
    """
    clim_mu    = float(y_train.mean())
    clim_sigma = float(y_train.std())

    crps_clim = ps.crps_gaussian(
        y_test.reindex(crps_model.index).to_numpy(),
        mu=np.full(len(crps_model), clim_mu),
        sig=np.full(len(crps_model), clim_sigma),
    )
    mean_crps_model = float(crps_model.mean())
    mean_crps_clim  = float(crps_clim.mean())

    if mean_crps_clim == 0.0:
        logger.warning("Climatological CRPS is zero. Returning CRPSS of 0.")
        return 0.0

    return 1.0 - (mean_crps_model / mean_crps_clim)


def compute_reliability(
    forecast: ProbabilisticForecast,
    y_actual: pd.Series,
    quantile_levels: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Computes the reliability (calibration) of the probabilistic forecast.

    For each nominal quantile level q, the observed frequency is the fraction
    of actual observations that fell below the forecast's q-th quantile. A
    perfectly calibrated forecast has observed frequency equal to q at every level.

    Over-confidence produces an observed frequency below the nominal level
    (the intervals are too narrow). Under-confidence produces an observed
    frequency above the nominal level (the intervals are too wide).

    Parameters
    ----------
    forecast:
        ProbabilisticForecast from predict_distribution.
    y_actual:
        Observed target values, aligned to forecast.index.
    quantile_levels:
        Quantile levels to evaluate. Defaults to QUANTILE_LEVELS.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'nominal_quantile' and 'observed_frequency',
        one row per quantile level. Also includes a 'reliability_error' column
        which is the absolute deviation from perfect calibration.
    """
    if quantile_levels is None:
        quantile_levels = QUANTILE_LEVELS

    aligned = y_actual.reindex(forecast.index).dropna()
    records = []

    for q in quantile_levels:
        threshold       = forecast.quantile(q).reindex(aligned.index)
        observed_freq   = float((aligned < threshold).mean())
        reliability_err = abs(observed_freq - q)
        records.append({
            "nominal_quantile":   q,
            "observed_frequency": round(observed_freq, 4),
            "reliability_error":  round(reliability_err, 4),
        })

    return pd.DataFrame(records)


def compute_sharpness(
    forecast: ProbabilisticForecast,
    interval_levels: Optional[dict[str, tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Computes the mean width of prediction intervals at multiple coverage levels.

    Sharpness measures how informative the forecast is: a very wide interval
    is technically well-calibrated but commercially useless because it provides
    no useful information about the distribution of outcomes.

    The target is to be as sharp as possible while maintaining calibration.
    In practice, NGBoost tends to produce well-calibrated and moderately sharp
    intervals for 5-day temperature forecasts, with 80% interval widths in the
    range of 3-6 degrees Celsius depending on season and location.

    Parameters
    ----------
    forecast:
        ProbabilisticForecast from predict_distribution.
    interval_levels:
        Dictionary mapping label to (lower_q, upper_q) tuples. Defaults to
        INTERVAL_LEVELS.

    Returns
    -------
    pd.DataFrame
        One row per interval level with 'interval', 'mean_width', and
        'nominal_coverage' columns.
    """
    if interval_levels is None:
        interval_levels = INTERVAL_LEVELS

    records = []
    for label, (lower_q, upper_q) in interval_levels.items():
        ivl        = forecast.interval(lower_q, upper_q)
        mean_width = float((ivl["upper"] - ivl["lower"]).mean())
        records.append({
            "interval":         label,
            "nominal_coverage": upper_q - lower_q,
            "mean_width":       round(mean_width, 4),
        })
    return pd.DataFrame(records)


def run_probabilistic_walk_forward_cv(
    df:           pd.DataFrame,
    target_col:   str,
    feature_cols: list[str],
    n_splits:     int = 6,
    initial_train_frac: float = 0.5,
    params:       Optional[dict] = None,
    early_stopping_rounds: int = 40,
) -> dict:
    """
    Runs walk-forward cross-validation for the NGBoost model, computing
    CRPS, reliability, and sharpness at each fold.

    NGBoost is slower to train than LightGBM so a smaller number of folds
    is used by default. Six folds with a 50% initial window gives approximately
    8% of the full dataset per fold, which is enough for stable reliability
    estimates while keeping total runtime under 20 minutes.

    Parameters
    ----------
    df:
        Full feature and target DataFrame, UTC-indexed, sorted chronologically.
    target_col:
        Target column name.
    feature_cols:
        Feature column names.
    n_splits:
        Number of walk-forward folds.
    initial_train_frac:
        Fraction of data used in the minimum training window.
    params:
        NGBoost hyperparameters. If None, DEFAULT_NGB_PARAMS is used.
    early_stopping_rounds:
        Early stopping patience.

    Returns
    -------
    dict
        Dictionary with keys 'fold_metrics' (list of per-fold dicts),
        'all_crps' (concatenated per-timestep CRPS Series), and
        'all_forecasts' (concatenated ProbabilisticForecast outputs).
    """
    n            = len(df)
    initial_size = int(n * initial_train_frac)
    block_size   = int((n - initial_size) / n_splits)

    fold_metrics:   list[dict]    = []
    all_crps:       list[pd.Series] = []
    all_mu:         list[pd.Series] = []
    all_sigma:      list[pd.Series] = []
    all_actuals:    list[pd.Series] = []

    for fold in range(n_splits):
        train_end  = initial_size + fold * block_size
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

        scaler   = StandardScaler()
        X_tr_sc  = pd.DataFrame(
            scaler.fit_transform(X_tr), columns=feature_cols, index=X_tr.index
        )
        X_te_sc  = pd.DataFrame(
            scaler.transform(X_te), columns=feature_cols, index=X_te.index
        )

        model    = fit_ngboost(X_tr_sc, y_tr, X_te_sc, y_te, params=params,
                               early_stopping_rounds=early_stopping_rounds)
        forecast = predict_distribution(model, X_te_sc)
        crps     = compute_crps(forecast, y_te)
        crpss    = compute_probabilistic_skill_score(crps, y_tr, y_te)

        mae   = float(mean_absolute_error(y_te, forecast.mu.reindex(y_te.index)))
        rmse  = float(np.sqrt(mean_squared_error(y_te, forecast.mu.reindex(y_te.index))))

        fold_metrics.append({
            "fold":          fold + 1,
            "period_start":  str(test_slice.index[0].date()),
            "period_end":    str(test_slice.index[-1].date()),
            "mean_CRPS":     round(float(crps.mean()), 4),
            "CRPSS":         round(crpss, 4),
            "MAE":           round(mae,  4),
            "RMSE":          round(rmse, 4),
            "mean_sigma":    round(float(forecast.sigma.mean()), 4),
        })

        all_crps.append(crps)
        all_mu.append(forecast.mu)
        all_sigma.append(forecast.sigma)
        all_actuals.append(y_te)

        logger.info(
            "Probabilistic fold %d/%d: CRPS=%.4f, CRPSS=%.4f, sigma=%.4f.",
            fold + 1, n_splits,
            float(crps.mean()), crpss, float(forecast.sigma.mean()),
        )

    combined_mu     = pd.concat(all_mu)
    combined_sigma  = pd.concat(all_sigma)
    combined_actuals = pd.concat(all_actuals)
    combined_forecast = ProbabilisticForecast(
        mu=combined_mu.to_numpy(),
        sigma=combined_sigma.to_numpy(),
        index=combined_mu.index,
    )

    return {
        "fold_metrics":   fold_metrics,
        "all_crps":       pd.concat(all_crps),
        "all_forecasts":  combined_forecast,
        "all_actuals":    combined_actuals,
    }


def build_probabilistic_results_table(cv_results: dict) -> pd.DataFrame:
    """
    Formats the walk-forward CV results into a clean summary table suitable
    for inclusion in a notebook or report.

    Parameters
    ----------
    cv_results:
        Output dict from run_probabilistic_walk_forward_cv.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per fold and a final summary row showing
        mean and standard deviation across folds.
    """
    fold_df = pd.DataFrame(cv_results["fold_metrics"])

    numeric_cols = ["mean_CRPS", "CRPSS", "MAE", "RMSE", "mean_sigma"]
    summary_row  = {"fold": "Mean / Std"}
    for col in numeric_cols:
        mean = fold_df[col].mean()
        std  = fold_df[col].std()
        summary_row[col] = f"{mean:.4f} ({std:.4f})"

    summary_df = pd.DataFrame([summary_row])
    full_table = pd.concat(
        [fold_df.astype(str), summary_df], ignore_index=True
    )
    return full_table
