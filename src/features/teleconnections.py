"""
Construction of subseasonal regime features from the CPC teleconnection indices
(NAO, AO, PNA, ENSO) for use in the week-2 to week-4 temperature regime
classifier.

Teleconnection indices capture large-scale, slowly evolving patterns in the
atmospheric circulation that carry predictability on timescales beyond the
deterministic range of NWP (roughly day 10-15). The NAO is the dominant source
of European subseasonal predictability, but the AO, PNA and ENSO provide
complementary signals that matter in different seasons and years.

The features built here are designed to answer a specific question that a
commodity gas trader would ask: given today's atmospheric circulation state,
what is the probability that NW Europe will be in a cold, neutral, or warm
temperature regime in two to four weeks' time?

The feature philosophy is:
    Current state     where is the index today?
    Recent tendency   which direction has it been moving?
    Persistence       how long has the current regime been in place?
    Phase coupling    are multiple indices simultaneously reinforcing?

All functions accept UTC-indexed pandas Series and return UTC-indexed DataFrames.

Setup required before use:
    pip install pandas numpy scikit-learn
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Thresholds for regime classification in units of standard deviations.
# These are conventional values used in the atmospheric science literature.
# Cold and warm events are defined symmetrically to produce balanced classes.
REGIME_COLD_THRESHOLD:     float = -0.8
REGIME_WARM_THRESHOLD:     float =  0.8
BLOCKING_THRESHOLD:        float = -1.5
STRONG_WESTERLY_THRESHOLD: float =  1.5

# Lead times in days for which lagged index values are constructed.
# Week-2 and week-4 leads correspond to the target forecast horizons.
SUBSEASONAL_LAGS: list[int] = [7, 10, 14, 21, 28]

# ENSO phase thresholds in degrees Celsius anomaly for the Nino 3.4 region.
ENSO_WARM_THRESHOLD: float =  0.5   # El Nino
ENSO_COLD_THRESHOLD: float = -0.5   # La Nina


def classify_nao_regime(
    nao: pd.Series,
    cold_threshold: float = REGIME_COLD_THRESHOLD,
    warm_threshold: float = REGIME_WARM_THRESHOLD,
) -> pd.Series:
    """
    Assigns a ternary regime label to each observation based on the NAO index
    value. Labels are 'cold' (negative NAO, blocking), 'neutral', and 'warm'
    (positive NAO, westerly flow).

    The labels map to European temperature expectation as follows. A cold regime
    indicates suppressed westerlies and elevated risk of cold air outbreaks over
    Germany and the Netherlands, which drives HDD above seasonal norms and
    increases TTF gas demand. A warm regime indicates enhanced westerlies and
    above-average temperatures, suppressing heating demand.

    Parameters
    ----------
    nao:
        Daily NAO index Series, UTC-indexed.
    cold_threshold:
        NAO value below which the regime is classified as cold.
    warm_threshold:
        NAO value above which the regime is classified as warm.

    Returns
    -------
    pd.Series
        Categorical Series with values 'cold', 'neutral', 'warm',
        same index as the input.
    """
    conditions = [
        nao < cold_threshold,
        nao > warm_threshold,
    ]
    choices = ["cold", "warm"]
    regime = pd.Series(
        np.select(conditions, choices, default="neutral"),
        index=nao.index,
        name="nao_regime",
    )
    return regime


def classify_enso_phase(nino34_anom: pd.Series) -> pd.Series:
    """
    Assigns a ternary ENSO phase label based on the Nino 3.4 SST anomaly.
    Labels are 'el_nino', 'neutral', and 'la_nina'.

    ENSO phase influences European winter temperatures on seasonal timescales
    through Rossby wave teleconnections across the Pacific and North Atlantic.
    La Nina winters tend to be associated with more frequent negative NAO
    episodes and elevated cold risk for NW Europe, particularly in
    December and January.

    Parameters
    ----------
    nino34_anom:
        Monthly Nino 3.4 SST anomaly Series in degrees Celsius, UTC-indexed.

    Returns
    -------
    pd.Series
        Categorical ENSO phase Series, same index as the input.
    """
    conditions = [
        nino34_anom >= ENSO_WARM_THRESHOLD,
        nino34_anom <= ENSO_COLD_THRESHOLD,
    ]
    choices = ["el_nino", "la_nina"]
    phase = pd.Series(
        np.select(conditions, choices, default="neutral"),
        index=nino34_anom.index,
        name="enso_phase",
    )
    return phase


def compute_index_lags(
    index_series: pd.Series,
    lags: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Creates lagged copies of a teleconnection index for a set of lead times.

    Lagged values of the NAO are among the most important features for subseasonal
    temperature prediction. An NAO value at lag-14 represents the state of the
    index two weeks before the forecast target date, which is within the predictable
    range for large-scale blocking patterns.

    Parameters
    ----------
    index_series:
        Daily index Series, UTC-indexed.
    lags:
        List of lag lengths in days. Defaults to SUBSEASONAL_LAGS.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per lag, named '{series.name}_lag{n}d'.
    """
    if lags is None:
        lags = SUBSEASONAL_LAGS

    name = index_series.name or "index"
    frames = {
        f"{name}_lag{lag}d": index_series.shift(lag)
        for lag in lags
    }
    return pd.DataFrame(frames, index=index_series.index)


def compute_index_tendency(
    index_series: pd.Series,
    short_window: int = 5,
    long_window: int  = 20,
) -> pd.DataFrame:
    """
    Computes short-term and medium-term rolling means of a teleconnection index
    and their difference as a tendency feature.

    The tendency represents the direction in which the index is moving. An index
    that was neutral but is now trending strongly negative has greater near-term
    forecast significance than a static negative value, because it signals that
    a blocking development is underway rather than decaying.

    Parameters
    ----------
    index_series:
        Daily index Series, UTC-indexed.
    short_window:
        Rolling window length in days for the short-term mean.
    long_window:
        Rolling window length in days for the medium-term mean.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling mean columns and a tendency column.
    """
    name = index_series.name or "index"

    short_mean = (
        index_series.rolling(short_window, min_periods=max(1, short_window // 2)).mean()
    )
    long_mean = (
        index_series.rolling(long_window, min_periods=max(1, long_window // 2)).mean()
    )
    tendency = short_mean - long_mean

    return pd.DataFrame({
        f"{name}_{short_window}d_mean": short_mean,
        f"{name}_{long_window}d_mean":  long_mean,
        f"{name}_tendency":             tendency,
    }, index=index_series.index)


def compute_persistence_features(
    index_series: pd.Series,
    cold_threshold: float = REGIME_COLD_THRESHOLD,
    warm_threshold: float = REGIME_WARM_THRESHOLD,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Quantifies how long the current NAO or AO regime has persisted. Persistence
    features capture the memory of the atmospheric circulation, which is the
    physical mechanism that gives subseasonal forecasts their skill beyond the
    NWP deterministic limit.

    A blocking event that has been in place for 12 days is more likely to persist
    for another week than one that started yesterday, because the planetary-wave
    structure that supports it requires time to break down.

    Parameters
    ----------
    index_series:
        Daily index Series, UTC-indexed.
    cold_threshold:
        Threshold below which the index is considered in a cold or blocking state.
    warm_threshold:
        Threshold above which the index is considered in a warm state.
    windows:
        Rolling window lengths in days for regime frequency features.
        Defaults to [5, 10, 20].

    Returns
    -------
    pd.DataFrame
        Persistence features including current-phase duration and
        rolling phase frequency over multiple windows.
    """
    if windows is None:
        windows = [5, 10, 20]

    name = index_series.name or "index"

    in_cold = (index_series < cold_threshold).astype(int)
    in_warm = (index_series > warm_threshold).astype(int)

    def consecutive_days(binary: pd.Series) -> pd.Series:
        """Counts consecutive days the binary flag has been active."""
        not_active = (binary == 0).cumsum()
        return binary.groupby(not_active).cumcount() * binary

    frames = {
        f"{name}_in_cold_phase":    in_cold,
        f"{name}_in_warm_phase":    in_warm,
        f"{name}_cold_consec_days": consecutive_days(in_cold),
        f"{name}_warm_consec_days": consecutive_days(in_warm),
    }

    for w in windows:
        min_periods = max(1, w // 2)
        frames[f"{name}_cold_freq_{w}d"] = (
            in_cold.rolling(w, min_periods=min_periods).mean()
        )
        frames[f"{name}_warm_freq_{w}d"] = (
            in_warm.rolling(w, min_periods=min_periods).mean()
        )

    return pd.DataFrame(frames, index=index_series.index)


def compute_phase_coupling(
    nao: pd.Series,
    ao: pd.Series,
    pna: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Constructs features that capture when multiple teleconnection indices are
    simultaneously in the same phase, producing a reinforced circulation signal.

    The combined cold regime (both NAO and AO simultaneously negative) is the
    most reliable precursor to cold air outbreaks over NW Europe. When the PNA
    index is simultaneously negative, this further reinforces the cold signal
    via Rossby wave propagation from the Pacific.

    These coupling features are non-linear combinations that a gradient boosting
    model can learn from but would be difficult to construct implicitly from the
    raw index values alone.

    Parameters
    ----------
    nao:
        Daily NAO index Series, UTC-indexed.
    ao:
        Daily AO index Series, UTC-indexed.
    pna:
        Daily PNA index Series, UTC-indexed. Optional.

    Returns
    -------
    pd.DataFrame
        Phase coupling feature DataFrame, UTC-indexed.
    """
    aligned = pd.DataFrame({"nao": nao, "ao": ao}).dropna()

    if pna is not None:
        aligned = aligned.join(pna.rename("pna"), how="left")

    frames = {}

    # Linear product captures simultaneous exceedance better than each index alone
    frames["nao_ao_product"]   = aligned["nao"] * aligned["ao"]
    frames["nao_ao_both_cold"] = (
        (aligned["nao"] < REGIME_COLD_THRESHOLD) & (aligned["ao"] < REGIME_COLD_THRESHOLD)
    ).astype(int)
    frames["nao_ao_both_warm"] = (
        (aligned["nao"] > REGIME_WARM_THRESHOLD) & (aligned["ao"] > REGIME_WARM_THRESHOLD)
    ).astype(int)
    frames["nao_ao_opposing"]  = (
        (aligned["nao"] * aligned["ao"]) < 0
    ).astype(int)

    # Magnitude of combined cold signal: sum when both negative, zero otherwise
    both_cold_magnitude = aligned[["nao", "ao"]].where(
        (aligned["nao"] < 0) & (aligned["ao"] < 0), other=0.0
    ).sum(axis=1)
    frames["combined_cold_magnitude"] = both_cold_magnitude

    if "pna" in aligned.columns:
        frames["triple_cold"] = (
            (aligned["nao"] < REGIME_COLD_THRESHOLD)
            & (aligned["ao"] < REGIME_COLD_THRESHOLD)
            & (aligned["pna"] < REGIME_COLD_THRESHOLD)
        ).astype(int)

    return pd.DataFrame(frames, index=aligned.index)


def fit_weekly_temperature_climatology(
    t2m_weekly: pd.Series,
) -> tuple[pd.Series, float]:
    """
    Computes the ISO week-of-year mean and the rolling standard deviation of a
    weekly temperature series from the training period.

    This should be called on training data only and the results passed to
    build_subseasonal_target for both training and test target construction.
    Fitting on training data alone prevents the standardisation from being
    contaminated by test-period observations.

    Typical usage:

        clim_mean, clim_std = fit_weekly_temperature_climatology(t2m_weekly_train)
        target_train = build_subseasonal_target(t2m_weekly_train,
                                               train_clim_mean=clim_mean,
                                               train_clim_std=clim_std)
        target_test  = build_subseasonal_target(t2m_weekly_test,
                                               train_clim_mean=clim_mean,
                                               train_clim_std=clim_std)

    Parameters
    ----------
    t2m_weekly:
        Weekly mean 2m temperature in degrees Celsius, UTC-indexed, covering
        the training period only.

    Returns
    -------
    tuple[pd.Series, float]
        A tuple of:
            clim_mean: Series mapping ISO week number (1–53) to training-period
                       mean temperature for that week.
            clim_std:  Scalar standard deviation of the full training series,
                       used for normalising weekly anomalies.
    """
    iso_week = t2m_weekly.index.isocalendar().week.astype(int)
    clim_mean = t2m_weekly.groupby(iso_week).mean()
    clim_std  = float(t2m_weekly.std())

    if clim_std == 0.0:
        raise ValueError(
            "Training temperature series has zero standard deviation. "
            "Cannot compute normalised anomalies."
        )

    return clim_mean, clim_std


def build_subseasonal_target(
    t2m_weekly: pd.Series,
    lead_weeks: int = 2,
    cold_threshold_sd: float = REGIME_COLD_THRESHOLD,
    warm_threshold_sd: float = REGIME_WARM_THRESHOLD,
    train_clim_mean: Optional[pd.Series] = None,
    train_clim_std: Optional[float] = None,
) -> pd.Series:
    """
    Constructs the target variable for the subseasonal regime classifier.

    The target is a ternary label representing the observed temperature regime
    in a given week, shifted forward by the lead time so that features at time t
    are paired with the regime at time t + lead_weeks.

    When train_clim_mean and train_clim_std are supplied they must have been
    fitted on the training period only, via fit_weekly_temperature_climatology.
    This ensures that the anomaly standardisation applied to test-period weeks
    uses a baseline that does not incorporate any test observations. Supplying
    these parameters is strongly recommended for all evaluation workflows.

    If either parameter is None, the mean and standard deviation are estimated
    from the input series itself. This is acceptable when the input is the
    training set, but will introduce forward-looking bias if the input spans
    the test period.

    Parameters
    ----------
    t2m_weekly:
        Weekly mean 2m temperature in degrees Celsius, UTC-indexed.
    lead_weeks:
        Number of weeks ahead to shift the target. Use 2 for week-2 forecasting
        and 4 for week-4 forecasting.
    cold_threshold_sd:
        Standard deviation threshold for cold regime classification.
    warm_threshold_sd:
        Standard deviation threshold for warm regime classification.
    train_clim_mean:
        ISO week-of-year mean temperatures fitted on the training period only,
        as returned by fit_weekly_temperature_climatology. If None, the mean is
        computed from the input series.
    train_clim_std:
        Scalar standard deviation fitted on the training period only, as returned
        by fit_weekly_temperature_climatology. If None, the std is computed from
        the input series.

    Returns
    -------
    pd.Series
        Ternary regime target ('cold', 'neutral', 'warm'), shifted so that
        the label at each index position corresponds to the regime that will
        occur lead_weeks later.
    """
    iso_week = t2m_weekly.index.isocalendar().week.astype(int)

    if train_clim_mean is not None:
        # Map each observation's ISO week to its training-period mean
        clim_mean = iso_week.map(train_clim_mean).values
        clim_mean = pd.Series(clim_mean, index=t2m_weekly.index)
    else:
        logger.warning(
            "build_subseasonal_target: train_clim_mean not supplied. Weekly mean "
            "will be estimated from the input series. This introduces forward-looking "
            "bias if the input spans both training and test periods. Supply "
            "train_clim_mean from fit_weekly_temperature_climatology to avoid this."
        )
        clim_mean = t2m_weekly.groupby(iso_week).transform("mean")

    if train_clim_std is not None:
        clim_std = train_clim_std
    else:
        logger.warning(
            "build_subseasonal_target: train_clim_std not supplied. Standard "
            "deviation will be estimated from the input series. This introduces "
            "forward-looking bias if the input spans both training and test periods."
        )
        clim_std = float(t2m_weekly.rolling(52 * 3, min_periods=52).std().median())

    if clim_std == 0.0:
        raise ValueError(
            "Climatological standard deviation is zero. Cannot standardise anomalies."
        )

    anom_sd = (t2m_weekly - clim_mean) / clim_std

    regime = classify_nao_regime(
        anom_sd,
        cold_threshold=cold_threshold_sd,
        warm_threshold=warm_threshold_sd,
    )
    regime.name = f"regime_target_lead{lead_weeks}w"

    # Shift target backward by lead_weeks so it aligns with features at prediction time
    target = regime.shift(-lead_weeks)
    return target


def build_teleconnection_features(
    nao: pd.Series,
    ao: pd.Series,
    pna: Optional[pd.Series] = None,
    nino34_anom: Optional[pd.Series] = None,
    lags: Optional[list[int]] = None,
    tendency_windows: Optional[tuple[int, int]] = None,
    persistence_windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Orchestrates the full teleconnection feature construction pipeline in a
    single call. Computes regime labels, lagged values, tendencies, persistence
    metrics, and phase coupling features for all provided indices.

    This is the primary entry point for the feature engineering notebook and
    for the subseasonal regime classifier training pipeline.

    Parameters
    ----------
    nao:
        Daily NAO index Series, UTC-indexed.
    ao:
        Daily AO index Series, UTC-indexed.
    pna:
        Daily PNA index Series, UTC-indexed. Optional.
    nino34_anom:
        Monthly Nino 3.4 SST anomaly, forward-filled to daily, UTC-indexed. Optional.
    lags:
        Lag lengths in days for lagged index features.
    tendency_windows:
        Tuple of (short_window, long_window) in days for tendency computation.
        Defaults to (5, 20).
    persistence_windows:
        Window lengths in days for persistence frequency features.

    Returns
    -------
    pd.DataFrame
        Wide daily DataFrame with all teleconnection features, UTC-indexed.
        Missing values at the start (from lags and rolling windows) are left
        as NaN for the modelling pipeline to handle via its own imputation strategy.
    """
    if tendency_windows is None:
        tendency_windows = (5, 20)

    frames = []

    # Raw index values
    frames.append(nao.rename("nao"))
    frames.append(ao.rename("ao"))
    if pna is not None:
        frames.append(pna.rename("pna"))
    if nino34_anom is not None:
        frames.append(nino34_anom.rename("nino34_anom"))

    # Regime classification
    frames.append(
        classify_nao_regime(nao).map({"cold": -1, "neutral": 0, "warm": 1})
                                 .rename("nao_regime_encoded")
    )

    # ENSO phase encoding
    if nino34_anom is not None:
        enso_phase = classify_enso_phase(nino34_anom)
        frames.append(
            enso_phase.map({"la_nina": -1, "neutral": 0, "el_nino": 1})
                      .rename("enso_phase_encoded")
        )

    # Lagged values
    frames.append(compute_index_lags(nao, lags=lags))
    frames.append(compute_index_lags(ao,  lags=lags))
    if pna is not None:
        frames.append(compute_index_lags(pna, lags=lags))

    # Tendencies
    short_w, long_w = tendency_windows
    frames.append(compute_index_tendency(nao, short_w, long_w))
    frames.append(compute_index_tendency(ao,  short_w, long_w))

    # Persistence
    frames.append(compute_persistence_features(nao, windows=persistence_windows))
    frames.append(compute_persistence_features(ao,  windows=persistence_windows))

    # Phase coupling
    frames.append(compute_phase_coupling(nao, ao, pna=pna))

    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index, utc=True)

    logger.info(
        "Built teleconnection feature set: %d rows, %d columns.",
        len(combined), len(combined.columns),
    )
    return combined