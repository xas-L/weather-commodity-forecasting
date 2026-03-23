"""
Here we compute Heating Degree Days (HDD) and Cooling Degree Days (CDD) from 2m
temperature, along with climatological anomalies, rolling accumulations, and
derived demand-proxy features.

HDD and CDD are the primary bridge between a temperature forecast and a gas or
power demand forecast. Energy traders almost never think in raw temperature units
when sizing a position. They think in HDD anomaly relative to the seasonal norm,
because that quantity maps directly to incremental demand above or below a
baseline.

The standard base temperature for European gas markets is 15.5 degrees Celsius.
This represents the approximate outdoor temperature below which residential and
commercial heating systems switch on. Above this threshold, space heating demand
is assumed to be negligible.

The 18.3 degree Celsius base is also provided as an alternative, as it is used
in some UK and North American market conventions. The degree_days function accepts
a configurable base so callers can select the appropriate convention for their
target market.

Setup required before use:
    pip install pandas numpy
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Base temperatures in Celsius for the two main market conventions.
BASE_TEMP_EUROPEAN: float = 15.5   # Standard for TTF, German EPEX, Dutch APX
BASE_TEMP_UK_NA:    float = 18.3   # Used by some UK and North American desks


def compute_degree_days(
    t2m_celsius: pd.Series,
    base_temp: float = BASE_TEMP_EUROPEAN,
) -> pd.DataFrame:
    """
    Computes hourly or daily HDD and CDD values from a 2m temperature series.

    HDD represents the demand for space heating: it is positive when the
    temperature falls below the base and zero otherwise. CDD represents the
    demand for cooling (air conditioning): it is positive above the base.

    If the input series is hourly, the caller should aggregate to daily sums
    before passing to this function, or use compute_daily_degree_days which
    handles that step automatically. Hourly degree days are valid as features
    in a short-term model but have a different interpretation from daily values.

    Parameters
    ----------
    t2m_celsius:
        Temperature series in degrees Celsius.
    base_temp:
        Base temperature in degrees Celsius. Defaults to the European gas
        market convention of 15.5 degrees.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'HDD' and 'CDD', same index as the input.
    """
    hdd = (base_temp - t2m_celsius).clip(lower=0.0)
    cdd = (t2m_celsius - base_temp).clip(lower=0.0)

    return pd.DataFrame({"HDD": hdd, "CDD": cdd}, index=t2m_celsius.index)


def compute_daily_degree_days(
    t2m_hourly: pd.Series,
    base_temp: float = BASE_TEMP_EUROPEAN,
) -> pd.DataFrame:
    """
    Aggregates an hourly temperature series to daily mean before computing
    HDD and CDD. The daily mean is the conventional basis for degree-day
    products traded on the CME and used in energy company load forecasts.

    Parameters
    ----------
    t2m_hourly:
        Hourly temperature series in degrees Celsius, UTC-indexed.
    base_temp:
        Base temperature in degrees Celsius.

    Returns
    -------
    pd.DataFrame
        Daily HDD and CDD, indexed at midnight UTC for each calendar day.
    """
    daily_mean = t2m_hourly.resample("1D").mean()
    return compute_degree_days(daily_mean, base_temp=base_temp)


def fit_hdd_climatology(degree_days: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Computes the day-of-year mean HDD and CDD from a degree-day DataFrame.

    This should be called on the training period only and the result passed to
    add_climatological_anomaly for both the training and test windows. Fitting
    the climatology on training data alone prevents test-period observations
    from contaminating the anomaly baseline.

    Typical usage:

        train_dd = build_degree_day_features(t2m_train)
        clim     = fit_hdd_climatology(train_dd)

        train_feat = build_degree_day_features(t2m_train, climatology=clim)
        test_feat  = build_degree_day_features(t2m_test,  climatology=clim)

    Parameters
    ----------
    degree_days:
        DataFrame containing 'HDD' and 'CDD' columns covering the training
        period only, as returned by compute_degree_days or
        compute_daily_degree_days.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary with keys 'HDD' and 'CDD', each mapping to a Series whose
        index is the integer day-of-year (1–366) and whose values are the
        training-period mean for that day.
    """
    df = degree_days.copy()
    df["doy"] = df.index.dayofyear

    clim: dict[str, pd.Series] = {}
    for col in ["HDD", "CDD"]:
        if col not in df.columns:
            logger.warning(
                "fit_hdd_climatology: column '%s' not found, skipping.", col
            )
            continue
        clim[col] = df.groupby("doy")[col].mean()

    return clim


def add_climatological_anomaly(
    degree_days: pd.DataFrame,
    climatology: Optional[dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Computes the deviation of each day's HDD and CDD from its day-of-year
    climatological mean.

    The anomaly is what a gas trader actually cares about. A day with HDD of
    8 in mid-January is unremarkable. A day with HDD of 8 in late March is
    a significant cold anomaly and implies above-seasonal gas burn.

    When climatology is supplied it must have been fitted on the training period
    only, via fit_hdd_climatology. This prevents test-period observations from
    influencing the anomaly baseline. If climatology is None, the mean is
    estimated from the input DataFrame itself, which is only valid when that
    DataFrame is the training set.

    Parameters
    ----------
    degree_days:
        DataFrame containing 'HDD' and 'CDD' columns, as returned by
        compute_degree_days or compute_daily_degree_days.
    climatology:
        Pre-fitted day-of-year climatology as returned by fit_hdd_climatology.
        Should always be supplied in evaluation workflows. When None, the mean
        is estimated from the input DataFrame — acceptable for training data
        only, and will introduce forward-looking bias if the input spans the
        test period.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns 'HDD_anom' and 'CDD_anom'
        representing the signed deviation from the climatological mean.
    """
    df = degree_days.copy()
    df["doy"] = df.index.dayofyear

    for col, anom_col in [("HDD", "HDD_anom"), ("CDD", "CDD_anom")]:
        if col not in df.columns:
            logger.warning("Column '%s' not found, skipping anomaly computation.", col)
            continue

        if climatology is not None and col in climatology:
            clim_mean = df["doy"].map(climatology[col])
        else:
            if climatology is None:
                logger.warning(
                    "add_climatological_anomaly: no climatology supplied for '%s'. "
                    "Mean will be estimated from the input DataFrame. This introduces "
                    "forward-looking bias if the input spans both training and test "
                    "periods. Supply climatology from fit_hdd_climatology to avoid this.",
                    col,
                )
            clim_mean = df.groupby("doy")[col].transform("mean")

        df[anom_col] = df[col] - clim_mean

    df = df.drop(columns=["doy"])
    return df


def add_rolling_accumulations(
    degree_days: pd.DataFrame,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Adds rolling sum accumulations of HDD and CDD over multiple windows.

    Rolling sums represent the cumulative thermal burden over recent periods.
    A 7-day HDD accumulation is the metric most commonly cited in European
    gas supply and demand reports, as it aligns with weekly storage withdrawal
    cycles and weekly forward curve settlement periods.

    The 30-day accumulation provides a medium-term view of whether the season
    is running warmer or colder than expected, which is more relevant for
    structured positions than for day-ahead trading.

    Parameters
    ----------
    degree_days:
        DataFrame with 'HDD' and 'CDD' columns.
    windows:
        Rolling window lengths in days. Defaults to [3, 7, 14, 30].

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional rolling sum columns for each window,
        named 'HDD_{n}d' and 'CDD_{n}d'.
    """
    if windows is None:
        windows = [3, 7, 14, 30]

    df = degree_days.copy()

    for col in ["HDD", "CDD"]:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_{w}d"] = df[col].rolling(w, min_periods=max(1, w // 2)).sum()

    return df


def add_anomaly_accumulations(
    degree_days: pd.DataFrame,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Adds rolling sum accumulations of the HDD and CDD anomaly columns.

    Where rolling raw accumulations capture the absolute thermal burden,
    rolling anomaly accumulations capture how persistently above or below
    seasonal norms conditions have been. A sustained positive HDD anomaly
    accumulation is the signal that most reliably correlates with TTF
    front-month price premium.

    This function expects 'HDD_anom' and 'CDD_anom' to already be present,
    so it should be called after add_climatological_anomaly.

    Parameters
    ----------
    degree_days:
        DataFrame with 'HDD_anom' and 'CDD_anom' columns.
    windows:
        Rolling window lengths in days. Defaults to [7, 14, 30].

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional anomaly accumulation columns.
    """
    if windows is None:
        windows = [7, 14, 30]

    df = degree_days.copy()

    for col in ["HDD_anom", "CDD_anom"]:
        if col not in df.columns:
            logger.warning("Column '%s' not found, skipping anomaly accumulation.", col)
            continue
        for w in windows:
            df[f"{col}_{w}d"] = df[col].rolling(w, min_periods=max(1, w // 2)).sum()

    return df


def add_seasonal_position(degree_days: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'seasonal_position' column encoding where each observation sits
    within the gas year, expressed as a fraction from 0 to 1.

    The European gas year runs from 1 October to 30 September. Seasonal position
    allows models to distinguish between, say, a cold anomaly in early winter
    (high market impact, storage still relatively full) and the same anomaly in
    late winter (lower impact, storage draw already baked into the forward curve).

    A sine transform of seasonal position is also added to give the model a
    smooth cyclic representation of the gas year phase, which helps gradient
    boosting models capture the asymmetric seasonality of heating demand.

    Parameters
    ----------
    degree_days:
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'gas_year_position', 'gas_year_sin', and
        'gas_year_cos' columns.
    """
    df = degree_days.copy()

    gas_year_start_doy = 274   # 1 October is day 274 in a non-leap year

    doy = df.index.dayofyear.to_numpy(dtype=float)
    shifted = (doy - gas_year_start_doy) % 365.25
    position = shifted / 365.25

    df["gas_year_position"] = position
    df["gas_year_sin"]      = np.sin(2.0 * np.pi * position)
    df["gas_year_cos"]      = np.cos(2.0 * np.pi * position)

    return df


def compute_weekly_hdd_forecast_error(
    hdd_actual: pd.Series,
    hdd_forecast: pd.Series,
) -> pd.DataFrame:
    """
    Computes the signed and absolute forecast error for weekly HDD accumulations.

    This is the primary metric used in the commodity linkage notebook to
    quantify how much demand surprise is attributable to weather forecast error.
    A positive error means actual conditions were colder than forecast, implying
    higher-than-expected gas demand.

    Parameters
    ----------
    hdd_actual:
        Realised daily HDD series, UTC-indexed.
    hdd_forecast:
        Forecast daily HDD series from the model, UTC-indexed.

    Returns
    -------
    pd.DataFrame
        Weekly-indexed DataFrame with columns 'hdd_actual_7d', 'hdd_forecast_7d',
        'hdd_error_7d', and 'hdd_abs_error_7d'.
    """
    actual_7d   = hdd_actual.resample("1W").sum()
    forecast_7d = hdd_forecast.resample("1W").sum()

    aligned = pd.DataFrame({
        "hdd_actual_7d":   actual_7d,
        "hdd_forecast_7d": forecast_7d,
    }).dropna()

    aligned["hdd_error_7d"]     = aligned["hdd_actual_7d"] - aligned["hdd_forecast_7d"]
    aligned["hdd_abs_error_7d"] = aligned["hdd_error_7d"].abs()

    return aligned


def build_degree_day_features(
    t2m_hourly: pd.Series,
    base_temp: float = BASE_TEMP_EUROPEAN,
    climatology: Optional[dict[str, pd.Series]] = None,
    rolling_windows: Optional[list[int]] = None,
    anomaly_windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Orchestrates the full degree-day feature construction pipeline in a single
    call. Aggregates hourly temperature to daily, computes HDD and CDD,
    adds climatological anomalies, rolling accumulations, and seasonal position.

    This is the primary entry point for the feature engineering notebook and
    for the model training pipeline.

    The climatology parameter should always be supplied in evaluation workflows
    so that the anomaly baseline is fixed on training data. Fit it once via
    fit_hdd_climatology on the training period, then pass the result here for
    both training and test feature construction.

    Parameters
    ----------
    t2m_hourly:
        Hourly 2m temperature in degrees Celsius, UTC-indexed.
    base_temp:
        Base temperature in degrees Celsius for degree-day calculation.
    climatology:
        Pre-fitted day-of-year climatology as returned by fit_hdd_climatology.
        If None, the mean is estimated from the input data. Acceptable for
        training data only; introduces forward-looking bias for test data.
    rolling_windows:
        Window lengths for raw HDD/CDD rolling sums. Passed to
        add_rolling_accumulations.
    anomaly_windows:
        Window lengths for anomaly rolling sums. Passed to
        add_anomaly_accumulations.

    Returns
    -------
    pd.DataFrame
        Daily-indexed DataFrame with the full degree-day feature set.
    """
    df = compute_daily_degree_days(t2m_hourly, base_temp=base_temp)
    df = add_climatological_anomaly(df, climatology=climatology)
    df = add_rolling_accumulations(df, windows=rolling_windows)
    df = add_anomaly_accumulations(df, windows=anomaly_windows)
    df = add_seasonal_position(df)

    logger.info(
        "Built degree-day feature set: %d daily rows, %d columns.",
        len(df), len(df.columns),
    )
    return df