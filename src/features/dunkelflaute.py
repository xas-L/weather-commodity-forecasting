"""
Here we detect Dunkelflaute events and construct related features for the short-term
model and the commodity linkage notebook.

Dunkelflaute (German: dark doldrums) refers to prolonged periods during which
both wind and solar generation are simultaneously suppressed below thresholds
that require the grid to rely heavily on dispatchable generation, primarily
natural gas in the German and Dutch markets. These events are the single largest
weather-driven source of gas demand spikes that are NOT captured by HDD alone,
because they represent demand from gas-fired power plants rather than from
residential and commercial heating.

The market impact: when a Dunkelflaute event of 48+ hours is forecast 2-3 days
ahead, German day-ahead power prices typically rise sharply and TTF intraday
prices follow. A commodity trader with early warning of these events has an
informational edge in both power and gas markets.

This module provides:
    Capacity factor computation from raw MW generation and installed capacity.
    Binary and continuous Dunkelflaute detection with configurable thresholds.
    Event cataloguing with start time, duration, and severity metrics.
    Forward-looking probability features for use as model inputs.
    Rolling renewable deficit features that quantify the current and recent
    state of the generation shortfall.

All functions operate on UTC-indexed hourly pandas Series or DataFrames.

Setup required before use:
    pip install pandas numpy
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default capacity factor thresholds for Dunkelflaute classification.
# These values are derived from analysis of German and Dutch grid data and
# represent the level below which gas-fired generation must increase
# substantially to maintain grid balance.
CF_WIND_THRESHOLD:   float = 0.10   # 10% of installed wind capacity
CF_SOLAR_THRESHOLD:  float = 0.05   # 5% of installed solar capacity (daytime only)

# Minimum event duration in hours. Events shorter than this are not classified
# as Dunkelflaute because they do not require significant gas dispatch changes.
MIN_EVENT_HOURS: int = 24

# Rolling windows in hours for generation deficit features.
DEFICIT_WINDOWS: list[int] = [6, 12, 24, 48, 72]


@dataclass
class DunkelflauteEvent:
    """
    Represents a single identified Dunkelflaute event with its key metrics.

    Attributes
    ----------
    start:
        UTC timestamp of the first qualifying hour.
    end:
        UTC timestamp of the last qualifying hour.
    duration_hours:
        Total duration in hours.
    mean_wind_cf:
        Mean wind capacity factor during the event.
    mean_solar_cf:
        Mean solar capacity factor during the event (NaN if solar is unavailable).
    max_deficit:
        Maximum combined renewable deficit (1 minus sum of CFs) during the event.
    """
    start:           pd.Timestamp
    end:             pd.Timestamp
    duration_hours:  int
    mean_wind_cf:    float
    mean_solar_cf:   float
    max_deficit:     float


def compute_wind_capacity_factor(
    wind_generation_mw: pd.Series,
    installed_capacity_mw: float,
) -> pd.Series:
    """
    Converts hourly wind generation in MW to a dimensionless capacity factor
    by dividing by the installed capacity. Values are clipped to [0, 1] to
    remove small over-generation artefacts that appear in TSO reported data.

    Using capacity factors rather than raw MW removes the upward trend from
    annual capacity additions, making thresholds stable across years. A 10%
    capacity factor threshold for a country with 50 GW of installed capacity
    means 5 GW of generation, whereas the same percentage means 3 GW for a
    country with 30 GW, which is the appropriate scaling.

    Parameters
    ----------
    wind_generation_mw:
        Hourly wind generation in MW, UTC-indexed.
    installed_capacity_mw:
        Installed wind capacity in MW for the relevant period.

    Returns
    -------
    pd.Series
        Wind capacity factor bounded between 0 and 1.

    Raises
    ------
    ValueError
        If installed_capacity_mw is not strictly positive.
    """
    if installed_capacity_mw <= 0:
        raise ValueError(
            f"Installed capacity must be positive, got {installed_capacity_mw}."
        )

    cf = (wind_generation_mw / installed_capacity_mw).clip(lower=0.0, upper=1.0)
    cf.name = "wind_cf"
    return cf


def compute_solar_capacity_factor(
    solar_generation_mw: pd.Series,
    installed_capacity_mw: float,
) -> pd.Series:
    """
    Converts hourly solar PV generation in MW to a capacity factor.

    Solar capacity factors are near zero at night, which would produce
    false positives in Dunkelflaute detection during nocturnal hours. The
    binary classification functions in this module handle this by only applying
    the solar threshold during daylight hours, but this function returns the
    raw capacity factor for use in rolling deficit features and other contexts.

    Parameters
    ----------
    solar_generation_mw:
        Hourly solar PV generation in MW, UTC-indexed.
    installed_capacity_mw:
        Installed solar PV capacity in MW for the relevant period.

    Returns
    -------
    pd.Series
        Solar capacity factor bounded between 0 and 1.
    """
    if installed_capacity_mw <= 0:
        raise ValueError(
            f"Installed capacity must be positive, got {installed_capacity_mw}."
        )

    cf = (solar_generation_mw / installed_capacity_mw).clip(lower=0.0, upper=1.0)
    cf.name = "solar_cf"
    return cf


def classify_dunkelflaute_hours(
    wind_cf: pd.Series,
    solar_cf: Optional[pd.Series] = None,
    wind_threshold: float = CF_WIND_THRESHOLD,
    solar_threshold: float = CF_SOLAR_THRESHOLD,
    solar_daylight_only: bool = True,
) -> pd.Series:
    """
    Classifies each hour as a Dunkelflaute candidate based on whether the
    wind and solar capacity factors are below their respective thresholds.

    If solar data is not available, classification is based on wind alone.
    This is acceptable for winter months when solar generation is low across
    all hours, but will produce less accurate classification in summer.

    The solar threshold is applied conditionally during daylight hours only
    when solar_daylight_only is True. Nighttime solar capacity factors are
    structurally zero and should not trigger false positives.

    Parameters
    ----------
    wind_cf:
        Hourly wind capacity factor Series, UTC-indexed.
    solar_cf:
        Hourly solar capacity factor Series, UTC-indexed. Optional.
    wind_threshold:
        Capacity factor threshold below which wind is classified as low.
    solar_threshold:
        Capacity factor threshold below which solar is classified as low.
    solar_daylight_only:
        If True, the solar threshold is only applied between 06:00 and 20:00
        local time (approximated as UTC+1 for Central Europe).

    Returns
    -------
    pd.Series
        Boolean Series; True for each hour qualifying as a Dunkelflaute hour.
    """
    low_wind = wind_cf < wind_threshold

    if solar_cf is None:
        result = low_wind
        result.name = "is_dunkelflaute_hour"
        return result

    low_solar = solar_cf < solar_threshold

    if solar_daylight_only:
        # Approximate daylight window for Central Europe as UTC+1, 06:00-20:00
        utc_plus_1_hour = wind_cf.index.hour + 1
        is_daylight = (utc_plus_1_hour >= 6) & (utc_plus_1_hour <= 20)
        # Solar constraint only applies during daylight; nighttime is always
        # considered low solar but should not trigger the classification
        solar_constraint = (~is_daylight) | low_solar
    else:
        solar_constraint = low_solar

    result = low_wind & solar_constraint
    result.name = "is_dunkelflaute_hour"
    return result


def apply_minimum_duration_filter(
    dunkelflaute_hourly: pd.Series,
    min_hours: int = MIN_EVENT_HOURS,
) -> pd.Series:
    """
    Filters the binary Dunkelflaute classification to remove events shorter
    than the minimum duration threshold. Short periods below the capacity
    factor thresholds are common and do not require meaningful gas dispatch
    changes, so they should not be included in the event catalogue or used
    to compute probability features.

    Parameters
    ----------
    dunkelflaute_hourly:
        Binary hourly classification Series from classify_dunkelflaute_hours.
    min_hours:
        Minimum consecutive hours required to count as a Dunkelflaute event.

    Returns
    -------
    pd.Series
        Filtered binary Series with short events removed (set to False).
    """
    result = dunkelflaute_hourly.copy()

    # Group consecutive True values and find run lengths
    not_active    = (~dunkelflaute_hourly).cumsum()
    group_lengths = dunkelflaute_hourly.groupby(not_active).transform("sum")

    # Remove runs below the minimum threshold
    result[dunkelflaute_hourly & (group_lengths < min_hours)] = False
    result.name = "is_dunkelflaute_event_hour"
    return result


def catalogue_events(
    dunkelflaute_filtered: pd.Series,
    wind_cf: pd.Series,
    solar_cf: Optional[pd.Series] = None,
) -> list[DunkelflauteEvent]:
    """
    Extracts a structured catalogue of Dunkelflaute events from the filtered
    binary classification. Each event is represented as a DunkelflauteEvent
    dataclass with start time, duration, and generation metrics.

    The event catalogue is used in the commodity linkage notebook to compute
    forecast lead time statistics and to match events against observed price
    movements.

    Parameters
    ----------
    dunkelflaute_filtered:
        Filtered binary Dunkelflaute classification, as returned by
        apply_minimum_duration_filter.
    wind_cf:
        Hourly wind capacity factor Series, aligned with the classification.
    solar_cf:
        Hourly solar capacity factor Series. Optional.

    Returns
    -------
    list[DunkelflauteEvent]
        List of DunkelflauteEvent instances, one per identified event.
    """
    events: list[DunkelflauteEvent] = []

    not_active = (~dunkelflaute_filtered).cumsum()
    groups     = dunkelflaute_filtered[dunkelflaute_filtered].groupby(not_active)

    for _, group_index in groups.groups.items():
        if len(group_index) == 0:
            continue

        start = group_index[0]
        end   = group_index[-1]
        mask  = (wind_cf.index >= start) & (wind_cf.index <= end)

        mean_wind  = float(wind_cf[mask].mean())
        mean_solar = float(solar_cf[mask].mean()) if solar_cf is not None else float("nan")

        combined_cf    = wind_cf[mask].copy()
        if solar_cf is not None:
            combined_cf = combined_cf + solar_cf[mask]
        max_deficit = float((1.0 - combined_cf).max())

        events.append(DunkelflauteEvent(
            start=start,
            end=end,
            duration_hours=len(group_index),
            mean_wind_cf=mean_wind,
            mean_solar_cf=mean_solar,
            max_deficit=max_deficit,
        ))

    logger.info("Identified %d Dunkelflaute events.", len(events))
    return events


def events_to_dataframe(events: list[DunkelflauteEvent]) -> pd.DataFrame:
    """
    Converts a list of DunkelflauteEvent instances to a tabular DataFrame
    for analysis and output to the commodity linkage notebook.

    Parameters
    ----------
    events:
        List of DunkelflauteEvent instances, as returned by catalogue_events.

    Returns
    -------
    pd.DataFrame
        One row per event with columns for all DunkelflauteEvent attributes.
        Index is the event start timestamp, UTC.
    """
    if not events:
        return pd.DataFrame(
            columns=["start", "end", "duration_hours", "mean_wind_cf",
                     "mean_solar_cf", "max_deficit"]
        )

    records = [
        {
            "start":           ev.start,
            "end":             ev.end,
            "duration_hours":  ev.duration_hours,
            "mean_wind_cf":    ev.mean_wind_cf,
            "mean_solar_cf":   ev.mean_solar_cf,
            "max_deficit":     ev.max_deficit,
        }
        for ev in events
    ]
    df = pd.DataFrame(records).set_index("start")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def compute_forward_probability(
    dunkelflaute_hourly: pd.Series,
    horizons_hours: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Computes the fraction of hours within each forward window that will be
    Dunkelflaute hours. This is the key model target for probabilistic
    Dunkelflaute forecasting.

    A forward probability of 0.7 at horizon 48 hours means that 70% of the
    next 48 hours are expected to be Dunkelflaute hours. A commodity trader
    can use this probability to size a gas-for-power demand position.

    Note that these forward probabilities are computed from actuals and should
    only be used as training targets, not as features. Using them as features
    would introduce forward-looking bias.

    Parameters
    ----------
    dunkelflaute_hourly:
        Binary Dunkelflaute classification (unfiltered or filtered), UTC-indexed.
    horizons_hours:
        Forward window lengths in hours for probability computation.
        Defaults to [24, 48, 72].

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per horizon, each containing the fraction of
        the next n hours that are Dunkelflaute hours.
    """
    if horizons_hours is None:
        horizons_hours = [24, 48, 72]

    frames = {}
    for h in horizons_hours:
        # Rolling sum over the next h hours (shift by -h to align with the start)
        forward_sum  = (
            dunkelflaute_hourly.astype(float)
            .iloc[::-1]
            .rolling(h, min_periods=1)
            .mean()
            .iloc[::-1]
        )
        frames[f"dunkelflaute_prob_{h}h_forward"] = forward_sum

    return pd.DataFrame(frames, index=dunkelflaute_hourly.index)


def compute_renewable_deficit_features(
    wind_cf: pd.Series,
    solar_cf: Optional[pd.Series] = None,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Constructs rolling deficit features that quantify the recent and current
    state of renewable generation relative to maximum possible output.

    The renewable deficit (1 minus capacity factor) represents the proportion
    of potential generation that the wind and solar fleet is failing to produce.
    A persistently high deficit drives gas demand from the power sector, and
    the rolling accumulation of this deficit is a strong predictor of gas price
    movement in the short term.

    Parameters
    ----------
    wind_cf:
        Hourly wind capacity factor Series.
    solar_cf:
        Hourly solar capacity factor Series. Optional.
    windows:
        Rolling window lengths in hours. Defaults to DEFICIT_WINDOWS.

    Returns
    -------
    pd.DataFrame
        Rolling deficit features for wind, solar (if available), and combined.
    """
    if windows is None:
        windows = DEFICIT_WINDOWS

    wind_deficit  = (1.0 - wind_cf).clip(lower=0.0)
    wind_deficit.name = "wind_deficit"

    frames = [wind_deficit]
    deficit_series = {"wind": wind_deficit}

    if solar_cf is not None:
        solar_deficit = (1.0 - solar_cf).clip(lower=0.0)
        solar_deficit.name = "solar_deficit"
        combined_deficit = ((wind_deficit + solar_deficit) / 2.0)
        combined_deficit.name = "combined_renewable_deficit"
        frames.append(solar_deficit)
        frames.append(combined_deficit)
        deficit_series["solar"]    = solar_deficit
        deficit_series["combined"] = combined_deficit
    else:
        deficit_series["combined"] = wind_deficit

    rolling_frames = []
    for name, series in deficit_series.items():
        for w in windows:
            min_periods = max(1, w // 2)
            rolling_frames.append(
                series.rolling(w, min_periods=min_periods)
                      .mean()
                      .rename(f"{name}_deficit_{w}h_mean")
            )
            rolling_frames.append(
                series.rolling(w, min_periods=min_periods)
                      .sum()
                      .rename(f"{name}_deficit_{w}h_sum")
            )

    result = pd.concat(frames + rolling_frames, axis=1)
    result.index = pd.to_datetime(result.index, utc=True)
    return result


def add_event_proximity_features(
    df: pd.DataFrame,
    events: list[DunkelflauteEvent],
    proximity_windows_hours: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Adds binary flags and countdown features indicating proximity to known
    Dunkelflaute events. These features are used in the commodity linkage
    notebook to evaluate how far in advance the model provides actionable signal.

    In a live forecasting context, these features would be replaced by the
    model's own probabilistic output. Their inclusion here is for backtesting
    purposes: we use the known event times to assess what the ideal lead-time
    signal would have looked like.

    Parameters
    ----------
    df:
        DataFrame with a UTC DatetimeIndex to which event proximity features
        will be appended.
    events:
        List of DunkelflauteEvent instances, as returned by catalogue_events.
    proximity_windows_hours:
        List of look-ahead windows in hours for proximity flag creation.
        A flag for window h is True if a Dunkelflaute event starts within
        the next h hours. Defaults to [24, 48, 72, 120].

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional proximity flag columns.
    """
    if proximity_windows_hours is None:
        proximity_windows_hours = [24, 48, 72, 120]

    result = df.copy()

    for h in proximity_windows_hours:
        flag = pd.Series(False, index=result.index, name=f"event_within_{h}h")
        for event in events:
            window_start = event.start - pd.Timedelta(hours=h)
            flag.loc[(result.index >= window_start) & (result.index < event.start)] = True
        result[flag.name] = flag.astype(int)

    return result


def build_dunkelflaute_features(
    wind_generation_mw: pd.Series,
    installed_wind_mw: float,
    solar_generation_mw: Optional[pd.Series] = None,
    installed_solar_mw: Optional[float] = None,
    wind_threshold: float = CF_WIND_THRESHOLD,
    solar_threshold: float = CF_SOLAR_THRESHOLD,
    min_event_hours: int = MIN_EVENT_HOURS,
) -> tuple[pd.DataFrame, list[DunkelflauteEvent]]:
    """
    Orchestrates the full Dunkelflaute feature construction pipeline in a
    single call. Computes capacity factors, classifies events, applies the
    duration filter, catalogues events, and constructs all forward-looking
    probability and deficit features.

    Parameters
    ----------
    wind_generation_mw:
        Hourly wind generation in MW, UTC-indexed.
    installed_wind_mw:
        Installed wind capacity in MW for the relevant period.
    solar_generation_mw:
        Hourly solar PV generation in MW. Optional.
    installed_solar_mw:
        Installed solar PV capacity in MW. Required if solar_generation_mw
        is provided.
    wind_threshold:
        Wind capacity factor threshold for Dunkelflaute classification.
    solar_threshold:
        Solar capacity factor threshold for Dunkelflaute classification.
    min_event_hours:
        Minimum event duration in hours for the catalogue filter.

    Returns
    -------
    tuple[pd.DataFrame, list[DunkelflauteEvent]]
        Feature DataFrame (hourly UTC-indexed) and the event catalogue.
    """
    wind_cf = compute_wind_capacity_factor(wind_generation_mw, installed_wind_mw)

    solar_cf: Optional[pd.Series] = None
    if solar_generation_mw is not None and installed_solar_mw is not None:
        solar_cf = compute_solar_capacity_factor(solar_generation_mw, installed_solar_mw)

    hourly_flags = classify_dunkelflaute_hours(
        wind_cf,
        solar_cf=solar_cf,
        wind_threshold=wind_threshold,
        solar_threshold=solar_threshold,
    )
    filtered_flags = apply_minimum_duration_filter(hourly_flags, min_hours=min_event_hours)
    events         = catalogue_events(filtered_flags, wind_cf, solar_cf=solar_cf)

    forward_probs  = compute_forward_probability(filtered_flags)
    deficit_feats  = compute_renewable_deficit_features(wind_cf, solar_cf=solar_cf)

    base = pd.DataFrame({
        "wind_cf":                   wind_cf,
        "is_dunkelflaute_hour":      hourly_flags.astype(int),
        "is_dunkelflaute_event_hour": filtered_flags.astype(int),
    })
    if solar_cf is not None:
        base["solar_cf"] = solar_cf

    feature_df = pd.concat([base, forward_probs, deficit_feats], axis=1)
    feature_df = add_event_proximity_features(feature_df, events)
    feature_df.index = pd.to_datetime(feature_df.index, utc=True)

    logger.info(
        "Built Dunkelflaute feature set: %d rows, %d columns, %d events catalogued.",
        len(feature_df), len(feature_df.columns), len(events),
    )
    return feature_df, events
