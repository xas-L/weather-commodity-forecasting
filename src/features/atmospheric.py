"""
Here we construct upper-atmosphere and synoptic-scale features from ERA5 pressure-level
fields. These features represent the large-scale circulation state that determines
surface weather regimes over NW Europe on the timescales relevant to energy trading.

The primary variables used here are:
    Z500   500 hPa geopotential height, the single most informative field for
           identifying blocking patterns and downstream surface temperature regimes.
    T850   850 hPa temperature, a lower-troposphere thermal indicator used as a
           proxy for warm and cold air advection into NW Europe.
    U300   300 hPa zonal wind, used to diagnose jet stream position and strength.
    U850   850 hPa zonal wind, combined with U300 to compute wind shear.

All xarray inputs to this module are expected to come from era5_pipeline.py.
All pandas outputs are UTC-indexed and ready for concatenation into the model
feature matrix.

Setup required before use:
    pip install xarray numpy pandas netcdf4
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# Reference point coordinates for blocking and teleconnection indices.
# These are standard meteorological reference locations used in the peer-reviewed
# literature and by operational NWP centres for regime diagnostics.
AZORES_LAT,   AZORES_LON   =  37.5, -25.5
ICELAND_LAT,  ICELAND_LON  =  65.0, -22.5
GREENLAND_LAT_BOUNDS        = (75.0,  60.0)
GREENLAND_LON_BOUNDS        = (-55.0, -15.0)

# Jet stream diagnostic latitude bands. The North Atlantic eddy-driven jet
# typically occupies 40-60N. Its position is the primary control on whether
# NW Europe experiences westerly flow (mild, wet) or meridional flow (cold, dry).
JET_LAT_NORTH: float = 60.0
JET_LAT_SOUTH: float = 40.0
JET_LON_WEST:  float = -30.0
JET_LON_EAST:  float = 10.0

# Standard pressure levels expected in the ERA5 pressure-level dataset.
LEVEL_500: int = 500
LEVEL_850: int = 850
LEVEL_300: int = 300


def extract_z500(ds: xr.Dataset) -> xr.DataArray:
    """
    Extracts the 500 hPa geopotential height field from the ERA5 pressure-level
    dataset and converts from geopotential (m^2 s^-2) to geopotential height (m).

    Z500 is the foundational upper-atmosphere variable in this project. It
    characterises the large-scale pressure pattern at the mid-tropospheric level
    where synoptic-scale Rossby waves propagate. Ridges (high Z500 anomaly) tend
    to steer surface cold air southward; troughs (low anomaly) tend to bring
    milder westerly flow to NW Europe.

    Parameters
    ----------
    ds:
        ERA5 pressure-level xarray Dataset containing the 'z' geopotential variable.

    Returns
    -------
    xr.DataArray
        Z500 in geopotential metres, dimensions (time, latitude, longitude).

    Raises
    ------
    KeyError
        If the 'z' variable or 500 hPa level is absent from the dataset.
    """
    standard_gravity = 9.80665

    if "z" not in ds:
        raise KeyError("Geopotential variable 'z' not found in dataset.")
    if LEVEL_500 not in ds["pressure_level"].values:
        raise KeyError(f"Pressure level {LEVEL_500} hPa not present in dataset.")

    z500 = ds["z"].sel(pressure_level=LEVEL_500) / standard_gravity
    z500.attrs["units"]     = "m"
    z500.attrs["long_name"] = "500 hPa Geopotential Height"
    return z500


def extract_t850(ds: xr.Dataset) -> xr.DataArray:
    """
    Extracts 850 hPa temperature from the ERA5 pressure-level dataset and
    converts from Kelvin to Celsius.

    T850 is used as a proxy for the thermal character of the air mass
    covering NW Europe. Cold air advection events (rapid T850 falls) are
    associated with sharp HDD spikes and the kind of demand surges that move
    the TTF front-week contract materially.

    Parameters
    ----------
    ds:
        ERA5 pressure-level xarray Dataset containing the 't' temperature variable.

    Returns
    -------
    xr.DataArray
        T850 in degrees Celsius, dimensions (time, latitude, longitude).
    """
    if "t" not in ds:
        raise KeyError("Temperature variable 't' not found in dataset.")
    if LEVEL_850 not in ds["pressure_level"].values:
        raise KeyError(f"Pressure level {LEVEL_850} hPa not present in dataset.")

    t850 = ds["t"].sel(pressure_level=LEVEL_850) - 273.15
    t850.attrs["units"]     = "degC"
    t850.attrs["long_name"] = "850 hPa Temperature"
    return t850


def compute_z500_anomaly(z500: xr.DataArray) -> xr.DataArray:
    """
    Removes the day-of-year climatological mean from the Z500 field to produce
    standardised anomalies.

    The climatology is computed as the multi-year mean for each calendar day
    across the entire time dimension of the input array. The anomaly field
    isolates the dynamically meaningful departures from the seasonal cycle,
    which are the signals that carry subseasonal predictability.

    Parameters
    ----------
    z500:
        Z500 field in geopotential metres, as returned by extract_z500.

    Returns
    -------
    xr.DataArray
        Z500 anomaly in geopotential metres, same dimensions as input.
    """
    climatology = z500.groupby("time.dayofyear").mean("time")
    anomaly     = z500.groupby("time.dayofyear") - climatology
    anomaly.attrs["long_name"] = "500 hPa Geopotential Height Anomaly"
    return anomaly


def compute_t850_anomaly(t850: xr.DataArray) -> xr.DataArray:
    """
    Removes the day-of-year climatological mean from the T850 field.
    The anomaly quantifies whether the mid-troposphere air mass over
    a given region is warmer or colder than the seasonal expectation.

    Parameters
    ----------
    t850:
        T850 field in degrees Celsius, as returned by extract_t850.

    Returns
    -------
    xr.DataArray
        T850 anomaly in degrees Celsius, same dimensions as input.
    """
    climatology = t850.groupby("time.dayofyear").mean("time")
    anomaly     = t850.groupby("time.dayofyear") - climatology
    anomaly.attrs["long_name"] = "850 hPa Temperature Anomaly"
    return anomaly


def compute_nao_z500_proxy(z500_anomaly: xr.DataArray) -> pd.Series:
    """
    Computes the Z500-based NAO proxy index as the standardised geopotential
    height difference between the Azores and Iceland reference points.

    This follows the Hurrell (1995) sea-level pressure definition adapted for
    the upper troposphere, and provides a dynamically consistent NAO measure
    that can be derived entirely from gridded reanalysis without requiring
    station data.

    A strongly positive value indicates enhanced westerlies over the North
    Atlantic, associated with mild wet conditions over NW Europe. A strongly
    negative value indicates weakened or reversed pressure gradient, associated
    with blocking, cold air outbreaks, and elevated gas demand.

    Parameters
    ----------
    z500_anomaly:
        Z500 anomaly field, as returned by compute_z500_anomaly.

    Returns
    -------
    pd.Series
        NAO proxy index, standardised to zero mean and unit variance,
        time-indexed in UTC.
    """
    azores  = z500_anomaly.interp(
        latitude=AZORES_LAT, longitude=AZORES_LON, method="linear"
    )
    iceland = z500_anomaly.interp(
        latitude=ICELAND_LAT, longitude=ICELAND_LON, method="linear"
    )

    raw = (azores - iceland).to_series()
    standardised = (raw - raw.mean()) / raw.std()
    standardised.name = "nao_z500_proxy"
    standardised.index = pd.to_datetime(standardised.index, utc=True)
    return standardised


def compute_greenland_blocking_index(z500_anomaly: xr.DataArray) -> pd.Series:
    """
    Computes an area-averaged Z500 anomaly over the Greenland blocking region.

    Greenland blocking is the dominant mid-tropospheric configuration associated
    with cold air outbreak events over NW Europe. A positive index indicates a
    persistent high-pressure ridge over Greenland that deflects the jet stream
    southward and allows Arctic air to advance into the TTF supply region.

    The area average uses cosine-latitude weighting to account for the
    convergence of meridians at higher latitudes, following standard
    meteorological practice.

    Parameters
    ----------
    z500_anomaly:
        Z500 anomaly field, as returned by compute_z500_anomaly.

    Returns
    -------
    pd.Series
        Greenland blocking index in geopotential metres, UTC-indexed.
    """
    lat_north, lat_south = GREENLAND_LAT_BOUNDS
    lon_west,  lon_east  = GREENLAND_LON_BOUNDS

    region  = z500_anomaly.sel(
        latitude=slice(lat_north, lat_south),
        longitude=slice(lon_west, lon_east),
    )
    weights  = np.cos(np.deg2rad(region.latitude))
    blocking = region.weighted(weights).mean(["latitude", "longitude"])

    series       = blocking.to_series()
    series.name  = "greenland_blocking_index"
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def compute_scandinavia_ridge_index(z500_anomaly: xr.DataArray) -> pd.Series:
    """
    Computes the Scandinavian ridge index as the area-mean Z500 anomaly over
    Scandinavia. A positive ridge over Scandinavia channels cold easterly flow
    into Germany and the Netherlands, creating a distinct cold-regime signature
    that the Greenland blocking index alone does not capture.

    The Scandinavian pattern (SCA) is one of the four main low-frequency
    teleconnection patterns identified by Barnston and Livezey (1987) and is
    particularly relevant for late-autumn and early-winter cold anomalies
    in NW Europe.

    Parameters
    ----------
    z500_anomaly:
        Z500 anomaly field, as returned by compute_z500_anomaly.

    Returns
    -------
    pd.Series
        Scandinavian ridge index in geopotential metres, UTC-indexed.
    """
    region  = z500_anomaly.sel(
        latitude=slice(70.0, 55.0),
        longitude=slice(5.0, 30.0),
    )
    weights  = np.cos(np.deg2rad(region.latitude))
    ridge    = region.weighted(weights).mean(["latitude", "longitude"])

    series       = ridge.to_series()
    series.name  = "scandinavia_ridge_index"
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def compute_jet_stream_index(ds: xr.Dataset) -> pd.DataFrame:
    """
    Diagnoses the North Atlantic eddy-driven jet stream position and strength
    from 300 hPa zonal wind.

    The jet stream position determines whether NW Europe sits in the westerly
    flow regime (mild, cloud-bearing airmasses) or to the south of the jet in
    the cold sector. Jet speed provides a measure of circulation vigour that
    correlates with the persistence and predictability of the prevailing regime.

    Position is estimated as the latitude of maximum zonal-mean U300 within the
    North Atlantic sector (30W to 10E, 40N to 70N). Strength is the zonal-mean
    U300 at that latitude.

    Parameters
    ----------
    ds:
        ERA5 pressure-level xarray Dataset containing the 'u' zonal wind variable.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'jet_position_lat' and 'jet_speed_ms', UTC-indexed.

    Raises
    ------
    KeyError
        If the 'u' variable or 300 hPa level is absent from the dataset.
    """
    if "u" not in ds:
        raise KeyError("Zonal wind variable 'u' not found in dataset.")
    if LEVEL_300 not in ds["pressure_level"].values:
        raise KeyError(f"Pressure level {LEVEL_300} hPa not present in dataset.")

    u300 = ds["u"].sel(pressure_level=LEVEL_300)

    jet_region = u300.sel(
        latitude=slice(JET_LAT_NORTH, JET_LAT_SOUTH),
        longitude=slice(JET_LON_WEST, JET_LON_EAST),
    )

    # Zonal mean over the North Atlantic sector
    u300_zonal_mean = jet_region.mean("longitude")

    # Latitude of maximum wind for each time step
    jet_lat_idx = u300_zonal_mean.argmax("latitude")
    jet_lat     = u300_zonal_mean.latitude.isel(latitude=jet_lat_idx)
    jet_speed   = u300_zonal_mean.max("latitude")

    df = pd.DataFrame({
        "jet_position_lat": jet_lat.to_series(),
        "jet_speed_ms":     jet_speed.to_series(),
    })
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def compute_wind_shear(ds: xr.Dataset, region_bounds: Optional[dict] = None) -> pd.Series:
    """
    Computes area-mean vertical wind shear between 300 hPa and 850 hPa as
    the difference in zonal wind between those two levels.

    Strong positive shear (fast upper-level westerlies relative to the surface)
    indicates an active baroclinic zone and frequent storm-track activity,
    associated with variable but generally mild conditions. Weak or negative
    shear indicates a decoupled troposphere, often associated with blocking
    or cold anticyclonic conditions.

    Parameters
    ----------
    ds:
        ERA5 pressure-level xarray Dataset with 'u' zonal wind at both
        300 hPa and 850 hPa.
    region_bounds:
        Optional dict with keys 'lat_north', 'lat_south', 'lon_west', 'lon_east'
        defining the averaging region. Defaults to the NW European domain.

    Returns
    -------
    pd.Series
        Area-mean wind shear in m/s, UTC-indexed.
    """
    if region_bounds is None:
        region_bounds = {
            "lat_north": 60.0,
            "lat_south": 45.0,
            "lon_west":  -5.0,
            "lon_east":  15.0,
        }

    u300 = ds["u"].sel(pressure_level=LEVEL_300)
    u850 = ds["u"].sel(pressure_level=LEVEL_850)

    region_300 = u300.sel(
        latitude=slice(region_bounds["lat_north"], region_bounds["lat_south"]),
        longitude=slice(region_bounds["lon_west"], region_bounds["lon_east"]),
    )
    region_850 = u850.sel(
        latitude=slice(region_bounds["lat_north"], region_bounds["lat_south"]),
        longitude=slice(region_bounds["lon_west"], region_bounds["lon_east"]),
    )

    weights = np.cos(np.deg2rad(region_300.latitude))
    shear   = (
        region_300.weighted(weights).mean(["latitude", "longitude"])
        - region_850.weighted(weights).mean(["latitude", "longitude"])
    )

    series       = shear.to_series()
    series.name  = "wind_shear_300_850"
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def compute_t850_nw_europe(t850_anomaly: xr.DataArray) -> pd.Series:
    """
    Computes the area-mean T850 anomaly over the NW European domain. This
    series is used as a direct lower-troposphere temperature proxy feature in
    the short-term temperature forecast model.

    T850 is less affected by near-surface boundary layer effects than T2m,
    making it a cleaner signal of synoptic-scale warm or cold advection.
    Including it alongside the surface T2m features allows the model to
    distinguish between surface inversions (cold ground, mild aloft) and
    genuine deep-layer cold events.

    Parameters
    ----------
    t850_anomaly:
        T850 anomaly field in degrees Celsius, as returned by compute_t850_anomaly.

    Returns
    -------
    pd.Series
        Area-mean T850 anomaly over NW Europe, UTC-indexed.
    """
    nw_europe = t850_anomaly.sel(
        latitude=slice(60.0, 45.0),
        longitude=slice(-5.0, 15.0),
    )
    weights  = np.cos(np.deg2rad(nw_europe.latitude))
    t850_avg = nw_europe.weighted(weights).mean(["latitude", "longitude"])

    series       = t850_avg.to_series()
    series.name  = "t850_anom_nw_europe"
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def add_blocking_persistence(
    atmospheric_df: pd.DataFrame,
    blocking_col: str = "greenland_blocking_index",
    threshold: float = 30.0,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Adds features capturing the persistence of blocking conditions.

    A single day of elevated blocking index is less significant than a week of
    sustained blocking. Regime persistence is a key feature for medium-range
    forecasting because blocking patterns tend to be self-sustaining on the
    timescale of 5-20 days due to the slow dissipation of Rossby wave energy.

    Parameters
    ----------
    atmospheric_df:
        DataFrame containing the blocking index column.
    blocking_col:
        Name of the blocking index column to use.
    threshold:
        Z500 anomaly threshold in geopotential metres above which a state
        is classified as a blocking day.
    windows:
        Rolling window lengths in days for persistence metrics.
        Defaults to [3, 5, 10].

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional blocking persistence columns.
    """
    if windows is None:
        windows = [3, 5, 10]

    df = atmospheric_df.copy()

    if blocking_col not in df.columns:
        logger.warning(
            "Blocking column '%s' not found. Skipping persistence features.",
            blocking_col,
        )
        return df

    df["blocking_day"] = (df[blocking_col] > threshold).astype(int)

    for w in windows:
        # Fraction of days in the window that were blocking days
        df[f"blocking_freq_{w}d"] = (
            df["blocking_day"].rolling(w, min_periods=max(1, w // 2)).mean()
        )
        # Rolling mean of the index itself over the window
        df[f"{blocking_col}_{w}d_mean"] = (
            df[blocking_col].rolling(w, min_periods=max(1, w // 2)).mean()
        )

    # Consecutive blocking days: how long the current event has lasted.
    # This captures the "stuck" regime signal that is most commercially relevant.
    consecutive = (
        df["blocking_day"]
        .groupby((df["blocking_day"] == 0).cumsum())
        .cumcount()
    )
    df["consecutive_blocking_days"] = consecutive * df["blocking_day"]

    return df


def build_atmospheric_features(
    ds_pressure: xr.Dataset,
    include_jet: bool = True,
    include_shear: bool = True,
) -> pd.DataFrame:
    """
    Orchestrates the full upper-atmosphere feature construction pipeline in a
    single call. Extracts Z500 and T850 fields, computes anomalies, derives all
    index series, and assembles them into a wide DataFrame ready for merging with
    surface and demand features.

    Parameters
    ----------
    ds_pressure:
        ERA5 pressure-level xarray Dataset, as returned by
        era5_pipeline.load_pressure_dataset.
    include_jet:
        If True, compute jet stream position and speed features. Requires
        300 hPa zonal wind in the dataset.
    include_shear:
        If True, compute vertical wind shear features. Requires 300 hPa and
        850 hPa zonal wind in the dataset.

    Returns
    -------
    pd.DataFrame
        Wide UTC-indexed DataFrame containing all atmospheric index features.
    """
    z500      = extract_z500(ds_pressure)
    z500_anom = compute_z500_anomaly(z500)
    t850      = extract_t850(ds_pressure)
    t850_anom = compute_t850_anomaly(t850)

    nao_proxy = compute_nao_z500_proxy(z500_anom)
    blocking  = compute_greenland_blocking_index(z500_anom)
    scand     = compute_scandinavia_ridge_index(z500_anom)
    t850_nwe  = compute_t850_nw_europe(t850_anom)

    series_list = [nao_proxy, blocking, scand, t850_nwe]

    if include_jet:
        try:
            jet_df = compute_jet_stream_index(ds_pressure)
            series_list.append(jet_df)
        except KeyError as exc:
            logger.warning("Jet stream features skipped: %s", exc)

    if include_shear:
        try:
            shear = compute_wind_shear(ds_pressure)
            series_list.append(shear)
        except KeyError as exc:
            logger.warning("Wind shear features skipped: %s", exc)

    df = pd.concat(series_list, axis=1)
    df = df.resample("1D").mean()
    df.index = pd.to_datetime(df.index, utc=True)

    df = add_blocking_persistence(df)

    logger.info(
        "Built atmospheric feature set: %d rows, %d columns.",
        len(df), len(df.columns),
    )
    return df
