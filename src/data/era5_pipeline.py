"""
Here we handle ingestion of ERA5 reanalysis data from the Copernicus Climate Data Store (CDS),
processing of NetCDF files via xarray, extraction of point-level time series for
target market locations, and persistence to Parquet for downstream feature engineering.

ERA5 is produced by ECMWF and is the primary ground-truth dataset for this project.
Single-level surface variables (T2m, wind, solar, precipitation) and pressure-level
fields (Z500, T850) are handled separately, as CDS serves them via different endpoints.

Setup required before use:
    pip install cdsapi xarray netcdf4 pandas pyarrow

    Create ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api/v2
        key: <your-uid>:<your-api-key>
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# Market-relevant point locations. All coordinates are (latitude, longitude).
# These anchor the point-level T2m and wind extractions to gas and power hubs.
MARKET_LOCATIONS: dict[str, tuple[float, float]] = {
    "amsterdam":  (52.37,  4.90),   # TTF hub region, Netherlands
    "hamburg":    (53.55,  9.99),   # North German gas demand centre
    "frankfurt":  (50.11,  8.68),   # Central European power load centre
    "london":     (51.51, -0.13),   # NBP hub region, UK
    "paris":      (48.86,  2.35),   # French power market
    "berlin":     (52.52, 13.41),   # German capital, large heating load
}

# European domain bounding box: [N, W, S, E]
EUROPE_DOMAIN: list[float] = [75.0, -15.0, 35.0, 40.0]

# Surface variables required for this project. Each maps to the CDS variable name.
SURFACE_VARIABLES: list[str] = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_precipitation",
    "mean_sea_level_pressure",
]

# Pressure-level variables and the levels to retrieve.
PRESSURE_VARIABLES: list[str] = ["geopotential", "temperature"]
PRESSURE_LEVELS: list[str] = ["500", "850"]


def build_cds_client() -> cdsapi.Client:
    """
    Instantiates the CDS API client. Credentials are read from ~/.cdsapirc
    automatically by the library. Raises if the credentials file is absent.
    """
    return cdsapi.Client()


def request_surface_data(
    client: cdsapi.Client,
    output_path: Path,
    years: list[int],
    months: Optional[list[int]] = None,
    time_steps: Optional[list[str]] = None,
) -> Path:
    """
    Submits a CDS request for ERA5 single-level surface variables over the
    European domain and saves the result as a NetCDF file.

    CDS requests are asynchronous and can take from minutes to several hours
    depending on queue length. The client blocks until the download completes.

    Parameters
    ----------
    client:
        Authenticated CDS API client instance.
    output_path:
        Destination path for the downloaded NetCDF file.
    years:
        Calendar years to retrieve, e.g. list(range(2015, 2024)).
    months:
        Months to include (1-12). Defaults to all twelve months.
    time_steps:
        UTC hour strings, e.g. ["00:00", "03:00", ...]. Defaults to 3-hourly,
        which gives 8 steps per day and keeps file sizes manageable.

    Returns
    -------
    Path
        The output path, confirmed to exist after download.
    """
    if months is None:
        months = list(range(1, 13))
    if time_steps is None:
        time_steps = [f"{h:02d}:00" for h in range(0, 24, 3)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info("Surface file already exists at %s, skipping download.", output_path)
        return output_path

    logger.info(
        "Requesting ERA5 surface data for years %s, %d months, %d time steps.",
        years, len(months), len(time_steps),
    )

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": SURFACE_VARIABLES,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": time_steps,
            "area": EUROPE_DOMAIN,
            "format": "netcdf",
        },
        str(output_path),
    )

    logger.info("Surface download complete: %s (%.1f MB).", output_path,
                output_path.stat().st_size / 1e6)
    return output_path


def request_pressure_level_data(
    client: cdsapi.Client,
    output_path: Path,
    years: list[int],
    months: Optional[list[int]] = None,
) -> Path:
    """
    Submits a CDS request for ERA5 pressure-level fields (Z500, T850) over
    Europe and saves as NetCDF. Only synoptic times (00Z, 12Z) are requested
    since we use these fields for regime classification, not high-frequency
    feature engineering.

    Parameters
    ----------
    client:
        Authenticated CDS API client instance.
    output_path:
        Destination path for the downloaded NetCDF file.
    years:
        Calendar years to retrieve.
    months:
        Months to include. Defaults to all twelve months.

    Returns
    -------
    Path
        The output path, confirmed to exist after download.
    """
    if months is None:
        months = list(range(1, 13))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info("Pressure-level file already exists at %s, skipping.", output_path)
        return output_path

    logger.info(
        "Requesting ERA5 pressure-level data for years %s, levels %s.",
        years, PRESSURE_LEVELS,
    )

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": PRESSURE_VARIABLES,
            "pressure_level": PRESSURE_LEVELS,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "12:00"],
            "area": EUROPE_DOMAIN,
            "format": "netcdf",
        },
        str(output_path),
    )

    logger.info("Pressure-level download complete: %s (%.1f MB).", output_path,
                output_path.stat().st_size / 1e6)
    return output_path


def load_surface_dataset(nc_path: Path) -> xr.Dataset:
    """
    Opens the ERA5 surface NetCDF file with chunking suitable for point
    extractions. Uses dask-backed lazy loading so the full grid is not
    loaded into memory until a computation is triggered.

    Parameters
    ----------
    nc_path:
        Path to the downloaded ERA5 surface NetCDF file.

    Returns
    -------
    xr.Dataset
        Lazily loaded dataset with time, latitude, longitude dimensions.
    """
    ds = xr.open_dataset(nc_path, chunks={"time": 500})
    logger.info(
        "Opened surface dataset: %d time steps, %d lat, %d lon.",
        ds.dims.get("time", 0), ds.dims.get("latitude", 0), ds.dims.get("longitude", 0),
    )
    return ds


def load_pressure_dataset(nc_path: Path) -> xr.Dataset:
    """
    Opens the ERA5 pressure-level NetCDF. Dimension order is
    (time, pressure_level, latitude, longitude).

    Parameters
    ----------
    nc_path:
        Path to the downloaded ERA5 pressure-level NetCDF file.

    Returns
    -------
    xr.Dataset
        Lazily loaded dataset.
    """
    ds = xr.open_dataset(nc_path, chunks={"time": 200})
    logger.info(
        "Opened pressure-level dataset with variables: %s.", list(ds.data_vars)
    )
    return ds


def extract_point_series(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    variables: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Extracts a time series at the nearest grid point to (lat, lon) for the
    specified variables. Bilinear interpolation is used rather than
    nearest-neighbour to reduce grid-discretisation errors at point locations.

    Parameters
    ----------
    ds:
        ERA5 xarray Dataset.
    lat:
        Target latitude in decimal degrees.
    lon:
        Target longitude in decimal degrees (negative = west).
    variables:
        List of variable names to extract. If None, all data variables are used.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with one column per variable. Index is UTC.
    """
    if variables is None:
        variables = list(ds.data_vars)

    point = ds[variables].interp(
        latitude=lat,
        longitude=lon,
        method="linear",
    )

    df = point.to_dataframe().reset_index().set_index("time")
    df.index = pd.to_datetime(df.index, utc=True)

    # Drop coordinate columns that xarray carries through interpolation
    coord_cols = [c for c in df.columns if c in ("latitude", "longitude", "level")]
    df = df.drop(columns=coord_cols, errors="ignore")

    return df


def compute_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives scalar wind speed and direction from the u/v components that ERA5
    provides. Both components are required to be present in the DataFrame.

    Parameters
    ----------
    df:
        DataFrame containing u10 and v10 columns (10m wind components).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two additional columns:
        'wind_speed_10m' (m/s) and 'wind_dir_10m' (meteorological degrees).
    """
    u_col = "u10"
    v_col = "v10"

    if u_col not in df.columns or v_col not in df.columns:
        logger.warning(
            "Wind component columns '%s' / '%s' not found. Skipping wind derivation.",
            u_col, v_col,
        )
        return df

    df["wind_speed_10m"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)

    # Meteorological convention: 0 = wind from north, increasing clockwise
    df["wind_dir_10m"] = (
        np.degrees(np.arctan2(-df[u_col], -df[v_col])) % 360
    )

    return df


def compute_t2m_celsius(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts ERA5 2m temperature from Kelvin to Celsius in-place.
    ERA5 stores all temperatures in Kelvin. The commodity markets work in
    Celsius, and all degree-day calculations in this project use Celsius.

    Parameters
    ----------
    df:
        DataFrame containing a 't2m' column in Kelvin.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 't2m' overwritten in Celsius.
    """
    kelvin_col = "t2m"
    if kelvin_col not in df.columns:
        logger.warning("Column '%s' not found, cannot convert to Celsius.", kelvin_col)
        return df

    df[kelvin_col] = df[kelvin_col] - 273.15
    df = df.rename(columns={kelvin_col: "t2m_celsius"})
    return df


def compute_z500(ds_pressure: xr.Dataset) -> xr.DataArray:
    """
    Computes 500 hPa geopotential height (Z500) in metres from the ERA5
    geopotential field (units: m^2 s^-2). Z500 is the primary synoptic field
    used in this project for blocking detection and NAO regime classification.

    Parameters
    ----------
    ds_pressure:
        ERA5 pressure-level dataset containing the 'z' geopotential variable.

    Returns
    -------
    xr.DataArray
        Z500 field in geopotential metres, dimensions (time, latitude, longitude).
    """
    standard_gravity = 9.80665  # m s^-2, WMO standard

    if "z" not in ds_pressure:
        raise KeyError("Geopotential variable 'z' not found in pressure-level dataset.")

    z500 = ds_pressure["z"].sel(pressure_level=500) / standard_gravity
    z500.attrs["units"] = "m"
    z500.attrs["long_name"] = "500 hPa Geopotential Height"
    return z500


def compute_z500_anomaly(z500: xr.DataArray) -> xr.DataArray:
    """
    Removes the day-of-year climatology from Z500 to produce standardised
    anomalies. The anomaly field is used for NAO proxy computation and
    Greenland blocking index calculation.

    Parameters
    ----------
    z500:
        Z500 field in geopotential metres, as returned by compute_z500.

    Returns
    -------
    xr.DataArray
        Z500 anomaly in geopotential metres, same dimensions as input.
    """
    climatology = z500.groupby("time.dayofyear").mean("time")
    anomaly = z500.groupby("time.dayofyear") - climatology
    anomaly.attrs["long_name"] = "500 hPa Geopotential Height Anomaly"
    return anomaly


def compute_nao_z500_proxy(z500_anomaly: xr.DataArray) -> pd.Series:
    """
    Computes a Z500-based NAO proxy as the standardised pressure difference
    between the Azores and Iceland, following the Hurrell (1995) approach.

    The Azores high and Icelandic low are the two pressure centres of action
    that define the NAO. A strongly positive index implies westerly flow over
    NW Europe (mild, wet); a strongly negative index implies blocking and
    potential cold air outbreaks over Germany and the Netherlands.

    Parameters
    ----------
    z500_anomaly:
        Z500 anomaly field, as returned by compute_z500_anomaly.

    Returns
    -------
    pd.Series
        Daily or 12-hourly NAO proxy index, time-indexed in UTC.
    """
    azores_lat, azores_lon   = 37.5, -25.5
    iceland_lat, iceland_lon = 65.0, -22.5

    azores  = z500_anomaly.interp(latitude=azores_lat,  longitude=azores_lon,  method="linear")
    iceland = z500_anomaly.interp(latitude=iceland_lat, longitude=iceland_lon, method="linear")

    nao_raw = (azores - iceland).to_series()
    nao_std = (nao_raw - nao_raw.mean()) / nao_raw.std()
    nao_std.name = "nao_z500_proxy"
    nao_std.index = pd.to_datetime(nao_std.index, utc=True)
    return nao_std


def compute_greenland_blocking_index(z500_anomaly: xr.DataArray) -> pd.Series:
    """
    Computes a Greenland blocking index as the area-mean Z500 anomaly over
    the Greenland/Arctic sector. Sustained positive values indicate a
    blocking anticyclone that deflects the jet stream southward, directing
    cold Arctic air towards NW Europe.

    This index provides a complementary signal to the NAO proxy: while the NAO
    characterises the overall meridional pressure gradient, the blocking index
    specifically captures the Omega-block configuration most correlated with
    European cold extremes.

    Parameters
    ----------
    z500_anomaly:
        Z500 anomaly field, as returned by compute_z500_anomaly.

    Returns
    -------
    pd.Series
        Greenland blocking index, time-indexed in UTC.
    """
    block_region = z500_anomaly.sel(
        latitude=slice(75.0, 60.0),
        longitude=slice(-55.0, -15.0),
    )

    weights = np.cos(np.deg2rad(block_region.latitude))
    blocking = block_region.weighted(weights).mean(["latitude", "longitude"])

    series = blocking.to_series()
    series.name = "greenland_blocking_index"
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def build_location_dataset(
    ds_surface: xr.Dataset,
    location_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Orchestrates the full point-extraction workflow for a single market location:
    loads the surface dataset, extracts the time series, derives wind speed and
    Celsius temperature, and saves the result to Parquet.

    Parameters
    ----------
    ds_surface:
        ERA5 surface xarray Dataset (from load_surface_dataset).
    location_name:
        Key from MARKET_LOCATIONS, e.g. 'amsterdam'.
    output_dir:
        Directory in which to write the output Parquet file.

    Returns
    -------
    pd.DataFrame
        Processed time series for the location, also saved to disk.

    Raises
    ------
    KeyError
        If location_name is not found in MARKET_LOCATIONS.
    """
    if location_name not in MARKET_LOCATIONS:
        raise KeyError(
            f"Unknown location '{location_name}'. "
            f"Available locations: {list(MARKET_LOCATIONS.keys())}"
        )

    lat, lon = MARKET_LOCATIONS[location_name]
    logger.info("Extracting ERA5 point series for %s (%.2fN, %.2fE).", location_name, lat, lon)

    df = extract_point_series(ds_surface, lat, lon)
    df = compute_t2m_celsius(df)
    df = compute_wind_speed(df)

    output_path = Path(output_dir) / f"era5_surface_{location_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    logger.info(
        "Saved %s surface series to %s (%d rows, %d columns).",
        location_name, output_path, len(df), len(df.columns),
    )
    return df


def build_all_locations(
    nc_surface_path: Path,
    output_dir: Path,
    locations: Optional[dict[str, tuple[float, float]]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Extracts and saves point time series for all market locations in a single
    pass over the surface NetCDF file.

    Parameters
    ----------
    nc_surface_path:
        Path to the ERA5 surface NetCDF file.
    output_dir:
        Directory in which Parquet files will be written.
    locations:
        Optional override of the default MARKET_LOCATIONS dict.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of location name to its processed DataFrame.
    """
    if locations is None:
        locations = MARKET_LOCATIONS

    ds = load_surface_dataset(nc_surface_path)
    results = {}

    for name in locations:
        try:
            results[name] = build_location_dataset(ds, name, output_dir)
        except Exception as exc:
            logger.error("Failed to process location '%s': %s", name, exc)

    ds.close()
    return results


def build_regime_fields(
    nc_pressure_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Orchestrates the pressure-level processing workflow: loads Z500, computes
    anomalies, and derives the NAO proxy and Greenland blocking index. All
    three series are concatenated into a single DataFrame and saved to Parquet.

    Parameters
    ----------
    nc_pressure_path:
        Path to the ERA5 pressure-level NetCDF file.
    output_dir:
        Directory in which the output Parquet file will be written.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'z500_anomaly_nao_azores', 'nao_z500_proxy',
        and 'greenland_blocking_index', time-indexed in UTC.
    """
    ds = load_pressure_dataset(nc_pressure_path)
    z500     = compute_z500(ds)
    z500_anom = compute_z500_anomaly(z500)

    nao_proxy = compute_nao_z500_proxy(z500_anom)
    blocking  = compute_greenland_blocking_index(z500_anom)

    df = pd.concat([nao_proxy, blocking], axis=1)
    df.index = pd.to_datetime(df.index, utc=True)

    output_path = Path(output_dir) / "era5_regime_fields.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    logger.info("Saved regime fields to %s (%d rows).", output_path, len(df))
    ds.close()
    return df


def run_era5_pipeline(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    years: Optional[list[int]] = None,
) -> None:
    """
    Top-level entry point for the ERA5 data pipeline. Downloads both surface
    and pressure-level files if they are not already present, then runs all
    extraction and processing steps.

    This function is idempotent: re-running it will skip any files that already
    exist on disk. This is intentional since CDS downloads can take hours.

    Parameters
    ----------
    raw_dir:
        Directory in which raw NetCDF files are stored.
    processed_dir:
        Directory in which processed Parquet files are written.
    years:
        Years to include. Defaults to 2015-2023 inclusive.
    """
    if years is None:
        years = list(range(2015, 2024))

    raw_dir       = Path(raw_dir)
    processed_dir = Path(processed_dir)

    client = build_cds_client()

    surface_nc  = request_surface_data(client, raw_dir / "era5_surface.nc",  years)
    pressure_nc = request_pressure_level_data(client, raw_dir / "era5_pressure.nc", years)

    build_all_locations(surface_nc, processed_dir)
    build_regime_fields(pressure_nc, processed_dir)

    logger.info("ERA5 pipeline complete.")
