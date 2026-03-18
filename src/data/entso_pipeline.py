"""
Here we pull electricity load, wind generation, solar generation, and day-ahead power prices
from the ENTSO-E Transparency Platform for the German (DE) and Dutch (NL) markets.

ENTSO-E is the European Network of Transmission System Operators for Electricity.
Their Transparency Platform publishes actual and forecast grid data at hourly resolution
with a free API. Registration is required to obtain an API key.

Registration: https://transparency.entsoe.eu

Setup required before use:
    pip install entsoe-py pandas pyarrow

Usage:
    from src.data.entso_pipeline import run_entso_pipeline
    run_entso_pipeline(api_key="your-key", output_dir="data/raw")

The ENTSO-E client returns pandas Series or DataFrames indexed in UTC. All outputs
from this module preserve UTC throughout to match ERA5 timestamps downstream.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError, PaginationError

logger = logging.getLogger(__name__)


# EIC country codes used by ENTSO-E for the Transparency Platform API.
COUNTRY_CODES: dict[str, str] = {
    "germany":     "DE",
    "netherlands": "NL",
    "france":      "FR",
    "belgium":     "BE",
    "uk":          "GB",
}

# ENTSO-E PSR (Power System Resource) type codes for generation by source.
# B16 = Solar, B19 = Wind Offshore, B18 = Wind Onshore.
PSR_WIND_ONSHORE:  str = "B18"
PSR_WIND_OFFSHORE: str = "B19"
PSR_SOLAR:         str = "B16"

# Retry configuration for handling transient API failures. ENTSO-E rate-limits
# unauthenticated bursts and occasionally returns 5xx during peak hours.
RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: int = 10


def build_client(api_key: str) -> EntsoePandasClient:
    """
    Instantiates the ENTSO-E pandas client with the provided API key.

    Parameters
    ----------
    api_key:
        Personal API key obtained from the ENTSO-E Transparency Platform.

    Returns
    -------
    EntsoePandasClient
        Authenticated client instance.
    """
    return EntsoePandasClient(api_key=api_key)


def _retry_query(func, *args, **kwargs) -> Optional[pd.Series | pd.DataFrame]:
    """
    Wraps an ENTSO-E API call with simple retry logic to handle transient
    failures. Returns None if all attempts fail, allowing callers to decide
    whether to raise or skip.

    Parameters
    ----------
    func:
        The EntsoePandasClient method to call.
    *args:
        Positional arguments forwarded to func.
    **kwargs:
        Keyword arguments forwarded to func.

    Returns
    -------
    pd.Series or pd.DataFrame or None
        Query result on success, None after exhausting all retries.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except NoMatchingDataError:
            logger.warning("No matching data returned for query: %s %s", args, kwargs)
            return None
        except PaginationError as exc:
            logger.warning("Pagination error on attempt %d/%d: %s", attempt, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        except Exception as exc:
            logger.error("Unexpected error on attempt %d/%d: %s", attempt, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                raise

    return None


def fetch_actual_load(
    client: EntsoePandasClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.Series]:
    """
    Retrieves actual electricity consumption (system load) in MW for a given
    country and time window. Data is sourced from TSO meter readings and is
    typically available with a 1-2 day lag.

    The load series is the primary downstream target for validating HDD-based
    demand forecasts. Weekly load anomalies relative to the seasonal mean
    provide a realised measure of weather-driven demand deviation.

    Parameters
    ----------
    client:
        Authenticated ENTSO-E client.
    country_code:
        Two-letter EIC country code, e.g. 'DE' or 'NL'.
    start:
        Start timestamp, UTC-aware.
    end:
        End timestamp, UTC-aware.

    Returns
    -------
    pd.Series or None
        Hourly load in MW, UTC-indexed. None if no data is available.
    """
    logger.info("Fetching actual load for %s from %s to %s.", country_code, start.date(), end.date())
    result = _retry_query(client.query_load, country_code, start=start, end=end)

    if result is None:
        logger.warning("No load data returned for %s.", country_code)
        return None

    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]

    result = result.resample("1h").mean()
    result.name = f"load_mw_{country_code.lower()}"
    return result


def fetch_wind_generation(
    client: EntsoePandasClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Retrieves actual wind generation (onshore and offshore combined) in MW.
    Wind generation is a direct function of hub-height wind speed at the grid
    level and is used to compute capacity factors for Dunkelflaute detection.

    Parameters
    ----------
    client:
        Authenticated ENTSO-E client.
    country_code:
        Two-letter EIC country code.
    start:
        Start timestamp, UTC-aware.
    end:
        End timestamp, UTC-aware.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns 'wind_onshore_mw' and 'wind_offshore_mw',
        hourly UTC-indexed. None if no data is available.
    """
    logger.info("Fetching wind generation for %s.", country_code)

    onshore  = _retry_query(
        client.query_generation, country_code, start=start, end=end, psr_type=PSR_WIND_ONSHORE
    )
    offshore = _retry_query(
        client.query_generation, country_code, start=start, end=end, psr_type=PSR_WIND_OFFSHORE
    )

    frames = {}
    if onshore is not None:
        on_series = onshore.iloc[:, 0] if isinstance(onshore, pd.DataFrame) else onshore
        frames["wind_onshore_mw"] = on_series.resample("1h").mean()

    if offshore is not None:
        off_series = offshore.iloc[:, 0] if isinstance(offshore, pd.DataFrame) else offshore
        frames["wind_offshore_mw"] = off_series.resample("1h").mean()

    if not frames:
        logger.warning("No wind generation data returned for %s.", country_code)
        return None

    df = pd.DataFrame(frames)
    df["wind_total_mw"] = df.sum(axis=1)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fetch_solar_generation(
    client: EntsoePandasClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.Series]:
    """
    Retrieves actual solar photovoltaic generation in MW. Used alongside wind
    generation to detect Dunkelflaute periods where both renewable sources are
    simultaneously suppressed, forcing greater reliance on gas-fired generation.

    Parameters
    ----------
    client:
        Authenticated ENTSO-E client.
    country_code:
        Two-letter EIC country code.
    start:
        Start timestamp, UTC-aware.
    end:
        End timestamp, UTC-aware.

    Returns
    -------
    pd.Series or None
        Hourly solar generation in MW, UTC-indexed. None if unavailable.
    """
    logger.info("Fetching solar generation for %s.", country_code)

    result = _retry_query(
        client.query_generation, country_code, start=start, end=end, psr_type=PSR_SOLAR
    )

    if result is None:
        logger.warning("No solar generation data returned for %s.", country_code)
        return None

    series = result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
    series = series.resample("1h").mean()
    series.name = f"solar_mw_{country_code.lower()}"
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def fetch_day_ahead_prices(
    client: EntsoePandasClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.Series]:
    """
    Retrieves day-ahead electricity prices in EUR/MWh. These are used in the
    commodity linkage notebook to assess whether Dunkelflaute forecasts contain
    information about price direction and magnitude.

    Day-ahead prices are published by the power exchange (EPEX SPOT for Germany
    and the Netherlands) at approximately 12:30 CET each day for the following
    delivery day, so there is a natural 12-18 hour publication lag to account for.

    Parameters
    ----------
    client:
        Authenticated ENTSO-E client.
    country_code:
        Two-letter EIC country code.
    start:
        Start timestamp, UTC-aware.
    end:
        End timestamp, UTC-aware.

    Returns
    -------
    pd.Series or None
        Hourly day-ahead prices in EUR/MWh, UTC-indexed. None if unavailable.
    """
    logger.info("Fetching day-ahead prices for %s.", country_code)

    result = _retry_query(client.query_day_ahead_prices, country_code, start=start, end=end)

    if result is None:
        logger.warning("No day-ahead prices returned for %s.", country_code)
        return None

    series = result.resample("1h").mean()
    series.name = f"da_price_eur_mwh_{country_code.lower()}"
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def fetch_installed_capacity(
    client: EntsoePandasClient,
    country_code: str,
    year: int,
) -> Optional[pd.DataFrame]:
    """
    Retrieves annual installed generation capacity by technology type in MW.
    This is used to convert raw generation volumes into capacity factors, which
    are dimensionless fractions that remove the upward trend in absolute
    generation from year-to-year capacity growth.

    Capacity factors are the correct metric for Dunkelflaute thresholds because
    the threshold should be relative to available capacity, not absolute MW.

    Parameters
    ----------
    client:
        Authenticated ENTSO-E client.
    country_code:
        Two-letter EIC country code.
    year:
        Reference year for installed capacity lookup.

    Returns
    -------
    pd.DataFrame or None
        Installed capacity by production type in MW for the given year.
    """
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end   = pd.Timestamp(f"{year}-12-31", tz="UTC")

    logger.info("Fetching installed capacity for %s (%d).", country_code, year)
    result = _retry_query(client.query_installed_generation_capacity, country_code,
                          start=start, end=end)

    if result is None:
        logger.warning("No capacity data returned for %s (%d).", country_code, year)
        return None

    return result


def compute_capacity_factors(
    generation: pd.Series,
    capacity_mw: float,
    name: str,
) -> pd.Series:
    """
    Converts hourly generation in MW to a capacity factor (0 to 1) by dividing
    by the installed capacity for the corresponding year. Values are clipped to
    [0, 1] to handle small over-generation artefacts in the reported data.

    Parameters
    ----------
    generation:
        Hourly generation in MW.
    capacity_mw:
        Installed capacity in MW for the relevant year.
    name:
        Name to assign to the output Series.

    Returns
    -------
    pd.Series
        Hourly capacity factor, bounded between 0 and 1.
    """
    if capacity_mw <= 0:
        raise ValueError(f"Installed capacity must be positive, got {capacity_mw}.")

    cf = (generation / capacity_mw).clip(lower=0.0, upper=1.0)
    cf.name = name
    return cf


def build_country_dataset(
    client: EntsoePandasClient,
    country_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
) -> dict[str, pd.Series | pd.DataFrame]:
    """
    Runs the full ENTSO-E ingestion workflow for a single country: load,
    wind, solar, and prices. Each series is saved to a separate Parquet file
    named by country and data type.

    Parameters
    ----------
    client:
        Authenticated ENTSO-E client.
    country_key:
        Key from COUNTRY_CODES, e.g. 'germany'.
    start:
        Start timestamp, UTC-aware.
    end:
        End timestamp, UTC-aware.
    output_dir:
        Directory in which to write Parquet files.

    Returns
    -------
    dict
        Mapping of data type label to the fetched Series or DataFrame.
        Missing data types are absent from the dict rather than None-valued.
    """
    if country_key not in COUNTRY_CODES:
        raise KeyError(
            f"Unknown country '{country_key}'. Available: {list(COUNTRY_CODES.keys())}"
        )

    cc = COUNTRY_CODES[country_key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.Series | pd.DataFrame] = {}

    fetchers = {
        "load":        lambda: fetch_actual_load(client, cc, start, end),
        "wind":        lambda: fetch_wind_generation(client, cc, start, end),
        "solar":       lambda: fetch_solar_generation(client, cc, start, end),
        "da_prices":   lambda: fetch_day_ahead_prices(client, cc, start, end),
    }

    for label, fetch_fn in fetchers.items():
        data = fetch_fn()
        if data is not None:
            path = output_dir / f"entso_{country_key}_{label}.parquet"
            data.to_parquet(path)
            logger.info("Saved %s %s to %s.", country_key, label, path)
            results[label] = data
        else:
            logger.warning("Skipping %s %s: no data returned.", country_key, label)

    return results


def merge_country_features(
    country_results: dict[str, pd.Series | pd.DataFrame],
    country_key: str,
) -> pd.DataFrame:
    """
    Concatenates all fetched series for a country into a single wide DataFrame
    aligned on a common UTC hourly index. Missing values introduced by
    resampling mismatches are left as NaN for the feature engineering step
    to handle.

    Parameters
    ----------
    country_results:
        Output from build_country_dataset.
    country_key:
        Country identifier, used to prefix column names.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with all available series as columns, hourly UTC-indexed.
    """
    frames = []
    for label, data in country_results.items():
        if isinstance(data, pd.Series):
            frames.append(data.rename(f"{country_key}_{label}"))
        elif isinstance(data, pd.DataFrame):
            data.columns = [f"{country_key}_{col}" for col in data.columns]
            frames.append(data)

    if not frames:
        logger.warning("No data available for %s to merge.", country_key)
        return pd.DataFrame()

    merged = pd.concat(frames, axis=1)
    merged = merged.resample("1h").mean()
    merged.index = pd.to_datetime(merged.index, utc=True)
    return merged


def run_entso_pipeline(
    api_key: str,
    output_dir: Path = Path("data/raw"),
    countries: Optional[list[str]] = None,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
) -> dict[str, pd.DataFrame]:
    """
    Top-level entry point for the ENTSO-E data pipeline. Fetches all data types
    for all specified countries and saves each to Parquet.

    Parameters
    ----------
    api_key:
        ENTSO-E Transparency Platform API key.
    output_dir:
        Directory in which to save output files.
    countries:
        Country keys to fetch. Defaults to Germany and Netherlands, which are
        the primary markets for TTF and EPEX desk relevance.
    start_date:
        ISO date string for the start of the retrieval window, e.g. '2018-01-01'.
    end_date:
        ISO date string for the end of the retrieval window.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of country key to its merged wide DataFrame.
    """
    if countries is None:
        countries = ["germany", "netherlands"]

    client = build_client(api_key)
    start  = pd.Timestamp(start_date, tz="UTC")
    end    = pd.Timestamp(end_date,   tz="UTC")

    all_merged: dict[str, pd.DataFrame] = {}

    for country in countries:
        logger.info("Processing ENTSO-E data for %s.", country)
        results = build_country_dataset(client, country, start, end, output_dir)
        merged  = merge_country_features(results, country)
        all_merged[country] = merged

    logger.info("ENTSO-E pipeline complete for countries: %s.", countries)
    return all_merged
