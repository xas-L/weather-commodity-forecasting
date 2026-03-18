"""
indices data module for the weather-commodity forecasting:).

Downloads, parses, and persists teleconnection indices published by the NOAA
Climate Prediction Centre (CPC). These indices represent large-scale atmospheric
circulation patterns that influence European weather on subseasonal timescales
(1-4 weeks), well beyond the range of deterministic NWP.

Indices retrieved by this module:
    NAO  (North Atlantic Oscillation)  daily and monthly
    AO   (Arctic Oscillation)          daily and monthly
    PNA  (Pacific/North American)      daily
    ENSO (Nino 3.4 SST anomaly)        monthly

Meteorological rationale for each index is documented at the function level.

All data is sourced from publicly accessible plain-text files on the CPC FTP
mirror, which requires no authentication. File formats vary by index; each has
a dedicated parser.

Setup required before use:
    pip install pandas requests pyarrow
"""

import io
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Source URLs for each CPC index. These are stable long-term file locations.
# Monthly files use a wide (year x month) table format.
# Daily files are a three-column (year, month, day, value) or similar format.
CPC_URLS: dict[str, str] = {
    "nao_monthly": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
        "norm.nao.monthly.b5001.current.ascii.table"
    ),
    "nao_daily": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
        "norm.daily.nao.index.b500101.current.ascii"
    ),
    "ao_monthly": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/"
        "monthly.ao.index.b50.current.ascii.table"
    ),
    "ao_daily": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/"
        "daily.ao.index.b500101.current.ascii"
    ),
    "pna_daily": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
        "norm.daily.pna.index.b500101.current.ascii"
    ),
    "enso_monthly": (
        "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
    ),
}

# Threshold values used downstream for regime classification.
# A trader asking "is this a blocking regime?" uses these thresholds.
NAO_BLOCKING_THRESHOLD: float = -1.5   # standard deviations; strongly negative NAO
NAO_POSITIVE_THRESHOLD: float =  1.0   # strongly positive NAO (westerly flow)
AO_NEGATIVE_THRESHOLD:  float = -1.0   # negative AO; polar vortex weakening


def _download_text(url: str, timeout: int = 30) -> str:
    """
    Downloads a plain-text file from a URL and returns the content as a string.
    Raises an HTTPError if the request does not succeed.

    Parameters
    ----------
    url:
        URL of the plain-text file to download.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    str
        Raw text content of the downloaded file.
    """
    logger.info("Downloading from %s.", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_cpc_monthly_wide(raw_text: str, index_name: str) -> pd.Series:
    """
    Parses CPC monthly index files that are stored in a wide table where rows
    are years and columns are months (Jan=1 through Dec=12). Missing values
    are represented as -99.9 or -999.0 in the source files.

    This is the format used for NAO monthly and AO monthly.

    Parameters
    ----------
    raw_text:
        Raw content of the downloaded monthly index file.
    index_name:
        Name to assign to the resulting Series, e.g. 'NAO' or 'AO'.

    Returns
    -------
    pd.Series
        Monthly time series, indexed at the first day of each month (UTC).
        Missing values are represented as NaN.
    """
    df = pd.read_csv(
        io.StringIO(raw_text),
        sep=r"\s+",
        header=0,
        na_values=["-99.9", "-999.0", "-999", "999.0"],
    )

    # The first column is the year; remaining columns are months
    year_col = df.columns[0]
    month_cols = df.columns[1:]

    melted = df.melt(id_vars=[year_col], value_vars=month_cols,
                     var_name="month_num", value_name=index_name)
    melted.columns = ["year", "month_num", index_name]

    # Month columns may be integer strings or abbreviated names such as 'JAN'
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,  "MAY": 5,  "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    if melted["month_num"].dtype == object:
        melted["month_num"] = melted["month_num"].str.upper().map(month_map)
    else:
        melted["month_num"] = melted["month_num"].astype(int)

    melted["date"] = pd.to_datetime(
        melted[["year", "month_num"]].rename(columns={"month_num": "month"}).assign(day=1)
    )
    series = (
        melted.dropna(subset=["date", index_name])
              .set_index("date")[index_name]
              .sort_index()
    )
    series.index = pd.DatetimeIndex(series.index).tz_localize("UTC")
    return series


def parse_cpc_daily_three_column(raw_text: str, index_name: str) -> pd.Series:
    """
    Parses CPC daily index files stored in a space-separated three-column format
    with columns (year, month, day, value) or sometimes (year day_of_year value).
    Missing values are represented as -99.9 or 999.000.

    This is the format used for NAO daily, AO daily, and PNA daily.

    Parameters
    ----------
    raw_text:
        Raw content of the downloaded daily index file.
    index_name:
        Name to assign to the resulting Series.

    Returns
    -------
    pd.Series
        Daily time series, UTC-indexed. Missing values are NaN.
    """
    df = pd.read_csv(
        io.StringIO(raw_text),
        sep=r"\s+",
        header=None,
        na_values=["-99.9", "-99.90", "999.0", "999.000", "-999.0"],
    )

    n_cols = df.shape[1]

    if n_cols == 4:
        # Format: year month day value
        df.columns = ["year", "month", "day", index_name]
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    elif n_cols == 3:
        # Format: year day_of_year value
        df.columns = ["year", "doy", index_name]
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + df["doy"].astype(str).str.zfill(3),
            format="%Y%j",
        )
    else:
        raise ValueError(
            f"Unexpected number of columns ({n_cols}) in daily CPC file for {index_name}. "
            f"Expected 3 or 4."
        )

    series = (
        df.dropna(subset=["date", index_name])
          .set_index("date")[index_name]
          .sort_index()
    )
    series.index = pd.DatetimeIndex(series.index).tz_localize("UTC")
    return series


def parse_enso_monthly(raw_text: str) -> pd.DataFrame:
    """
    Parses the NOAA CPC ENSO index file, which contains multiple SST indices
    in a fixed-width format. The Nino 3.4 region SST anomaly is the standard
    ENSO indicator used in seasonal and subseasonal weather forecasting.

    A positive Nino 3.4 anomaly (El Nino) is associated with a mild, wet
    European winter in the medium to long term; La Nina (negative) tends towards
    colder, more blocked winters over NW Europe, particularly in December-January.

    The file format is:
        YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM

    Parameters
    ----------
    raw_text:
        Raw content of the NOAA ENSO index file.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame with columns for each Nino region and anomaly,
        UTC-indexed at the first of each month.
    """
    col_names = [
        "year", "month",
        "nino12", "nino12_anom",
        "nino3",  "nino3_anom",
        "nino4",  "nino4_anom",
        "nino34", "nino34_anom",
    ]

    df = pd.read_csv(
        io.StringIO(raw_text),
        sep=r"\s+",
        skiprows=1,
        names=col_names,
        na_values=["-99.9", "99.9", "-999.0"],
    )

    df = df.dropna(subset=["year", "month"])
    df["date"] = pd.to_datetime(
        df[["year", "month"]].assign(day=1).rename(columns={"month": "month"})
    )
    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")

    # Retain only the most useful columns for downstream modelling
    return df[["nino34", "nino34_anom"]]


def add_regime_flags(
    nao: pd.Series,
    ao: pd.Series,
) -> pd.DataFrame:
    """
    Derives categorical regime flags from the NAO and AO indices. These binary
    and ternary flags are used as features in the subseasonal regime classifier
    and as interpretable signals in the commodity linkage notebook.

    The combined_cold_regime flag is the most commercially useful single feature:
    it identifies periods where both the NAO and AO are simultaneously negative,
    which corresponds to the strongest cold-air-outbreak risk over NW Europe.

    Parameters
    ----------
    nao:
        Daily or monthly NAO index Series, UTC-indexed.
    ao:
        Daily or monthly AO index Series, UTC-indexed (same frequency as nao).

    Returns
    -------
    pd.DataFrame
        DataFrame with the original indices plus derived regime columns.
    """
    df = pd.DataFrame({"nao": nao, "ao": ao}).dropna(how="all")

    df["nao_negative"]    = (df["nao"] < NAO_BLOCKING_THRESHOLD).astype(int)
    df["nao_positive"]    = (df["nao"] > NAO_POSITIVE_THRESHOLD).astype(int)
    df["ao_negative"]     = (df["ao"]  < AO_NEGATIVE_THRESHOLD).astype(int)

    # Ternary regime: cold (-1), neutral (0), warm (+1)
    df["nao_regime"] = 0
    df.loc[df["nao"] < NAO_BLOCKING_THRESHOLD, "nao_regime"] = -1
    df.loc[df["nao"] > NAO_POSITIVE_THRESHOLD, "nao_regime"] =  1

    # Combined cold-air-outbreak signal requires both indices negative
    df["combined_cold_regime"] = (
        (df["nao"] < NAO_BLOCKING_THRESHOLD) & (df["ao"] < AO_NEGATIVE_THRESHOLD)
    ).astype(int)

    return df


def compute_rolling_features(
    nao: pd.Series,
    ao: pd.Series,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Computes rolling mean and standard deviation of the NAO and AO indices
    over multiple windows. Rolling features capture regime persistence, which
    is the key subseasonal predictability mechanism: a blocking pattern that
    has been in place for 10 days is more likely to persist than one that
    started yesterday.

    Parameters
    ----------
    nao:
        Daily NAO index Series, UTC-indexed.
    ao:
        Daily AO index Series, UTC-indexed.
    windows:
        Rolling window lengths in days. Defaults to [5, 10, 20, 30].

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling mean and std columns for each window,
        plus the trend (short window minus long window).
    """
    if windows is None:
        windows = [5, 10, 20, 30]

    frames = []
    for series, name in [(nao, "nao"), (ao, "ao")]:
        for w in windows:
            rolling = series.rolling(w, min_periods=max(1, w // 2))
            frames.append(rolling.mean().rename(f"{name}_{w}d_mean"))
            frames.append(rolling.std().rename(f"{name}_{w}d_std"))

        # Trend: difference between short-term (5d) and medium-term (20d) mean.
        # A strongly negative trend means the index is falling quickly toward
        # a blocking state, which has greater forecast value than a stable negative.
        short_mean = series.rolling(5,  min_periods=3).mean()
        long_mean  = series.rolling(20, min_periods=10).mean()
        frames.append((short_mean - long_mean).rename(f"{name}_trend_5d_vs_20d"))

    return pd.concat(frames, axis=1)


def fetch_index(
    index_key: str,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Downloads and parses a single CPC index by key. Optionally caches the raw
    text file to avoid repeated downloads when iterating during development.

    Parameters
    ----------
    index_key:
        One of the keys in CPC_URLS, e.g. 'nao_daily' or 'enso_monthly'.
    cache_dir:
        Directory to cache raw text files. Caching is disabled if None.
    force_refresh:
        If True, re-downloads the file even if a cached version exists.

    Returns
    -------
    pd.Series or pd.DataFrame
        Parsed index as a time-indexed Series or DataFrame, UTC-indexed.

    Raises
    ------
    KeyError
        If index_key is not found in CPC_URLS.
    ValueError
        If the raw file cannot be parsed with the expected format.
    """
    if index_key not in CPC_URLS:
        raise KeyError(
            f"Unknown index key '{index_key}'. Available keys: {list(CPC_URLS.keys())}"
        )

    url = CPC_URLS[index_key]
    raw_text: Optional[str] = None

    if cache_dir is not None:
        cache_dir  = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{index_key}.txt"

        if cache_path.exists() and not force_refresh:
            logger.info("Loading cached index from %s.", cache_path)
            raw_text = cache_path.read_text(encoding="utf-8")

    if raw_text is None:
        raw_text = _download_text(url)
        if cache_dir is not None:
            cache_path.write_text(raw_text, encoding="utf-8")
            logger.info("Cached raw text to %s.", cache_path)

    if "monthly" in index_key and index_key != "enso_monthly":
        name = index_key.split("_")[0].upper()
        return parse_cpc_monthly_wide(raw_text, name)

    elif "daily" in index_key:
        name = index_key.split("_")[0].upper()
        return parse_cpc_daily_three_column(raw_text, name)

    elif index_key == "enso_monthly":
        return parse_enso_monthly(raw_text)

    else:
        raise ValueError(f"No parser registered for index key '{index_key}'.")


def build_teleconnection_dataset(
    output_path: Path,
    cache_dir: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Downloads and assembles all CPC teleconnection indices into a single
    DataFrame. Daily indices (NAO, AO, PNA) and monthly ENSO are aligned on
    a daily frequency; monthly ENSO is forward-filled within each calendar
    month since the monthly value is valid for the entire month.

    This is the primary output of this module and feeds directly into the
    subseasonal regime classifier and the feature engineering pipeline.

    Parameters
    ----------
    output_path:
        File path at which to save the assembled Parquet file.
    cache_dir:
        Directory for raw text caching. Passed to fetch_index.
    start_date:
        Optional ISO date string to trim the output, e.g. '2015-01-01'.
    end_date:
        Optional ISO date string to trim the output, e.g. '2023-12-31'.

    Returns
    -------
    pd.DataFrame
        Wide daily DataFrame with columns:
        NAO, AO, PNA (daily), nino34, nino34_anom (monthly forward-filled),
        plus all regime flags and rolling features derived from NAO and AO.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    daily_keys   = ["nao_daily", "ao_daily", "pna_daily"]
    monthly_keys = ["enso_monthly"]

    daily_series = []
    for key in daily_keys:
        logger.info("Fetching %s.", key)
        series = fetch_index(key, cache_dir=cache_dir)
        daily_series.append(series)

    daily_df = pd.concat(daily_series, axis=1)

    enso = fetch_index("enso_monthly", cache_dir=cache_dir)

    # Resample ENSO monthly to daily and forward-fill within each month.
    # The monthly value represents the state for the full month.
    enso_daily = enso.resample("1D").ffill()
    enso_daily = enso_daily.reindex(daily_df.index, method="ffill")

    combined = pd.concat([daily_df, enso_daily], axis=1)

    nao = combined.get("NAO", pd.Series(dtype=float))
    ao  = combined.get("AO",  pd.Series(dtype=float))

    if not nao.empty and not ao.empty:
        regime_flags   = add_regime_flags(nao, ao)
        rolling_feats  = compute_rolling_features(nao, ao)
        combined       = pd.concat([combined, regime_flags.drop(columns=["nao", "ao"]),
                                    rolling_feats], axis=1)

    # Trim to requested date range if specified
    if start_date is not None:
        combined = combined.loc[pd.Timestamp(start_date, tz="UTC"):]
    if end_date is not None:
        combined = combined.loc[:pd.Timestamp(end_date, tz="UTC")]

    combined.to_parquet(output_path)
    logger.info(
        "Saved teleconnection dataset to %s (%d rows, %d columns).",
        output_path, len(combined), len(combined.columns),
    )
    return combined


def run_cpc_pipeline(
    output_dir: Path = Path("data/processed"),
    cache_dir: Optional[Path] = Path("data/raw/cpc_cache"),
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Top-level entry point for the CPC teleconnection pipeline. Downloads all
    indices, assembles the full feature DataFrame, and saves to Parquet.

    Parameters
    ----------
    output_dir:
        Directory in which to save the output Parquet file.
    cache_dir:
        Directory for caching raw text downloads from NOAA.
    start_date:
        Start of the date range to include in the output.
    end_date:
        End of the date range to include in the output.
    force_refresh:
        If True, ignores cached files and re-downloads everything from NOAA.

    Returns
    -------
    pd.DataFrame
        Assembled teleconnection feature DataFrame, also saved to disk.
    """
    output_path = Path(output_dir) / "cpc_teleconnections.parquet"

    return build_teleconnection_dataset(
        output_path=output_path,
        cache_dir=cache_dir,
        start_date=start_date,
        end_date=end_date,
    )
