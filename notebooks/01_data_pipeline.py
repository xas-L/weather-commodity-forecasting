# %% [markdown]
# # Notebook 01: Data Pipeline and Quality Audit
#
# This notebook runs the three data ingestion pipelines (ERA5, ENTSO-E, CPC)
# and performs a structured quality audit on every raw series before any
# feature engineering takes place.
#
# Data quality is the foundation of everything that follows. A poor imputation
# decision made silently here will corrupt every model downstream and produce
# results that look plausible but are wrong. Every missing-value decision is
# made explicitly and documented in the audit table at the end of this notebook.
#
# **Outputs written to disk**
#
# - `data/raw/era5_surface.nc` and `data/raw/era5_pressure.nc`
# - `data/processed/era5_surface_{location}.parquet` for each market location
# - `data/processed/era5_regime_fields.parquet`
# - `data/raw/entso_{country}_{type}.parquet` for each country and data type
# - `data/processed/cpc_teleconnections.parquet`
# - `data/processed/audit_summary.parquet`

# %% [markdown]
# ## 1. Imports and configuration

# %%
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import run_era5_pipeline, run_entso_pipeline, run_cpc_pipeline
from src.data.era5_pipeline import MARKET_LOCATIONS
from src.data.entso_pipeline import COUNTRY_CODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths. All pipelines use these directories as defaults.
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Data window. 2018 gives five full winters before the 2023 end date,
# which is sufficient for walk-forward CV with an 8-fold scheme.
START_DATE = "2018-01-01"
END_DATE   = "2023-12-31"
YEARS      = list(range(2018, 2024))

print(f"Data window: {START_DATE} to {END_DATE}")
print(f"Market locations: {list(MARKET_LOCATIONS.keys())}")
print(f"ENTSO-E countries: {list(COUNTRY_CODES.keys())}")

# %% [markdown]
# ## 2. ERA5 reanalysis pipeline
#
# Downloads surface and pressure-level fields for the European domain and
# extracts point time series for each market location. CDS API downloads
# are asynchronous and may take several hours on first run. Subsequent
# runs skip files that already exist on disk.

# %%
run_era5_pipeline(
    raw_dir=RAW_DIR,
    processed_dir=PROCESSED_DIR,
    years=YEARS,
)

# %% [markdown]
# ### 2.1 Inspect extracted surface series

# %%
era5_frames = {}
for location in MARKET_LOCATIONS:
    path = PROCESSED_DIR / f"era5_surface_{location}.parquet"
    if path.exists():
        era5_frames[location] = pd.read_parquet(path)
        logger.info(
            "Loaded %s: %d rows, index %s to %s.",
            location,
            len(era5_frames[location]),
            era5_frames[location].index.min().date(),
            era5_frames[location].index.max().date(),
        )
    else:
        logger.warning("Surface file missing for %s. Re-run the ERA5 pipeline.", location)

# %%
# Display a summary table: row count, missing values, and date range per location.
audit_rows = []
for loc, df in era5_frames.items():
    audit_rows.append({
        "location":    loc,
        "n_rows":      len(df),
        "start":       str(df.index.min().date()),
        "end":         str(df.index.max().date()),
        "pct_missing": round(df.isnull().mean().mean() * 100, 3),
        "t2m_min":     round(df["t2m_celsius"].min(), 2) if "t2m_celsius" in df.columns else None,
        "t2m_max":     round(df["t2m_celsius"].max(), 2) if "t2m_celsius" in df.columns else None,
    })

era5_audit = pd.DataFrame(audit_rows).set_index("location")
print(era5_audit.to_string())

# %%
# Plot daily mean T2m for Amsterdam and Berlin to confirm the seasonal cycle
# is physically reasonable before proceeding to feature engineering.
if "amsterdam" in era5_frames and "berlin" in era5_frames:
    ams_daily = era5_frames["amsterdam"]["t2m_celsius"].resample("1D").mean()
    ber_daily = era5_frames["berlin"]["t2m_celsius"].resample("1D").mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ams_daily.index, ams_daily.values, linewidth=0.8,
            alpha=0.85, label="Amsterdam (TTF region)")
    ax.plot(ber_daily.index, ber_daily.values, linewidth=0.8,
            alpha=0.75, label="Berlin")
    ax.set_title("Daily mean 2m temperature: Amsterdam and Berlin")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ### 2.2 Inspect regime fields (Z500 NAO proxy and Greenland blocking index)

# %%
regime_path = PROCESSED_DIR / "era5_regime_fields.parquet"
if regime_path.exists():
    regime_fields = pd.read_parquet(regime_path)
    print(f"Regime fields shape: {regime_fields.shape}")
    print(regime_fields.describe().round(3).to_string())
else:
    logger.warning("Regime fields file not found. Re-run the ERA5 pipeline.")
    regime_fields = pd.DataFrame()

# %%
if not regime_fields.empty and "nao_z500_proxy" in regime_fields.columns:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(regime_fields.index, regime_fields["nao_z500_proxy"],
                 linewidth=0.7, colour="#264478")
    axes[0].axhline(0, colour="black", linewidth=0.6, linestyle="--", alpha=0.5)
    axes[0].axhline(-1.5, colour="#c00000", linewidth=0.6, linestyle=":",
                    alpha=0.6, label="Blocking threshold (-1.5 SD)")
    axes[0].set_ylabel("NAO Z500 proxy (SD)")
    axes[0].set_title("Z500-based NAO index: negative values indicate blocking risk")
    axes[0].legend(fontsize=8)

    if "greenland_blocking_index" in regime_fields.columns:
        axes[1].fill_between(
            regime_fields.index,
            0,
            regime_fields["greenland_blocking_index"].clip(lower=0),
            alpha=0.5, colour="#c00000", label="Positive (blocking)",
        )
        axes[1].fill_between(
            regime_fields.index,
            regime_fields["greenland_blocking_index"].clip(upper=0),
            0,
            alpha=0.4, colour="#264478", label="Negative",
        )
        axes[1].axhline(0, colour="black", linewidth=0.6)
        axes[1].set_ylabel("Greenland blocking index (m)")
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 3. ENTSO-E power market pipeline
#
# Fetches actual load, wind generation, solar generation, and day-ahead prices
# for Germany and the Netherlands. Rate-limit backoff is built into the pipeline.

# %%
ENTSO_KEY = os.environ.get("ENTSO_API_KEY", "")
if not ENTSO_KEY:
    logger.warning(
        "ENTSO_API_KEY environment variable not set. "
        "Set it with: export ENTSO_API_KEY='your-key'"
    )

if ENTSO_KEY:
    run_entso_pipeline(
        api_key=ENTSO_KEY,
        output_dir=RAW_DIR,
        countries=["germany", "netherlands"],
        start_date=START_DATE,
        end_date=END_DATE,
    )

# %% [markdown]
# ### 3.1 Inspect German load and wind generation

# %%
de_load_path = RAW_DIR / "entso_germany_load.parquet"
de_wind_path = RAW_DIR / "entso_germany_wind.parquet"

de_load = pd.read_parquet(de_load_path) if de_load_path.exists() else pd.DataFrame()
de_wind = pd.read_parquet(de_wind_path) if de_wind_path.exists() else pd.DataFrame()

if not de_load.empty:
    print(f"German load: {len(de_load)} hourly rows")
    print(f"  Mean: {de_load.iloc[:, 0].mean():.0f} MW")
    print(f"  Min:  {de_load.iloc[:, 0].min():.0f} MW")
    print(f"  Max:  {de_load.iloc[:, 0].max():.0f} MW")
    missing_pct = de_load.isnull().mean().iloc[0] * 100
    print(f"  Missing: {missing_pct:.2f}%")

if not de_wind.empty:
    print(f"\nGerman wind generation: {len(de_wind)} hourly rows")
    print(de_wind.describe().round(0).to_string())

# %%
if not de_load.empty and not de_wind.empty:
    load_daily = de_load.iloc[:, 0].resample("1D").mean()
    wind_daily = de_wind["wind_total_mw"].resample("1D").mean() if "wind_total_mw" in de_wind.columns else pd.Series()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(load_daily.index, load_daily.values,
                 linewidth=0.7, colour="#264478")
    axes[0].set_ylabel("Daily mean load (MW)")
    axes[0].set_title("German electricity demand")
    axes[0].grid(axis="y", linewidth=0.4, alpha=0.4)

    if not wind_daily.empty:
        axes[1].fill_between(wind_daily.index, 0, wind_daily.values,
                             alpha=0.55, colour="#70ad47")
        axes[1].set_ylabel("Daily mean wind generation (MW)")
        axes[1].set_title("German wind generation")
        axes[1].grid(axis="y", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. NOAA CPC teleconnection pipeline

# %%
run_cpc_pipeline(
    output_dir=PROCESSED_DIR,
    cache_dir=RAW_DIR / "cpc_cache",
    start_date=START_DATE,
    end_date=END_DATE,
)

# %%
cpc_path = PROCESSED_DIR / "cpc_teleconnections.parquet"
if cpc_path.exists():
    cpc_df = pd.read_parquet(cpc_path)
    print(f"CPC teleconnections: {cpc_df.shape[0]} rows, {cpc_df.shape[1]} columns")
    print(f"Columns: {list(cpc_df.columns)}")
    print(f"\nDescriptive statistics for raw indices:")
    raw_indices = [c for c in ["NAO", "AO", "PNA"] if c in cpc_df.columns]
    print(cpc_df[raw_indices].describe().round(3).to_string())
else:
    logger.warning("CPC file not found.")
    cpc_df = pd.DataFrame()

# %%
if not cpc_df.empty and "NAO" in cpc_df.columns and "AO" in cpc_df.columns:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for ax, col, colour, label in [
        (axes[0], "NAO", "#264478", "NAO daily index"),
        (axes[1], "AO",  "#ed7d31", "AO daily index"),
    ]:
        ax.plot(cpc_df.index, cpc_df[col].values, linewidth=0.7, colour=colour)
        ax.axhline(0,    colour="black",   linewidth=0.6, linestyle="--", alpha=0.5)
        ax.axhline(-1.5, colour="#c00000", linewidth=0.6, linestyle=":",  alpha=0.6,
                   label="Blocking threshold")
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Combined data quality audit
#
# Aggregates missing value statistics and date range coverage across all
# data sources into a single audit table. Any series with more than 5%
# missing values requires an explicit imputation decision before it enters
# the feature engineering step.

# %%
def _audit_series(name: str, series: pd.Series) -> dict:
    """Computes summary statistics for a single data series."""
    return {
        "series":      name,
        "n_rows":      len(series),
        "start":       str(series.dropna().index.min().date()) if not series.dropna().empty else "N/A",
        "end":         str(series.dropna().index.max().date()) if not series.dropna().empty else "N/A",
        "pct_missing": round(series.isnull().mean() * 100, 3),
        "mean":        round(float(series.mean()), 3) if not series.empty else None,
        "std":         round(float(series.std()),  3) if not series.empty else None,
    }


audit_records = []

if "amsterdam" in era5_frames:
    df = era5_frames["amsterdam"]
    for col in df.columns:
        audit_records.append(_audit_series(f"ERA5 amsterdam / {col}", df[col]))

if not de_load.empty:
    audit_records.append(_audit_series("ENTSO-E DE load", de_load.iloc[:, 0]))

if not de_wind.empty and "wind_total_mw" in de_wind.columns:
    audit_records.append(_audit_series("ENTSO-E DE wind total", de_wind["wind_total_mw"]))

if not cpc_df.empty:
    for col in ["NAO", "AO", "PNA"]:
        if col in cpc_df.columns:
            audit_records.append(_audit_series(f"CPC {col}", cpc_df[col]))

audit_df = pd.DataFrame(audit_records)

# Flag series with more than 5% missing values for manual review.
audit_df["needs_review"] = audit_df["pct_missing"] > 5.0

print("Data quality audit:")
print(audit_df.to_string(index=False))

# %%
# Persist the audit table for reference in subsequent notebooks.
audit_path = PROCESSED_DIR / "audit_summary.parquet"
audit_df.to_parquet(audit_path)
print(f"\nAudit summary saved to {audit_path}")

flagged = audit_df[audit_df["needs_review"]]
if flagged.empty:
    print("All series pass the 5% missing threshold. Proceeding to EDA.")
else:
    print(f"\nWARNING: {len(flagged)} series exceed 5% missing:")
    print(flagged[["series", "pct_missing"]].to_string(index=False))

# %% [markdown]
# ## 6. Next steps
#
# All raw and processed files are written. The next notebook
# (`02_eda_atmospheric.ipynb`) loads the processed ERA5 and CPC data to
# perform exploratory analysis of the atmospheric features and establish
# the physical narrative that motivates the model design.
