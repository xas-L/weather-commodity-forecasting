# %% [markdown]
# # Notebook 03: Feature Engineering
#
# This notebook assembles the full feature matrix used by both the short-term
# and subseasonal models. Features from all four engineering modules (degree days,
# atmospheric, teleconnections, Dunkelflaute) are built, aligned to a common
# UTC daily index, and joined into a single wide Parquet file.
#
# **Outputs written to disk**
#
# - `data/processed/features_shortterm.parquet`
#   Daily feature matrix for the short-term LightGBM and NGBoost models.
# - `data/processed/features_subseasonal.parquet`
#   Weekly feature matrix for the XGBoost regime classifier.
# - `data/processed/targets_shortterm.parquet`
#   Daily target (T2m and HDD) series for the short-term models.
# - `data/processed/targets_subseasonal.parquet`
#   Weekly regime labels for the subseasonal classifier at 2-week lead.

# %% [markdown]
# ## 1. Imports and data loading

# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features import (
    build_degree_day_features,
    build_atmospheric_features,
    build_teleconnection_features,
    build_dunkelflaute_features,
)
from src.features.teleconnections import build_subseasonal_target
import xarray as xr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

PROCESSED_DIR = Path("data/processed")
RAW_DIR       = Path("data/raw")

# Load pre-processed inputs from notebook 01.
ams     = pd.read_parquet(PROCESSED_DIR / "era5_surface_amsterdam.parquet")
cpc     = pd.read_parquet(PROCESSED_DIR / "cpc_teleconnections.parquet")

de_wind_path  = RAW_DIR / "entso_germany_wind.parquet"
de_solar_path = RAW_DIR / "entso_germany_solar.parquet"
de_wind  = pd.read_parquet(de_wind_path)  if de_wind_path.exists()  else pd.DataFrame()
de_solar = pd.read_parquet(de_solar_path) if de_solar_path.exists() else pd.DataFrame()

print(f"Amsterdam ERA5: {len(ams)} rows")
print(f"CPC indices: {len(cpc)} rows")

# %% [markdown]
# ## 2. Degree-day features

# %%
dd_features = build_degree_day_features(ams["t2m_celsius"])
print(f"Degree-day features: {dd_features.shape}")

# %% [markdown]
# ## 3. Atmospheric features from ERA5 pressure levels

# %%
pressure_nc = RAW_DIR / "era5_pressure.nc"
if pressure_nc.exists():
    ds_pressure = xr.open_dataset(pressure_nc, chunks={"time": 200})
    atmo_features = build_atmospheric_features(ds_pressure)
    ds_pressure.close()
    print(f"Atmospheric features: {atmo_features.shape}")
else:
    # Fall back to the pre-computed regime fields saved by era5_pipeline.py.
    logger.warning(
        "Pressure-level NetCDF not found. Loading pre-computed regime fields."
    )
    regime_path   = PROCESSED_DIR / "era5_regime_fields.parquet"
    atmo_features = pd.read_parquet(regime_path) if regime_path.exists() else pd.DataFrame()
    print(f"Loaded regime fields as atmospheric features: {atmo_features.shape}")

# %% [markdown]
# ## 4. Teleconnection features

# %%
# Extract the raw daily indices from the CPC file, which already contains
# regime flags and rolling features computed in cpc_indices.py.
nao = cpc["NAO"] if "NAO" in cpc.columns else pd.Series(dtype=float)
ao  = cpc["AO"]  if "AO"  in cpc.columns else pd.Series(dtype=float)
pna = cpc["PNA"] if "PNA" in cpc.columns else None
nino34_anom = cpc["nino34_anom"] if "nino34_anom" in cpc.columns else None

tele_features = build_teleconnection_features(
    nao=nao,
    ao=ao,
    pna=pna,
    nino34_anom=nino34_anom,
)
print(f"Teleconnection features: {tele_features.shape}")

# %% [markdown]
# ## 5. Dunkelflaute features

# %%
INSTALLED_WIND_DE_MW  = 63_000.0
INSTALLED_SOLAR_DE_MW = 59_000.0

if not de_wind.empty:
    wind_gen  = de_wind["wind_total_mw"] if "wind_total_mw" in de_wind.columns \
                else de_wind.iloc[:, 0]
    solar_gen = de_solar.iloc[:, 0] if not de_solar.empty else None

    dunk_features, events = build_dunkelflaute_features(
        wind_generation_mw=wind_gen,
        installed_wind_mw=INSTALLED_WIND_DE_MW,
        solar_generation_mw=solar_gen,
        installed_solar_mw=INSTALLED_SOLAR_DE_MW if solar_gen is not None else None,
    )
    # Resample to daily for the short-term feature matrix.
    dunk_daily = dunk_features.resample("1D").agg({
        "wind_cf":                    "mean",
        "is_dunkelflaute_event_hour": "max",
        "wind_deficit_24h_mean":      "mean",
        "combined_deficit_24h_mean":  "mean",
        "dunkelflaute_prob_48h_forward": "max",
    }).rename(columns={"is_dunkelflaute_event_hour": "dunkelflaute_day"})
    print(f"Dunkelflaute daily features: {dunk_daily.shape}")
else:
    logger.warning("Wind data not available. Dunkelflaute features will be absent.")
    dunk_daily = pd.DataFrame()

# %% [markdown]
# ## 6. Assembling the short-term feature matrix

# %%
# Resample all features to a common daily UTC index. Atmospheric features from
# ERA5 are already daily. Teleconnection features are already daily. Degree-day
# features are already daily.

frames_to_join = [
    dd_features.add_prefix("dd_").drop(columns=[
        c for c in dd_features.add_prefix("dd_").columns
        if "HDD" not in c and "CDD" not in c and "gas_year" not in c
    ], errors="ignore"),
    dd_features[["HDD", "CDD", "HDD_anom", "HDD_7d", "HDD_anom_7d",
                 "gas_year_position", "gas_year_sin", "gas_year_cos"]],
]

if not atmo_features.empty:
    frames_to_join.append(atmo_features)

if not tele_features.empty:
    # For the short-term model, keep only the most predictive teleconnection
    # features to limit the number of near-zero-importance columns.
    shortterm_tele_cols = [c for c in tele_features.columns if any(
        kw in c for kw in ["nao", "ao", "blocking", "combined_cold", "pna"]
    )]
    frames_to_join.append(tele_features[shortterm_tele_cols])

if not dunk_daily.empty:
    frames_to_join.append(dunk_daily)

# Concatenate on columns, align on the date index.
features_shortterm = pd.concat(frames_to_join, axis=1)

# Remove duplicate columns that may appear if multiple frames share names.
features_shortterm = features_shortterm.loc[:, ~features_shortterm.columns.duplicated()]

# Drop rows with excessive missing values (more than 30% of columns missing).
row_missing_frac = features_shortterm.isnull().mean(axis=1)
features_shortterm = features_shortterm[row_missing_frac <= 0.30]

print(f"Short-term feature matrix: {features_shortterm.shape}")
print(f"Date range: {features_shortterm.index.min().date()} to {features_shortterm.index.max().date()}")
print(f"Columns with any NaN: {features_shortterm.isnull().any().sum()}")

# %% [markdown]
# ## 7. Assembling the subseasonal feature matrix

# %%
# The subseasonal model uses weekly aggregations of all features.
# Teleconnection features dominate at this horizon; atmospheric and
# degree-day features are included as context but are secondary.

if not tele_features.empty:
    tele_weekly = tele_features.resample("1W").mean()

    atmo_weekly = atmo_features.resample("1W").mean() if not atmo_features.empty \
                  else pd.DataFrame()

    dd_weekly = dd_features[["HDD_7d", "HDD_anom_7d"]].resample("1W").mean()

    sub_frames = [tele_weekly]
    if not atmo_weekly.empty:
        sub_frames.append(atmo_weekly)
    sub_frames.append(dd_weekly)

    features_subseasonal = pd.concat(sub_frames, axis=1)
    features_subseasonal = features_subseasonal.loc[
        :, ~features_subseasonal.columns.duplicated()
    ]
    print(f"Subseasonal feature matrix: {features_subseasonal.shape}")
else:
    logger.warning("Teleconnection features not available. Subseasonal matrix will be empty.")
    features_subseasonal = pd.DataFrame()

# %% [markdown]
# ## 8. Target variable construction

# %%
# Short-term target: daily mean 2m temperature in °C.
t2m_daily  = ams["t2m_celsius"].resample("1D").mean().rename("t2m_celsius")
hdd_daily  = dd_features["HDD"].rename("HDD")

target_shortterm = pd.concat([t2m_daily, hdd_daily], axis=1)
target_shortterm = target_shortterm.reindex(features_shortterm.index)

print(f"Short-term targets: {target_shortterm.shape}")
print(target_shortterm.describe().round(3).to_string())

# %%
# Subseasonal target: ternary regime label at week-2 lead.
if not features_subseasonal.empty:
    t2m_weekly = t2m_daily.resample("1W").mean()
    regime_target = build_subseasonal_target(t2m_weekly, lead_weeks=2)
    regime_target = regime_target.reindex(features_subseasonal.index)

    cold_pct    = (regime_target == "cold").mean() * 100
    neutral_pct = (regime_target == "neutral").mean() * 100
    warm_pct    = (regime_target == "warm").mean() * 100
    print(f"\nSubseasonal target class distribution (week-2 lead):")
    print(f"  Cold:    {cold_pct:.1f}%")
    print(f"  Neutral: {neutral_pct:.1f}%")
    print(f"  Warm:    {warm_pct:.1f}%")
else:
    regime_target = pd.Series(dtype=str)

# %% [markdown]
# ## 9. Feature inspection and top correlations

# %%
# Pearson correlations with the short-term T2m target.
# The top features should be physically interpretable.
if not features_shortterm.empty:
    target_aligned = t2m_daily.reindex(features_shortterm.index)
    correlations   = features_shortterm.corrwith(target_aligned).dropna().abs().sort_values(ascending=False)

    print("Top 20 features by absolute correlation with daily T2m:")
    print(correlations.head(20).round(4).to_string())

# %%
# Plot the top-10 most correlated features as a horizontal bar chart.
if not features_shortterm.empty:
    top10 = correlations.head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top10.index[::-1], top10.values[::-1], colour="#264478", alpha=0.85)
    ax.set_xlabel("Absolute Pearson correlation with T2m")
    ax.set_title("Top 10 features by correlation with daily mean T2m (Amsterdam)")
    ax.grid(axis="x", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    plt.show()

# %%
# Missing value profile of the final short-term feature matrix.
missing_profile = (
    features_shortterm.isnull().mean() * 100
).sort_values(ascending=False)

if missing_profile[missing_profile > 0].empty:
    print("No missing values in the short-term feature matrix.")
else:
    print("Features with missing values:")
    print(missing_profile[missing_profile > 0].round(2).to_string())

# %% [markdown]
# ## 10. Persist feature matrices and targets

# %%
features_shortterm.to_parquet(PROCESSED_DIR / "features_shortterm.parquet")
target_shortterm.to_parquet(PROCESSED_DIR / "targets_shortterm.parquet")
print(f"Short-term feature matrix saved: {features_shortterm.shape}")
print(f"Short-term targets saved: {target_shortterm.shape}")

if not features_subseasonal.empty:
    features_subseasonal.to_parquet(PROCESSED_DIR / "features_subseasonal.parquet")
    print(f"Subseasonal feature matrix saved: {features_subseasonal.shape}")

if not regime_target.empty:
    regime_target.dropna().to_frame("regime_label").to_parquet(
        PROCESSED_DIR / "targets_subseasonal.parquet"
    )
    print("Subseasonal targets saved.")

# %% [markdown]
# ## 11. Next steps
#
# The feature matrices are ready for model training. Notebook 04 loads these
# files and runs walk-forward cross-validation for all models against all
# baselines, producing the comparison table and SHAP attribution analysis.
