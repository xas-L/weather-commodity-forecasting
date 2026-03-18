"""
src/data

Data ingestion and persistence layer for the weather-commodity forecasting.
Each module handles a single external data source and exposes a top-level
run_*_pipeline entry point that is idempotent on repeated calls.

Modules:
    era5_pipeline    ERA5 reanalysis from ECMWF via the Copernicus CDS API.
    entso_pipeline   Power system data from the ENTSO-E Transparency Platform.
    cpc_indices      Teleconnection indices (NAO, AO, ENSO) from NOAA CPC.
"""

from src.data.era5_pipeline import run_era5_pipeline
from src.data.entso_pipeline import run_entso_pipeline
from src.data.cpc_indices import run_cpc_pipeline

__all__ = [
    "run_era5_pipeline",
    "run_entso_pipeline",
    "run_cpc_pipeline",
]
