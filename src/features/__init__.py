"""
Feature engineering layer for the weather-commodity forecasting project.
Each module constructs a coherent set of features from a single domain
and exposes a primary build_* entry point that returns a UTC-indexed DataFrame.

Modules:
    degree_days        HDD and CDD with climatological anomalies, rolling
                       accumulations, and seasonal position encoding.
    atmospheric        Upper-atmosphere synoptic features from ERA5 pressure-level
                       fields: Z500 anomaly, blocking indices, jet stream
                       position, wind shear, and persistence metrics.
    teleconnections    Subseasonal regime features from CPC teleconnection
                       indices: lagged values, tendencies, persistence flags,
                       and phase-coupling interactions between NAO, AO, and PNA.
    dunkelflaute       Low-wind and low-solar detection, event cataloguing,
                       forward probability construction, and renewable deficit
                       features for gas-for-power demand modelling.

Typical usage in the feature engineering notebook:

    from src.features import (
        build_degree_day_features,
        build_atmospheric_features,
        build_teleconnection_features,
        build_dunkelflaute_features,
    )
"""

from src.features.degree_days       import build_degree_day_features
from src.features.atmospheric       import build_atmospheric_features
from src.features.teleconnections   import build_teleconnection_features
from src.features.dunkelflaute      import build_dunkelflaute_features

__all__ = [
    "build_degree_day_features",
    "build_atmospheric_features",
    "build_teleconnection_features",
    "build_dunkelflaute_features",
]
