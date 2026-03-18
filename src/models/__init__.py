"""
Model training and inference layer.
Each module encapsulates a distinct modelling approach and exposes a consistent
interface: fit on training data, predict on test data, evaluate with appropriate
metrics for that model type.

Modules:
    baseline            Climatological, persistence, and NWP-direct baselines.
                        All other models are benchmarked against these.
    lgbm_forecaster     Short-range deterministic temperature and HDD forecasting
                        using LightGBM with walk-forward CV, Optuna tuning,
                        and SHAP-based feature attribution.
    ngboost_prob        Probabilistic temperature forecasting using NGBoost,
                        evaluated with CRPS, reliability, and sharpness metrics.
    regime_classifier   Subseasonal NAO regime classifier using XGBoost with
                        probability calibration, evaluated via Brier score and
                        Brier Skill Score.

Typical usage:

    from src.models.baseline import evaluate_baselines, compute_skill_score
    from src.models.lgbm_forecaster import run_walk_forward_cv, fit_final_model
    from src.models.ngboost_prob import fit_ngboost, predict_distribution, compute_crps
    from src.models.regime_classifier import RegimeClassifier, evaluate_regime_classifier
"""

from src.models.baseline import (
    ClimatologyBaseline,
    PersistenceBaseline,
    NWPDirectBaseline,
    compute_skill_score,
    evaluate_baselines,
)
from src.models.lgbm_forecaster import (
    run_walk_forward_cv,
    fit_final_model,
    compute_shap_values,
    build_shap_summary,
)
from src.models.ngboost_prob import (
    fit_ngboost,
    predict_distribution,
    compute_crps,
    compute_reliability,
    compute_sharpness,
)
from src.models.regime_classifier import (
    RegimeClassifier,
    evaluate_regime_classifier,
    run_regime_walk_forward_cv,
)

__all__ = [
    "ClimatologyBaseline",
    "PersistenceBaseline",
    "NWPDirectBaseline",
    "compute_skill_score",
    "evaluate_baselines",
    "run_walk_forward_cv",
    "fit_final_model",
    "compute_shap_values",
    "build_shap_summary",
    "fit_ngboost",
    "predict_distribution",
    "compute_crps",
    "compute_reliability",
    "compute_sharpness",
    "RegimeClassifier",
    "evaluate_regime_classifier",
    "run_regime_walk_forward_cv",
]
