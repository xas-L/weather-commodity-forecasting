"""
src/evaluation

Evaluation layer for the weather-commodity forecasting project. Provides all
diagnostic functions, metric computations, and plotting utilities required to
assess deterministic and probabilistic forecast quality, and to link forecast
errors to commercial outcomes in the commodity linkage notebook.

The evaluation layer is deliberately separated from the model layer. Models
in src/models/ produce outputs; this layer measures how good those outputs
are and frames those measurements in terms that are legible to both a
meteorological reviewer and a non-meteorologist trading PM.

Modules:
    deterministic   MAE, RMSE, skill scores, residual analysis, seasonal
                    error breakdowns, regime-conditional error stratification,
                    HDD error vs demand correlation, and model comparison tables.
    probabilistic   CRPS, CRPSS, PIT histograms, interval coverage, spread-skill
                    diagrams, reliability diagrams for NGBoost and the regime
                    classifier, rolling CRPS over time, and cold probability
                    vs outcome visualisation.

Typical usage in notebook 04:

    from src.evaluation import (
        compute_metrics,
        build_model_comparison_table,
        compute_interval_coverage,
        plot_reliability_diagram,
        plot_pit_histogram,
    )

Typical usage in notebook 05:

    from src.evaluation import (
        compute_hdd_error_demand_correlation,
        plot_hdd_error_vs_demand,
        plot_cold_probability_vs_outcome,
        summarise_cold_event_errors,
    )
"""

from src.evaluation.deterministic import (
    compute_metrics,
    compute_residuals,
    compute_mae_by_month,
    compute_mae_by_season,
    compute_error_by_regime,
    compute_hdd_error_demand_correlation,
    build_model_comparison_table,
    plot_predictions_vs_actual,
    plot_residuals_over_time,
    plot_residual_distribution,
    plot_mae_by_month,
    plot_skill_score_by_fold,
    plot_hdd_error_vs_demand,
    summarise_cold_event_errors,
)

from src.evaluation.probabilistic import (
    compute_pit_values,
    compute_interval_coverage,
    compute_spread_skill,
    compute_conditional_crps,
    compute_regime_classifier_reliability,
    compute_sigma_by_season,
    build_probabilistic_summary_table,
    plot_reliability_diagram,
    plot_regime_reliability_diagram,
    plot_pit_histogram,
    plot_spread_skill,
    plot_prediction_intervals,
    plot_crps_over_time,
    plot_cold_probability_vs_outcome,
)

__all__ = [
    # deterministic
    "compute_metrics",
    "compute_residuals",
    "compute_mae_by_month",
    "compute_mae_by_season",
    "compute_error_by_regime",
    "compute_hdd_error_demand_correlation",
    "build_model_comparison_table",
    "plot_predictions_vs_actual",
    "plot_residuals_over_time",
    "plot_residual_distribution",
    "plot_mae_by_month",
    "plot_skill_score_by_fold",
    "plot_hdd_error_vs_demand",
    "summarise_cold_event_errors",
    # probabilistic
    "compute_pit_values",
    "compute_interval_coverage",
    "compute_spread_skill",
    "compute_conditional_crps",
    "compute_regime_classifier_reliability",
    "compute_sigma_by_season",
    "build_probabilistic_summary_table",
    "plot_reliability_diagram",
    "plot_regime_reliability_diagram",
    "plot_pit_histogram",
    "plot_spread_skill",
    "plot_prediction_intervals",
    "plot_crps_over_time",
    "plot_cold_probability_vs_outcome",
]
