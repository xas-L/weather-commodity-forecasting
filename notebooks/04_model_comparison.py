# %% [markdown]
# # Notebook 04: Model Comparison
#
# This notebook trains all models, runs walk-forward cross-validation,
# and produces the full evaluation suite. Results are organised in the order
# a reviewer would read them: baselines first, then deterministic models,
# then probabilistic models, then the subseasonal classifier.
#
# Every figure and table answers a specific commercial question. The narrative
# thread is: "does this model add skill relative to the cheapest available
# alternative, and does that skill exist in the periods when it matters most?"
#
# **Outputs written to disk**
#
# - `data/outputs/wfcv_lgbm.parquet`: LightGBM walk-forward fold metrics
# - `data/outputs/wfcv_ngboost.parquet`: NGBoost walk-forward fold metrics
# - `data/outputs/wfcv_regime.parquet`: Regime classifier walk-forward fold metrics
# - `data/outputs/predictions_lgbm.parquet`: Out-of-sample LightGBM predictions
# - `data/outputs/forecasts_ngboost.parquet`: Out-of-sample NGBoost distributions
# - `models/lgbm_final.txt`: Fitted final LightGBM model
# - `reports/figures/`: All evaluation figures

# %% [markdown]
# ## 1. Imports and data loading

# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import (
    ClimatologyBaseline,
    PersistenceBaseline,
    NWPDirectBaseline,
    evaluate_baselines,
    run_walk_forward_cv,
    fit_final_model,
    compute_shap_values,
    build_shap_summary,
    fit_ngboost,
    predict_distribution,
    compute_crps,
    compute_reliability,
    compute_sharpness,
    RegimeClassifier,
    evaluate_regime_classifier,
    run_regime_walk_forward_cv,
)
from src.models.lgbm_forecaster import (
    tune_hyperparameters,
    scale_features,
    build_meteorological_attribution,
    WalkForwardResult,
)
from src.models.ngboost_prob import (
    compute_probabilistic_skill_score,
    run_probabilistic_walk_forward_cv,
    build_probabilistic_results_table,
)
from src.evaluation import (
    compute_metrics,
    compute_residuals,
    compute_mae_by_month,
    compute_mae_by_season,
    compute_error_by_regime,
    build_model_comparison_table,
    plot_predictions_vs_actual,
    plot_residuals_over_time,
    plot_residual_distribution,
    plot_mae_by_month,
    plot_skill_score_by_fold,
    compute_pit_values,
    compute_interval_coverage,
    compute_spread_skill,
    compute_regime_classifier_reliability,
    build_probabilistic_summary_table,
    plot_reliability_diagram,
    plot_regime_reliability_diagram,
    plot_pit_histogram,
    plot_spread_skill,
    plot_prediction_intervals,
    plot_crps_over_time,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR   = Path("data/outputs")
MODELS_DIR    = Path("models")
FIGURES_DIR   = Path("reports/figures")
for d in [OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Target column. All models predict T2m in degrees Celsius.
TARGET = "t2m_celsius"

# %%
features = pd.read_parquet(PROCESSED_DIR / "features_shortterm.parquet")
targets  = pd.read_parquet(PROCESSED_DIR / "targets_shortterm.parquet")

# Align and drop any rows where the target is missing.
df = features.join(targets[[TARGET]], how="inner").dropna(subset=[TARGET])
feature_cols = [c for c in df.columns if c != TARGET and c not in targets.columns]

print(f"Combined dataset: {df.shape[0]} rows, {len(feature_cols)} features")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

# Chronological 80/20 train/test split. The test set represents the most
# recent 20% of the data, which is the typical deployment scenario.
split_idx  = int(len(df) * 0.80)
df_train   = df.iloc[:split_idx]
df_test    = df.iloc[split_idx:]
y_train    = df_train[TARGET]
y_test     = df_test[TARGET]
X_train    = df_train[feature_cols]
X_test     = df_test[feature_cols]

print(f"Train: {df_train.index.min().date()} to {df_train.index.max().date()} ({len(df_train)} rows)")
print(f"Test:  {df_test.index.min().date()}  to {df_test.index.max().date()}  ({len(df_test)} rows)")

# %% [markdown]
# ## 2. Baseline evaluation
#
# Baselines must be established before looking at any ML results. The skill
# score of every subsequent model is defined relative to the climatological
# RMSE computed here.

# %%
baseline_results = evaluate_baselines(
    y_train=y_train,
    y_test=y_test,
    X_test=pd.DataFrame(index=y_test.index),
)
print("Baseline evaluation:")
print(baseline_results.to_string())

clim_model = ClimatologyBaseline()
clim_model.fit(pd.DataFrame(index=y_train.index), y_train)
clim_preds   = pd.Series(clim_model.predict(pd.DataFrame(index=y_test.index)),
                          index=y_test.index)
clim_rmse    = float(np.sqrt(((y_test - clim_preds) ** 2).mean()))
print(f"\nClimatological RMSE on test set: {clim_rmse:.4f}°C")
print("All model skill scores will be computed relative to this value.")

# %% [markdown]
# ## 3. LightGBM walk-forward cross-validation

# %%
print("Running LightGBM walk-forward CV (8 folds)...")
lgbm_cv_result = run_walk_forward_cv(
    df=df,
    target_col=TARGET,
    feature_cols=feature_cols,
    n_splits=8,
    initial_train_frac=0.50,
)

print(lgbm_cv_result)
print(lgbm_cv_result.summary().to_string(index=False))

# Persist fold metrics.
lgbm_cv_df = pd.DataFrame(lgbm_cv_result.fold_metrics)
lgbm_cv_df.to_parquet(OUTPUTS_DIR / "wfcv_lgbm.parquet")

# %%
# Skill score by fold plot.
fig = plot_skill_score_by_fold(
    {"LightGBM": lgbm_cv_result.fold_metrics},
    title="LightGBM skill score per walk-forward fold",
)
fig.savefig(FIGURES_DIR / "lgbm_skill_by_fold.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Optional: Optuna hyperparameter tuning
#
# Uncomment the block below to run Optuna tuning. This takes approximately
# 5-15 minutes on a standard laptop. The default parameters in lgbm_forecaster.py
# are already reasonable and tuning typically provides a marginal improvement.

# %%
# best_params = tune_hyperparameters(
#     df=df,
#     target_col=TARGET,
#     feature_cols=feature_cols,
#     n_trials=50,
#     timeout_secs=600,
# )
# print("Best params:", best_params)
best_params = None   # Use defaults for now.

# %% [markdown]
# ## 5. Final LightGBM model and SHAP analysis

# %%
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

lgbm_final = fit_final_model(
    X_train=X_train_sc,
    y_train=y_train,
    X_val=X_test_sc,
    y_val=y_test,
    params=best_params,
)

lgbm_preds = pd.Series(lgbm_final.predict(X_test_sc), index=y_test.index,
                        name="LightGBM")

lgbm_metrics = compute_metrics(y_test, lgbm_preds, clim_rmse=clim_rmse)
print("LightGBM final model metrics:")
print(lgbm_metrics)

# Persist predictions.
lgbm_preds.to_frame("lgbm_pred").to_parquet(OUTPUTS_DIR / "predictions_lgbm.parquet")
from src.models.lgbm_forecaster import save_model
save_model(lgbm_final, MODELS_DIR / "lgbm_final.txt")

# %%
# SHAP feature importance.
shap_values, explainer = compute_shap_values(lgbm_final, X_test_sc)
shap_summary = build_shap_summary(shap_values, feature_cols, top_n=20)
print("\nTop 20 features by mean absolute SHAP value:")
print(shap_summary.to_string(index=False))

# %%
# Meteorological attribution: group SHAP values by category.
feature_groups = {
    "NAO regime":         [c for c in feature_cols if "nao" in c.lower()],
    "AO":                 [c for c in feature_cols if "ao_"  in c.lower() or c.lower() == "ao"],
    "Thermal demand (HDD)": [c for c in feature_cols if "hdd" in c.lower() or "cdd" in c.lower()],
    "Blocking pattern":   [c for c in feature_cols if "blocking" in c.lower() or "greenland" in c.lower()],
    "Jet stream":         [c for c in feature_cols if "jet"  in c.lower()],
    "Wind generation":    [c for c in feature_cols if "wind" in c.lower() or "dunk" in c.lower()],
    "Seasonal position":  [c for c in feature_cols if "gas_year" in c.lower()],
}

attribution = build_meteorological_attribution(
    shap_values=shap_values,
    feature_names=feature_cols,
    feature_groups=feature_groups,
    index=X_test_sc.index,
)
print("\nMean absolute SHAP by meteorological group:")
print(attribution.abs().mean().sort_values(ascending=False).round(4).to_string())

# %% [markdown]
# ## 6. Deterministic evaluation: all models

# %%
pers_model = PersistenceBaseline(lag=1)
pers_model.fit(pd.DataFrame(index=y_train.index), y_train)
pers_preds = pd.Series(pers_model.predict(pd.DataFrame(index=y_test.index)),
                        index=y_test.index)

all_results = {
    "Climatology": compute_metrics(y_test, clim_preds,  clim_rmse=clim_rmse),
    "Persistence":  compute_metrics(y_test, pers_preds,  clim_rmse=clim_rmse),
    "LightGBM":    lgbm_metrics,
}

comparison_table = build_model_comparison_table(all_results)
print("\nModel comparison table:")
print(comparison_table.to_string())

# %%
# Time series of predictions vs actual for the test period.
fig = plot_predictions_vs_actual(
    y_actual=y_test,
    predictions={"Climatology": clim_preds, "Persistence": pers_preds, "LightGBM": lgbm_preds},
    title=f"Forecast vs actual: {df_test.index.min().date()} to {df_test.index.max().date()}",
)
fig.savefig(FIGURES_DIR / "predictions_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Residual distribution comparison.
residuals = {
    "Climatology": compute_residuals(y_test, clim_preds),
    "Persistence":  compute_residuals(y_test, pers_preds),
    "LightGBM":    compute_residuals(y_test, lgbm_preds),
}
fig = plot_residual_distribution(residuals, title="Residual distributions by model")
fig.savefig(FIGURES_DIR / "residual_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# MAE by month to show seasonal error profile.
mae_by_month = {
    "Climatology": compute_mae_by_month(y_test, clim_preds),
    "LightGBM":    compute_mae_by_month(y_test, lgbm_preds),
}
fig = plot_mae_by_month(mae_by_month, title="MAE by calendar month")
fig.savefig(FIGURES_DIR / "mae_by_month.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Error by NAO regime (requires the regime label from the CPC data).
cpc = pd.read_parquet(PROCESSED_DIR / "cpc_teleconnections.parquet")
if "NAO" in cpc.columns:
    regime_series = pd.cut(
        cpc["NAO"].resample("1D").mean().reindex(y_test.index),
        bins=[-np.inf, -0.8, 0.8, np.inf],
        labels=["cold", "neutral", "warm"],
    ).rename("nao_regime")

    regime_errors = compute_error_by_regime(y_test, lgbm_preds, regime_series)
    print("\nLightGBM MAE by NAO regime:")
    print(regime_errors.to_string())

# %% [markdown]
# ## 7. NGBoost probabilistic model

# %%
print("Running NGBoost walk-forward CV (6 folds)...")
ngb_cv_results = run_probabilistic_walk_forward_cv(
    df=df,
    target_col=TARGET,
    feature_cols=feature_cols,
    n_splits=6,
    initial_train_frac=0.50,
)

results_table = build_probabilistic_results_table(ngb_cv_results)
print(results_table.to_string())

pd.DataFrame(ngb_cv_results["fold_metrics"]).to_parquet(OUTPUTS_DIR / "wfcv_ngboost.parquet")

# %%
# Fit final NGBoost model on the full training set for evaluation figures.
ngb_final = fit_ngboost(X_train_sc, y_train, X_test_sc, y_test)
forecast   = predict_distribution(ngb_final, X_test_sc)
crps_vals  = compute_crps(forecast, y_test)
crpss      = compute_probabilistic_skill_score(crps_vals, y_train, y_test)

print(f"NGBoost final model: mean CRPS={crps_vals.mean():.4f}°C, CRPSS={crpss:.4f}")
print(f"Mean forecast sigma: {forecast.sigma.mean():.4f}°C")

# Persist NGBoost distribution parameters.
forecast.to_dataframe().to_parquet(OUTPUTS_DIR / "forecasts_ngboost.parquet")

# %%
# Reliability diagram.
reliability = compute_reliability(forecast, y_test)
fig = plot_reliability_diagram(reliability, model_name="NGBoost")
fig.savefig(FIGURES_DIR / "ngboost_reliability.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# PIT histogram.
pit = compute_pit_values(y_test, forecast.mu, forecast.sigma)
fig = plot_pit_histogram(pit, model_name="NGBoost")
fig.savefig(FIGURES_DIR / "ngboost_pit.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Interval coverage table.
coverage = compute_interval_coverage(y_test, forecast.mu, forecast.sigma)
print("\nNGBoost interval coverage:")
print(coverage.to_string())

# %%
# Spread-skill diagram.
spread_skill_df = compute_spread_skill(y_test, forecast.mu, forecast.sigma)
fig = plot_spread_skill(spread_skill_df, model_name="NGBoost")
fig.savefig(FIGURES_DIR / "ngboost_spread_skill.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Prediction interval time series for a winter window.
winter_start = str(y_test.index.min().year) + "-12-01"
winter_end   = str(y_test.index.min().year + 1) + "-02-28"
fig = plot_prediction_intervals(
    y_actual=y_test,
    mu=forecast.mu,
    sigma=forecast.sigma,
    zoom_period=(winter_start, winter_end),
    title=f"NGBoost 80% prediction interval: DJF {y_test.index.min().year}/{y_test.index.min().year+1}",
)
fig.savefig(FIGURES_DIR / "ngboost_intervals_winter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Consolidated probabilistic summary table

# %%
# Build the final summary table comparing all models across both
# deterministic and probabilistic metrics.
all_prob_results = {
    "Climatology": {"mean_CRPS": float(ps.crps_gaussian(
        y_test.to_numpy(),
        mu=np.full(len(y_test), float(y_train.mean())),
        sig=np.full(len(y_test), float(y_train.std())),
    ).mean()), "CRPSS": 0.0},
    "NGBoost": {"mean_CRPS": float(crps_vals.mean()), "CRPSS": round(crpss, 4)},
}

# Add coverage for NGBoost.
for label, row in coverage.iterrows():
    all_prob_results["NGBoost"][f"coverage_{label}"] = row["empirical_coverage"]
    all_prob_results["NGBoost"][f"width_{label}"]    = row["mean_width"]

import properscoring as ps
prob_table = build_probabilistic_summary_table(all_prob_results)
print("\nProbabilistic model summary:")
print(prob_table.to_string())

# %% [markdown]
# ## 9. Subseasonal regime classifier

# %%
sub_feat_path = PROCESSED_DIR / "features_subseasonal.parquet"
sub_targ_path = PROCESSED_DIR / "targets_subseasonal.parquet"

if sub_feat_path.exists() and sub_targ_path.exists():
    sub_features = pd.read_parquet(sub_feat_path)
    sub_targets  = pd.read_parquet(sub_targ_path)["regime_label"]

    # Align and drop nulls.
    sub_df = sub_features.join(sub_targets.rename("regime"), how="inner").dropna()
    sub_feature_cols = [c for c in sub_df.columns if c != "regime"]

    print(f"Subseasonal dataset: {sub_df.shape[0]} weekly rows, {len(sub_feature_cols)} features")
    print(f"Class balance: {sub_df['regime'].value_counts().to_dict()}")

    print("\nRunning regime classifier walk-forward CV (6 folds)...")
    regime_cv = run_regime_walk_forward_cv(
        df=sub_df,
        target_col="regime",
        feature_cols=sub_feature_cols,
        n_splits=6,
    )

    regime_cv_df = pd.DataFrame(regime_cv["fold_metrics"])
    regime_cv_df.to_parquet(OUTPUTS_DIR / "wfcv_regime.parquet")

    print("\nRegime classifier walk-forward results:")
    print(regime_cv_df[["fold", "brier_score_cold", "brier_skill_score",
                          "auc_cold_regime", "cold_f1"]].to_string(index=False))

    # Fit a final regime classifier on 80% of the weekly data.
    sub_split    = int(len(sub_df) * 0.80)
    sub_train    = sub_df.iloc[:sub_split]
    sub_test     = sub_df.iloc[sub_split:]

    clf = RegimeClassifier(calibrate=True)
    clf.fit(sub_train[sub_feature_cols], sub_train["regime"])

    y_sub_pred  = clf.predict(sub_test[sub_feature_cols])
    y_sub_proba = clf.predict_proba(sub_test[sub_feature_cols])

    regime_eval = evaluate_regime_classifier(
        y_actual=sub_test["regime"],
        y_predicted=y_sub_pred,
        y_proba=y_sub_proba,
        y_train=sub_train["regime"],
    )
    print("\nFinal regime classifier evaluation:")
    for k, v in regime_eval.items():
        print(f"  {k}: {v}")

    # Regime reliability diagram.
    regime_reliability = compute_regime_classifier_reliability(
        sub_test["regime"], y_sub_proba["cold"]
    )
    fig = plot_regime_reliability_diagram(regime_reliability)
    fig.savefig(FIGURES_DIR / "regime_reliability.png", dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("Subseasonal feature or target files not found. Run notebook 03 first.")

# %% [markdown]
# ## 10. Final comparison summary
#
# Consolidate all key metrics into the single comparison table that will
# appear in the model comparison section of the methodology report.

# %%
final_summary = {
    "Climatology":  {"MAE": round(float((y_test - clim_preds).abs().mean()), 4),
                     "RMSE": round(clim_rmse, 4), "skill_score": 0.0},
    "Persistence":  compute_metrics(y_test, pers_preds, clim_rmse=clim_rmse),
    "LightGBM":     lgbm_metrics,
    "NGBoost (det.)": compute_metrics(y_test, forecast.mu, clim_rmse=clim_rmse),
}

print("Final deterministic comparison table:")
print(build_model_comparison_table(final_summary).to_string())
print("\nNGBoost CRPSS vs climatological distribution:", round(crpss, 4))
