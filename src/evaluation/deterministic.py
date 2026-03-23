"""
deterministic.py

Comprehensive deterministic evaluation of point forecasts from the LightGBM
and baseline models. Provides per-model metric computation, residual analysis,
seasonal and regime-conditional error breakdowns, HDD forecast error
correlation with realised demand, and formatted comparison tables for use
in the model comparison notebook.

The central concept in this module is the skill score relative to climatology.
Raw RMSE and MAE values are not self-interpreting: an RMSE of 1.2 degrees
Celsius is excellent for a 10-day forecast and poor for a day-ahead one.
A skill score of 0.31 means the model reduces forecast error by 31% over
doing nothing but predicting the seasonal mean, and that interpretation
holds regardless of location or season.

Three levels of granularity are provided:

    Aggregate metrics     Single numbers over the full test period.
    Temporal breakdown    Error by month, by season, and by calendar week.
    Conditional analysis  Error stratified by NAO regime and Dunkelflaute state,
                          which is where the commercial value of the model is most
                          visible: do errors increase during the events that matter?

All plotting functions return matplotlib Figure objects rather than calling
plt.show() directly. This keeps the functions testable and composable in
notebooks where display is handled by the caller.

Setup required before use:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# Season definitions by month number. Meteorological seasons are used throughout
# because they align with the physical drivers of weather-demand linkage:
# DJF is the gas demand peak, MAM and SON are transition periods with high
# forecast uncertainty, JJA is the Dunkelflaute shoulder season.
METEOROLOGICAL_SEASONS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

# Colour palette for multi-model plots. Using a fixed palette ensures
# consistent visual encoding across all figures in the notebook.
MODEL_COLOURS: dict[str, str] = {
    "Climatology": "#8c8c8c",
    "Persistence":  "#5b9bd5",
    "NWP Direct":  "#ed7d31",
    "LightGBM":    "#70ad47",
    "NGBoost":     "#264478",
}


def compute_metrics(
    y_actual:   pd.Series,
    y_pred:     pd.Series,
    clim_rmse:  Optional[float] = None,
) -> dict[str, float]:
    """
    Computes MAE, RMSE, and optionally the skill score relative to a
    provided climatological RMSE.

    Parameters
    ----------
    y_actual:
        Observed target values.
    y_pred:
        Predicted values, aligned to y_actual.index.
    clim_rmse:
        RMSE of the climatological baseline over the same period. If provided,
        the skill score is included in the output. If None, the skill score
        key is absent from the returned dict.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'MAE', 'RMSE', and optionally 'skill_score'.
    """
    aligned = pd.DataFrame({"actual": y_actual, "pred": y_pred}).dropna()
    if aligned.empty:
        logger.warning("compute_metrics: no overlapping observations after alignment.")
        return {}

    mae  = float(mean_absolute_error(aligned["actual"], aligned["pred"]))
    rmse = float(np.sqrt(mean_squared_error(aligned["actual"], aligned["pred"])))

    out = {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}

    if clim_rmse is not None and clim_rmse > 0:
        out["skill_score"] = round(1.0 - rmse / clim_rmse, 4)

    return out


def compute_residuals(
    y_actual: pd.Series,
    y_pred:   pd.Series,
) -> pd.Series:
    """
    Computes signed residuals (actual minus predicted) as a UTC-indexed Series.

    A positive residual means the actual temperature was higher than forecast
    (warm surprise). A negative residual means the actual was colder than
    forecast (cold surprise). Cold surprises are the commercially significant
    ones: they imply realised HDD above forecast, meaning gas demand was
    higher than expected.

    Parameters
    ----------
    y_actual:
        Observed target values.
    y_pred:
        Predicted values, aligned to y_actual.index.

    Returns
    -------
    pd.Series
        Signed residuals, same index as the input after alignment.
    """
    aligned = pd.DataFrame({"actual": y_actual, "pred": y_pred}).dropna()
    residuals = aligned["actual"] - aligned["pred"]
    residuals.name = "residual"
    return residuals


def compute_mae_by_month(
    y_actual: pd.Series,
    y_pred:   pd.Series,
) -> pd.Series:
    """
    Computes mean absolute error broken down by calendar month.

    Monthly error profiles reveal seasonal model behaviour. Models that
    perform well in summer but poorly in winter (DJF) are less commercially
    useful than the aggregate metrics suggest because winter accuracy is
    what matters for gas demand forecasting.

    Parameters
    ----------
    y_actual:
        Observed target values with DatetimeIndex.
    y_pred:
        Predicted values aligned to y_actual.

    Returns
    -------
    pd.Series
        MAE by month (1 to 12), indexed by integer month number.
    """
    residuals = compute_residuals(y_actual, y_pred).abs()
    return residuals.groupby(residuals.index.month).mean().rename("MAE_by_month")


def compute_mae_by_season(
    y_actual: pd.Series,
    y_pred:   pd.Series,
) -> pd.Series:
    """
    Computes mean absolute error broken down by meteorological season
    (DJF, MAM, JJA, SON).

    Parameters
    ----------
    y_actual:
        Observed target values with DatetimeIndex.
    y_pred:
        Predicted values aligned to y_actual.

    Returns
    -------
    pd.Series
        MAE by season, indexed by season label string.
    """
    abs_err = compute_residuals(y_actual, y_pred).abs()

    month_to_season = {}
    for season, months in METEOROLOGICAL_SEASONS.items():
        for m in months:
            month_to_season[m] = season

    season_labels = abs_err.index.month.map(month_to_season)
    return abs_err.groupby(season_labels).mean().rename("MAE_by_season")


def compute_error_by_regime(
    y_actual:     pd.Series,
    y_pred:       pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    Stratifies forecast errors by NAO regime or any other ternary label.

    The key question this answers for a trading desk is: does the model's
    accuracy degrade specifically during cold regime events? A model that
    has low aggregate error but large errors precisely during cold anomalies
    is worse than its headline metric suggests. Conversely, a model with
    lower error in cold regimes than neutral regimes is providing disproportionate
    skill exactly when that skill has the highest commercial value.

    Parameters
    ----------
    y_actual:
        Observed target values.
    y_pred:
        Predicted values aligned to y_actual.
    regime_labels:
        Series of regime labels ('cold', 'neutral', 'warm') aligned to the
        same index as y_actual. Typically from RegimeClassifier.predict or
        from the observed weekly regime target.

    Returns
    -------
    pd.DataFrame
        One row per regime with columns 'MAE', 'RMSE', 'n_observations',
        and 'mean_residual' (signed bias in each regime).
    """
    residuals = compute_residuals(y_actual, y_pred)
    df = pd.DataFrame({
        "residual":     residuals,
        "abs_residual": residuals.abs(),
        "regime":       regime_labels.reindex(residuals.index),
    }).dropna()

    records = []
    for regime, group in df.groupby("regime"):
        mae  = float(group["abs_residual"].mean())
        rmse = float(np.sqrt((group["residual"] ** 2).mean()))
        records.append({
            "regime":          regime,
            "MAE":             round(mae,  4),
            "RMSE":            round(rmse, 4),
            "mean_residual":   round(float(group["residual"].mean()), 4),
            "n_observations":  len(group),
        })

    return pd.DataFrame(records).set_index("regime")


def compute_hdd_error_demand_correlation(
    hdd_actual:   pd.Series,
    hdd_forecast: pd.Series,
    realised_load: pd.Series,
) -> pd.DataFrame:
    """
    Quantifies the relationship between weekly HDD forecast error and realised
    electricity demand deviation. This is the core of the commodity linkage
    analysis: it demonstrates that weather forecast quality directly determines
    demand forecast quality and, by extension, trading position accuracy.

    The analysis resamples all inputs to a common weekly frequency, computes
    the signed HDD forecast error (actual minus forecast), and correlates it
    with the week-on-week change in realised load. A positive correlation
    indicates that cold surprises (positive HDD error) are followed by demand
    surprises in the same direction, as expected from the physical HDD-to-demand
    relationship.

    Parameters
    ----------
    hdd_actual:
        Realised daily HDD Series, UTC-indexed.
    hdd_forecast:
        Forecast daily HDD Series from the model, UTC-indexed.
    realised_load:
        Hourly or daily realised electricity load in MW, UTC-indexed.
        Typically sourced from ENTSO-E via entso_pipeline.py.

    Returns
    -------
    pd.DataFrame
        Weekly-indexed DataFrame with columns for HDD actual, HDD forecast,
        HDD error, load change, and a summary row containing the Pearson
        and Spearman correlations between HDD error and load change.
    """
    hdd_actual_7d   = hdd_actual.resample("1W").sum()
    hdd_forecast_7d = hdd_forecast.resample("1W").sum()
    load_weekly     = realised_load.resample("1W").mean()

    aligned = pd.DataFrame({
        "hdd_actual_7d":    hdd_actual_7d,
        "hdd_forecast_7d":  hdd_forecast_7d,
        "load_mw":          load_weekly,
    }).dropna()

    aligned["hdd_error_7d"]  = aligned["hdd_actual_7d"] - aligned["hdd_forecast_7d"]
    aligned["load_change_pct"] = aligned["load_mw"].pct_change() * 100

    pearson  = float(aligned["hdd_error_7d"].corr(aligned["load_change_pct"]))
    spearman = float(aligned["hdd_error_7d"].corr(aligned["load_change_pct"], method="spearman"))

    logger.info(
        "HDD error vs load change: Pearson=%.3f, Spearman=%.3f over %d weeks.",
        pearson, spearman, len(aligned),
    )

    aligned.attrs["pearson_correlation"]  = pearson
    aligned.attrs["spearman_correlation"] = spearman
    return aligned


def build_model_comparison_table(
    results: dict[str, dict[str, float]],
    sort_by: str = "skill_score",
) -> pd.DataFrame:
    """
    Assembles a model comparison table from a dictionary of per-model metric
    dictionaries. This is the primary output of notebook 04 and the first
    table a reviewer will look at.

    The table is sorted by the specified metric in descending order, so the
    best-performing model appears at the top. Models without a skill score
    (e.g. if the climatological RMSE was not provided) have that column set
    to NaN rather than being excluded.

    Parameters
    ----------
    results:
        Dictionary mapping model name to its metric dict. Each metric dict
        should contain at least 'MAE' and 'RMSE', and optionally 'skill_score'.
        The expected structure is the output of compute_metrics.
    sort_by:
        Column to sort by. Descending for 'skill_score', ascending for 'MAE'
        or 'RMSE'. The function detects which direction is better automatically.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table with one row per model.
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    lower_is_better = {"MAE", "RMSE", "MSE"}
    ascending = sort_by in lower_is_better
    df = df.sort_values(sort_by, ascending=ascending)

    return df


def plot_predictions_vs_actual(
    y_actual:    pd.Series,
    predictions: dict[str, pd.Series],
    title:       str = "Forecast vs Actual",
    zoom_period: Optional[tuple[str, str]] = None,
    figsize:     tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plots one or more model predictions against observed values over time.

    Multiple models can be overlaid on the same axis by passing a dictionary
    of name-to-Series mappings. A zoom_period can be specified to focus on
    a commercially interesting window such as a cold spell or Dunkelflaute event.

    Parameters
    ----------
    y_actual:
        Observed target values with DatetimeIndex.
    predictions:
        Dictionary mapping model name to its prediction Series.
    title:
        Figure title.
    zoom_period:
        Optional tuple of ISO date strings (start, end) to restrict the x-axis.
        Example: ('2021-01-01', '2021-02-28') for the February 2021 cold wave.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object. Call fig.savefig() or display in a notebook.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_actual = y_actual
    plot_preds  = predictions

    if zoom_period is not None:
        start, end   = zoom_period
        plot_actual  = y_actual.loc[start:end]
        plot_preds   = {name: s.loc[start:end] for name, s in predictions.items()}

    ax.plot(
        plot_actual.index, plot_actual.values,
        label="Actual", colour="black", linewidth=1.4, alpha=0.9, zorder=5,
    )

    for model_name, pred_series in plot_preds.items():
        colour = MODEL_COLOURS.get(model_name, None)
        ax.plot(
            pred_series.index, pred_series.values,
            label=model_name, colour=colour,
            linestyle="--", linewidth=1.0, alpha=0.75,
        )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_residuals_over_time(
    residuals_dict: dict[str, pd.Series],
    title:          str = "Forecast residuals over time",
    figsize:        tuple[int, int] = (15, 4),
) -> plt.Figure:
    """
    Plots signed residuals over time for one or more models on a shared axis.

    Persistent positive or negative residual runs indicate that the model
    is systematically wrong during specific periods, often corresponding to
    blocking events or unusual seasonal transitions that are underrepresented
    in the training data.

    Parameters
    ----------
    residuals_dict:
        Dictionary mapping model name to its residual Series (from compute_residuals).
    title:
        Figure title.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, resid in residuals_dict.items():
        colour = MODEL_COLOURS.get(model_name, None)
        ax.plot(resid.index, resid.values, label=model_name, colour=colour,
                linewidth=0.8, alpha=0.7)

    ax.axhline(0, colour="black", linewidth=0.8, linestyle="-")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (°C)")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_residual_distribution(
    residuals_dict: dict[str, pd.Series],
    title:          str = "Residual distribution by model",
    figsize:        tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plots KDE-smoothed residual distributions for multiple models on a shared axis.

    A well-specified model should produce residuals centred near zero with no
    heavy systematic skew. Negative skew (cold bias) is the most commercially
    concerning pattern because it implies the model is systematically
    underestimating cold events.

    Parameters
    ----------
    residuals_dict:
        Dictionary mapping model name to its residual Series.
    title:
        Figure title.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, resid in residuals_dict.items():
        colour = MODEL_COLOURS.get(model_name, None)
        sns.kdeplot(resid.dropna(), ax=ax, label=model_name, colour=colour,
                    linewidth=1.4, fill=False)

    ax.axvline(0, colour="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Residual (°C)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_mae_by_month(
    mae_by_month_dict: dict[str, pd.Series],
    title:             str = "MAE by calendar month",
    figsize:           tuple[int, int] = (11, 5),
) -> plt.Figure:
    """
    Plots mean absolute error by calendar month for multiple models.

    The expected pattern for a well-fitted model is lower MAE in summer
    (JJA) and higher MAE in winter (DJF) due to the greater synoptic
    variability in winter. A model with higher winter MAE than climatology
    in DJF should not be presented to a gas desk.

    Parameters
    ----------
    mae_by_month_dict:
        Dictionary mapping model name to a Series indexed by month number (1-12),
        as returned by compute_mae_by_month.
    title:
        Figure title.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, 13)
    width = 0.8 / max(len(mae_by_month_dict), 1)

    for i, (model_name, mae_series) in enumerate(mae_by_month_dict.items()):
        offset = (i - len(mae_by_month_dict) / 2) * width + width / 2
        colour = MODEL_COLOURS.get(model_name, None)
        values = [float(mae_series.get(m, np.nan)) for m in range(1, 13)]
        ax.bar(x + offset, values, width=width * 0.9,
               label=model_name, colour=colour, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("MAE (°C)")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_skill_score_by_fold(
    fold_results_dict: dict[str, list[dict]],
    title:             str = "Skill score per walk-forward fold",
    figsize:           tuple[int, int] = (11, 5),
) -> plt.Figure:
    """
    Plots the walk-forward cross-validation skill score per fold for one or
    more models. Each fold represents a contiguous test period; the plot
    shows how model skill evolves across different weather regimes and years.

    A model with consistently positive skill scores across all folds provides
    a more reliable forecast than one with high mean skill driven by one
    exceptional fold.

    Parameters
    ----------
    fold_results_dict:
        Dictionary mapping model name to the list of per-fold metric dicts
        from WalkForwardResult.fold_metrics (lgbm_forecaster.py). Each dict
        must contain 'fold' and 'skill_score' keys.
    title:
        Figure title.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, fold_metrics in fold_results_dict.items():
        folds  = [m["fold"]        for m in fold_metrics]
        skills = [m["skill_score"] for m in fold_metrics]
        colour = MODEL_COLOURS.get(model_name, None)
        ax.plot(folds, skills, marker="o", label=model_name,
                colour=colour, linewidth=1.2, markersize=5)

    ax.axhline(0, colour="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Skill score vs climatology")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_hdd_error_vs_demand(
    linkage_df: pd.DataFrame,
    pearson_r:  Optional[float] = None,
    title:      str = "Weekly HDD forecast error vs realised load change",
    figsize:    tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Scatter plot of weekly HDD forecast error against realised load change.

    This figure is the centrepiece of the commodity linkage notebook. A
    positive slope demonstrates that cold surprises (positive HDD error)
    produce positive demand surprises, quantifying the commercial stakes
    of weather forecast accuracy in MW of unexpectedly high gas demand.

    Parameters
    ----------
    linkage_df:
        Output from compute_hdd_error_demand_correlation, containing columns
        'hdd_error_7d' and 'load_change_pct'.
    pearson_r:
        Optional Pearson correlation to annotate on the figure. If None,
        it is computed from linkage_df.
    title:
        Figure title.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = linkage_df["hdd_error_7d"].dropna()
    y = linkage_df["load_change_pct"].reindex(x.index).dropna()
    x = x.reindex(y.index)

    ax.scatter(x, y, s=18, alpha=0.55, colour="#264478", edgecolours="none")

    if len(x) > 2:
        m, b    = np.polyfit(x, y, 1)
        x_line  = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, colour="#ed7d31",
                linewidth=1.4, label=f"OLS fit (slope={m:.2f})")

    if pearson_r is None and len(x) > 1:
        pearson_r = float(x.corr(y))

    if pearson_r is not None:
        ax.annotate(
            f"Pearson r = {pearson_r:.3f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=10, colour="#333333",
        )

    ax.axhline(0, colour="black", linewidth=0.6, linestyle="-", alpha=0.4)
    ax.axvline(0, colour="black", linewidth=0.6, linestyle="-", alpha=0.4)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Weekly HDD forecast error (actual minus forecast, degree-days)")
    ax.set_ylabel("Week-on-week load change (%)")
    ax.legend(fontsize=9)
    ax.grid(linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


def summarise_cold_event_errors(
    y_actual:     pd.Series,
    y_pred:       pd.Series,
    cold_events:  pd.DatetimeIndex,
    window_days:  int = 3,
) -> pd.DataFrame:
    """
    Computes mean absolute error during and around identified cold events,
    providing a before/during/after breakdown.

    This function is used in the commodity linkage notebook to answer the
    specific question: how accurate is the model's forecast during the cold
    spells that move gas prices? If errors are larger during these events,
    the model is least useful when it matters most.

    Parameters
    ----------
    y_actual:
        Observed target values with DatetimeIndex.
    y_pred:
        Predicted values aligned to y_actual.
    cold_events:
        DatetimeIndex of cold event start timestamps (e.g. Dunkelflaute
        events or blocking onset dates).
    window_days:
        Number of days before and after each event to include in the
        comparison windows.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows 'before', 'during', 'after' and columns
        'MAE', 'mean_residual', 'n_observations'.
    """
    residuals = compute_residuals(y_actual, y_pred)
    delta     = pd.Timedelta(days=window_days)
    records   = {"before": [], "during": [], "after": []}

    for event_start in cold_events:
        before_mask = (
            (residuals.index >= event_start - delta)
            & (residuals.index < event_start)
        )
        during_mask = (
            (residuals.index >= event_start)
            & (residuals.index < event_start + delta)
        )
        after_mask  = (
            (residuals.index >= event_start + delta)
            & (residuals.index < event_start + 2 * delta)
        )
        records["before"].append(residuals[before_mask])
        records["during"].append(residuals[during_mask])
        records["after"].append(residuals[after_mask])

    rows = []
    for period, series_list in records.items():
        if series_list:
            combined = pd.concat(series_list)
            rows.append({
                "period":          period,
                "MAE":             round(float(combined.abs().mean()), 4),
                "mean_residual":   round(float(combined.mean()), 4),
                "n_observations":  len(combined),
            })

    return pd.DataFrame(rows).set_index("period")
