"""
probabilistic.py

Comprehensive evaluation of probabilistic forecast outputs from NGBoost
(short-term temperature distributions) and the XGBoost regime classifier
(subseasonal cold regime probabilities).

Probabilistic evaluation requires a different mental model to deterministic
evaluation. A point forecast is either right or wrong, within some tolerance.
A probabilistic forecast is never simply right or wrong: it is assessed by
whether the stated probabilities are consistent with observed frequencies
(calibration) and whether the probability mass is concentrated tightly
around the outcome (sharpness). Both properties must hold simultaneously
for the forecast to be useful.

The framework used here distinguishes four qualities:

    Calibration     Does the 30th percentile forecast contain the outcome
                    30% of the time? Measured via the reliability diagram.
    Sharpness       How narrow are the prediction intervals? Conditional on
                    calibration, narrower is always better.
    Resolution      Does the forecast distribution vary meaningfully with
                    atmospheric state, or does it issue nearly the same
                    distribution regardless of conditions?
    CRPS skill      Does the model outperform a climatological distribution
                    baseline in expected CRPS?

For the regime classifier, an analogous framework applies. Calibration is
assessed via the reliability diagram for the cold class probability.
Resolution is assessed by comparing cold probability distributions on cold
versus non-cold observed weeks. Skill is assessed via the Brier Skill Score.

All plotting functions return matplotlib Figure objects without calling
plt.show(), to keep them composable in notebooks.

Setup required before use:
    pip install matplotlib seaborn pandas numpy scipy properscoring
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import properscoring as ps
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


# Standard quantile levels for reliability diagram computation.
# These match the levels in ngboost_prob.py for consistency.
RELIABILITY_QUANTILES: list[float] = [
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
]

# Prediction interval coverage levels for sharpness and coverage tables.
INTERVAL_COVERAGES: dict[str, tuple[float, float]] = {
    "50%":  (0.25, 0.75),
    "80%":  (0.10, 0.90),
    "90%":  (0.05, 0.95),
}

# Meteorological seasons, matching deterministic.py convention.
METEOROLOGICAL_SEASONS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3,  4, 5],
    "JJA": [6,  7, 8],
    "SON": [9, 10, 11],
}


def compute_pit_values(
    y_actual: pd.Series,
    mu:       pd.Series,
    sigma:    pd.Series,
) -> pd.Series:
    """
    Computes Probability Integral Transform (PIT) values for a set of Normal
    predictive distributions.

    The PIT maps each observation through its own predictive CDF:

        PIT_t = Phi((y_t - mu_t) / sigma_t)

    where Phi is the standard Normal CDF. If the predictive distributions are
    correctly specified, the PIT values should be uniformly distributed on [0, 1].
    Deviations from uniformity indicate miscalibration:

        Clustering near 0 or 1  systematic bias (observations in the tails)
        U-shape                 overdispersion (intervals too wide)
        Arch shape              underdispersion (intervals too narrow)

    PIT histograms are a more sensitive diagnostic than the reliability diagram
    for detecting specific failure modes of the predictive distribution.

    Parameters
    ----------
    y_actual:
        Observed target values.
    mu:
        Predicted distribution means, aligned to y_actual.index.
    sigma:
        Predicted distribution standard deviations, aligned to y_actual.index.

    Returns
    -------
    pd.Series
        PIT values between 0 and 1, indexed as y_actual.
    """
    aligned = pd.DataFrame({
        "actual": y_actual,
        "mu":     mu,
        "sigma":  sigma,
    }).dropna()

    pit = pd.Series(
        stats.norm.cdf(aligned["actual"], loc=aligned["mu"], scale=aligned["sigma"]),
        index=aligned.index,
        name="pit",
    )
    return pit


def compute_interval_coverage(
    y_actual:  pd.Series,
    mu:        pd.Series,
    sigma:     pd.Series,
    coverages: Optional[dict[str, tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Computes the empirical coverage of prediction intervals at multiple
    nominal levels and compares against the expected coverage.

    A 80% prediction interval should contain the observation 80% of the time.
    Empirical coverage below the nominal level indicates underdispersion
    (overconfidence). Empirical coverage above the nominal level indicates
    overdispersion (underconfidence). Both are commercially costly: an
    overconfident model understates position risk, an underconfident model
    overstates it.

    Parameters
    ----------
    y_actual:
        Observed target values.
    mu:
        Predicted means aligned to y_actual.
    sigma:
        Predicted standard deviations aligned to y_actual.
    coverages:
        Dictionary mapping label to (lower_q, upper_q) quantile pairs.
        Defaults to INTERVAL_COVERAGES.

    Returns
    -------
    pd.DataFrame
        One row per interval level with columns 'nominal_coverage',
        'empirical_coverage', 'coverage_error', and 'mean_width'.
    """
    if coverages is None:
        coverages = INTERVAL_COVERAGES

    aligned = pd.DataFrame({
        "actual": y_actual,
        "mu":     mu,
        "sigma":  sigma,
    }).dropna()

    records = []
    for label, (lower_q, upper_q) in coverages.items():
        lower_bound = stats.norm.ppf(lower_q, loc=aligned["mu"], scale=aligned["sigma"])
        upper_bound = stats.norm.ppf(upper_q, loc=aligned["mu"], scale=aligned["sigma"])

        in_interval      = (aligned["actual"] >= lower_bound) & (aligned["actual"] <= upper_bound)
        empirical_cov    = float(in_interval.mean())
        nominal_cov      = upper_q - lower_q
        mean_width       = float((upper_bound - lower_bound).mean())
        coverage_error   = empirical_cov - nominal_cov

        records.append({
            "interval":          label,
            "nominal_coverage":  round(nominal_cov,    4),
            "empirical_coverage": round(empirical_cov, 4),
            "coverage_error":    round(coverage_error, 4),
            "mean_width":        round(mean_width,     4),
        })

    return pd.DataFrame(records).set_index("interval")


def compute_spread_skill(
    y_actual: pd.Series,
    mu:       pd.Series,
    sigma:    pd.Series,
    n_bins:   int = 10,
) -> pd.DataFrame:
    """
    Computes the spread-skill relationship by binning forecast standard
    deviations and computing the RMSE within each sigma bin.

    For a perfectly calibrated model, the RMSE within each sigma bin should
    equal the mean sigma of that bin: sigma is exactly the predicted
    standard deviation of the error. This relationship is known as the
    spread-skill relationship and is a standard diagnostic for probabilistic
    NWP systems.

    In practice, a model that produces larger sigma values during uncertain
    forecast situations (e.g. blocking onset, rapid temperature transitions)
    and smaller sigma values during predictable situations (stable airmasses)
    demonstrates genuine uncertainty quantification. A model that outputs a
    nearly constant sigma has learned nothing about forecast uncertainty.

    Parameters
    ----------
    y_actual:
        Observed target values.
    mu:
        Predicted means.
    sigma:
        Predicted standard deviations.
    n_bins:
        Number of sigma bins for the spread-skill diagram.

    Returns
    -------
    pd.DataFrame
        One row per sigma bin with columns 'mean_sigma', 'rmse_in_bin',
        and 'n_observations'.
    """
    aligned = pd.DataFrame({
        "actual": y_actual,
        "mu":     mu,
        "sigma":  sigma,
    }).dropna()

    aligned["sq_error"] = (aligned["actual"] - aligned["mu"]) ** 2
    aligned["sigma_bin"] = pd.qcut(aligned["sigma"], q=n_bins, duplicates="drop")

    records = []
    for bin_label, group in aligned.groupby("sigma_bin", observed=True):
        records.append({
            "sigma_bin":       str(bin_label),
            "mean_sigma":      round(float(group["sigma"].mean()), 4),
            "rmse_in_bin":     round(float(np.sqrt(group["sq_error"].mean())), 4),
            "n_observations":  len(group),
        })

    return pd.DataFrame(records)


def compute_conditional_crps(
    y_actual:      pd.Series,
    mu:            pd.Series,
    sigma:         pd.Series,
    condition:     pd.Series,
) -> pd.DataFrame:
    """
    Computes mean CRPS conditional on a categorical variable such as NAO
    regime, season, or Dunkelflaute state.

    This answers the question: does the probabilistic forecast skill vary
    by atmospheric regime? A model that performs well in neutral regimes but
    poorly in cold ones provides less commercial value than its aggregate
    CRPS suggests.

    Parameters
    ----------
    y_actual:
        Observed target values.
    mu:
        Predicted means.
    sigma:
        Predicted standard deviations.
    condition:
        Categorical Series aligned to the same index. Values define the
        conditioning groups.

    Returns
    -------
    pd.DataFrame
        One row per condition value with columns 'mean_CRPS', 'std_CRPS',
        and 'n_observations'.
    """
    aligned = pd.DataFrame({
        "actual":    y_actual,
        "mu":        mu,
        "sigma":     sigma,
        "condition": condition,
    }).dropna()

    aligned["crps"] = ps.crps_gaussian(
        aligned["actual"].to_numpy(),
        mu=aligned["mu"].to_numpy(),
        sig=aligned["sigma"].to_numpy(),
    )

    records = []
    for cond_val, group in aligned.groupby("condition"):
        records.append({
            "condition":      cond_val,
            "mean_CRPS":      round(float(group["crps"].mean()), 4),
            "std_CRPS":       round(float(group["crps"].std()),  4),
            "n_observations": len(group),
        })

    return pd.DataFrame(records).set_index("condition")


def compute_regime_classifier_reliability(
    y_actual:         pd.Series,
    cold_probability: pd.Series,
    n_bins:           int = 10,
) -> pd.DataFrame:
    """
    Computes the reliability of the regime classifier's cold probability
    forecast. For each probability bin, the observed frequency of cold
    outcomes is computed and compared against the bin midpoint.

    A perfectly calibrated classifier lies on the diagonal: when it says
    40% probability of cold, cold occurs 40% of the time. Deviations indicate
    systematic over- or underconfidence.

    This is directly analogous to the reliability diagram for continuous
    probabilistic forecasts but adapted for binary (cold or not cold) outcomes.

    Parameters
    ----------
    y_actual:
        Observed regime labels Series with values in {'cold', 'neutral', 'warm'}.
    cold_probability:
        Predicted cold regime probability Series, aligned to y_actual.index.
    n_bins:
        Number of probability bins.

    Returns
    -------
    pd.DataFrame
        One row per bin with columns 'bin_midpoint', 'observed_cold_frequency',
        'reliability_error', and 'n_observations'.
    """
    aligned = pd.DataFrame({
        "cold_actual": (y_actual == "cold").astype(int),
        "cold_prob":   cold_probability,
    }).dropna()

    bins          = np.linspace(0.0, 1.0, n_bins + 1)
    aligned["bin"] = pd.cut(aligned["cold_prob"], bins=bins, include_lowest=True)

    records = []
    for bin_label, group in aligned.groupby("bin", observed=True):
        bin_mid       = float(bin_label.mid)
        obs_freq      = float(group["cold_actual"].mean())
        rel_error     = obs_freq - bin_mid
        records.append({
            "bin_midpoint":            round(bin_mid,  4),
            "observed_cold_frequency": round(obs_freq, 4),
            "reliability_error":       round(rel_error, 4),
            "n_observations":          len(group),
        })

    return pd.DataFrame(records)


def plot_reliability_diagram(
    reliability_df:   pd.DataFrame,
    model_name:       str = "NGBoost",
    title:            str = "Reliability diagram",
    figsize:          tuple[int, int] = (6, 6),
) -> plt.Figure:
    """
    Plots a reliability diagram for a continuous probabilistic forecast.

    The x-axis is the nominal quantile level; the y-axis is the observed
    frequency of outcomes below that quantile. The diagonal represents
    perfect calibration.

    Parameters
    ----------
    reliability_df:
        Output from ngboost_prob.compute_reliability, containing columns
        'nominal_quantile' and 'observed_frequency'.
    model_name:
        Label for the model line in the legend.
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

    ax.plot([0, 1], [0, 1], colour="black", linewidth=1.0,
            linestyle="--", label="Perfect calibration", alpha=0.7)

    ax.plot(
        reliability_df["nominal_quantile"],
        reliability_df["observed_frequency"],
        marker="o", markersize=5, linewidth=1.4,
        colour="#264478", label=model_name,
    )

    ax.fill_between(
        reliability_df["nominal_quantile"],
        reliability_df["nominal_quantile"],
        reliability_df["observed_frequency"],
        alpha=0.12, colour="#264478",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Observed frequency")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_regime_reliability_diagram(
    reliability_df: pd.DataFrame,
    model_name:     str = "Regime classifier",
    title:          str = "Cold regime probability reliability",
    figsize:        tuple[int, int] = (6, 6),
) -> plt.Figure:
    """
    Plots the reliability diagram for the regime classifier's cold probability.
    Bubble size represents the number of observations in each probability bin.

    Parameters
    ----------
    reliability_df:
        Output from compute_regime_classifier_reliability, containing
        'bin_midpoint', 'observed_cold_frequency', and 'n_observations'.
    model_name:
        Label for the model in the legend.
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

    ax.plot([0, 1], [0, 1], colour="black", linewidth=1.0,
            linestyle="--", label="Perfect calibration", alpha=0.7)

    n_obs = reliability_df["n_observations"]
    sizes = 20 + 60 * (n_obs / n_obs.max())

    ax.scatter(
        reliability_df["bin_midpoint"],
        reliability_df["observed_cold_frequency"],
        s=sizes, colour="#264478", alpha=0.8,
        label=f"{model_name} (bubble = n obs)",
        zorder=4,
    )

    ax.plot(
        reliability_df["bin_midpoint"],
        reliability_df["observed_cold_frequency"],
        colour="#264478", linewidth=1.2, alpha=0.7, zorder=3,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Forecast cold probability")
    ax.set_ylabel("Observed cold frequency")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_pit_histogram(
    pit_values:  pd.Series,
    model_name:  str = "NGBoost",
    title:       str = "PIT histogram",
    n_bins:      int = 20,
    figsize:     tuple[int, int] = (7, 4),
) -> plt.Figure:
    """
    Plots the PIT histogram with a horizontal reference line at the expected
    uniform density. Deviations from the flat reference line indicate
    miscalibration. An arch-shaped histogram indicates underdispersion;
    a U-shaped histogram indicates overdispersion; a systematic left or
    right skew indicates a bias in the predictive mean.

    Parameters
    ----------
    pit_values:
        PIT values from compute_pit_values, between 0 and 1.
    model_name:
        Label used in the figure title.
    title:
        Figure title.
    n_bins:
        Number of histogram bins.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    clean = pit_values.dropna()
    ax.hist(clean, bins=n_bins, colour="#264478", alpha=0.75,
            edgecolor="white", linewidth=0.4, density=True)

    ax.axhline(1.0, colour="black", linewidth=1.0, linestyle="--",
               label="Uniform reference", alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_title(f"{title} ({model_name})", fontsize=12, pad=10)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_spread_skill(
    spread_skill_df: pd.DataFrame,
    model_name:      str = "NGBoost",
    title:           str = "Spread-skill relationship",
    figsize:         tuple[int, int] = (7, 6),
) -> plt.Figure:
    """
    Plots the spread-skill diagram: predicted sigma (spread) on the x-axis
    against RMSE within each sigma bin (skill) on the y-axis. The diagonal
    represents a perfectly spread-skill consistent forecast.

    Points above the diagonal indicate that the model is overconfident in
    situations where it predicts low uncertainty. Points below indicate
    that the model is underconfident when it predicts high uncertainty.

    Parameters
    ----------
    spread_skill_df:
        Output from compute_spread_skill, containing 'mean_sigma' and
        'rmse_in_bin'.
    model_name:
        Label for the model in the legend.
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

    s_max = max(
        spread_skill_df["mean_sigma"].max(),
        spread_skill_df["rmse_in_bin"].max(),
    ) * 1.05

    ax.plot([0, s_max], [0, s_max], colour="black", linewidth=1.0,
            linestyle="--", label="Perfect spread-skill", alpha=0.7)

    ax.plot(
        spread_skill_df["mean_sigma"],
        spread_skill_df["rmse_in_bin"],
        marker="o", markersize=5, linewidth=1.2,
        colour="#264478", label=model_name,
    )

    ax.set_xlim(0, s_max)
    ax.set_ylim(0, s_max)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Mean forecast sigma (°C)")
    ax.set_ylabel("RMSE within sigma bin (°C)")
    ax.legend(fontsize=9)
    ax.grid(linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_prediction_intervals(
    y_actual:   pd.Series,
    mu:         pd.Series,
    sigma:      pd.Series,
    lower_q:    float = 0.10,
    upper_q:    float = 0.90,
    title:      str = "Forecast with 80% prediction interval",
    zoom_period: Optional[tuple[str, str]] = None,
    figsize:    tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plots actual observations, the forecast mean, and a prediction interval
    band over time.

    Observations falling outside the shaded band represent interval violations.
    A correctly calibrated 80% interval should show approximately 20%
    violation rate. The figure makes over- and under-confidence visually
    immediate in a way that summary statistics alone do not.

    Parameters
    ----------
    y_actual:
        Observed target values with DatetimeIndex.
    mu:
        Predicted means aligned to y_actual.
    sigma:
        Predicted standard deviations aligned to y_actual.
    lower_q:
        Lower quantile for the interval band.
    upper_q:
        Upper quantile for the interval band.
    title:
        Figure title.
    zoom_period:
        Optional (start, end) ISO date strings to restrict the x-axis.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    lower = pd.Series(
        stats.norm.ppf(lower_q, loc=mu.values, scale=sigma.values),
        index=mu.index,
    )
    upper = pd.Series(
        stats.norm.ppf(upper_q, loc=mu.values, scale=sigma.values),
        index=mu.index,
    )

    if zoom_period is not None:
        start, end = zoom_period
        y_actual   = y_actual.loc[start:end]
        mu         = mu.loc[start:end]
        lower      = lower.loc[start:end]
        upper      = upper.loc[start:end]

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(
        lower.index, lower.values, upper.values,
        alpha=0.25, colour="#264478",
        label=f"{int((upper_q - lower_q) * 100)}% prediction interval",
    )
    ax.plot(mu.index, mu.values, colour="#264478", linewidth=1.2,
            label="Forecast mean", alpha=0.9)
    ax.plot(y_actual.index, y_actual.values, colour="black",
            linewidth=1.0, alpha=0.85, label="Actual")

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_crps_over_time(
    crps_series_dict: dict[str, pd.Series],
    rolling_window:   int = 30,
    title:            str = "Rolling mean CRPS over time",
    figsize:          tuple[int, int] = (13, 4),
) -> plt.Figure:
    """
    Plots rolling mean CRPS over time for one or more models. Spikes in
    rolling CRPS reveal periods when the model was particularly uncertain
    or poorly calibrated, which can often be linked to specific atmospheric
    events such as sudden stratospheric warmings or NAO phase transitions.

    Parameters
    ----------
    crps_series_dict:
        Dictionary mapping model name to its per-timestep CRPS Series, as
        returned by ngboost_prob.compute_crps.
    rolling_window:
        Number of days over which to compute the rolling mean.
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

    for model_name, crps_series in crps_series_dict.items():
        rolling = crps_series.rolling(rolling_window, min_periods=rolling_window // 2).mean()
        colour  = None
        if model_name == "NGBoost":
            colour = "#264478"
        ax.plot(rolling.index, rolling.values, label=model_name,
                colour=colour, linewidth=1.1, alpha=0.85)

    ax.set_title(f"{title} ({rolling_window}-day window)", fontsize=12, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("CRPS (°C)")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_cold_probability_vs_outcome(
    cold_probability: pd.Series,
    y_actual_regime:  pd.Series,
    title:            str = "Cold regime probability vs observed outcome",
    figsize:          tuple[int, int] = (13, 4),
) -> plt.Figure:
    """
    Plots the cold regime probability series as a filled area, with observed
    cold regime weeks marked as vertical shaded bands.

    This figure is the primary visual representation of the subseasonal
    forecast product for a non-meteorologist PM. It directly answers the
    question: does the model assign elevated cold probability before cold
    weeks actually occur?

    Parameters
    ----------
    cold_probability:
        Predicted cold regime probability Series, UTC-indexed.
    y_actual_regime:
        Observed weekly regime labels ('cold', 'neutral', 'warm'),
        UTC-indexed.
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

    ax.fill_between(
        cold_probability.index,
        0,
        cold_probability.values,
        alpha=0.35, colour="#264478",
        label="Cold probability",
    )
    ax.plot(cold_probability.index, cold_probability.values,
            colour="#264478", linewidth=1.0, alpha=0.8)

    cold_weeks = y_actual_regime[y_actual_regime == "cold"].index
    for ts in cold_weeks:
        ax.axvspan(ts, ts + pd.Timedelta(days=7), alpha=0.18,
                   colour="#c00000", linewidth=0)

    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], colour="#264478", linewidth=1.4, label="Cold probability"),
        Patch(facecolour="#c00000", alpha=0.3, label="Observed cold week"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right", framealpha=0.8)
    ax.axhline(0.4, colour="black", linewidth=0.7, linestyle=":", alpha=0.5)

    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cold regime probability")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def build_probabilistic_summary_table(
    model_results: dict[str, dict],
) -> pd.DataFrame:
    """
    Assembles a consolidated probabilistic evaluation summary table for
    inclusion in the model comparison notebook.

    Each entry in model_results should contain the keys produced by
    ngboost_prob.run_probabilistic_walk_forward_cv and by the interval
    coverage and sharpness computations in this module.

    Parameters
    ----------
    model_results:
        Dictionary mapping model name to a metric dict containing any
        combination of: 'mean_CRPS', 'CRPSS', 'MAE', 'RMSE',
        'coverage_80pct', 'width_80pct'.

    Returns
    -------
    pd.DataFrame
        One row per model, columns matching the available metric keys.
        Missing values are represented as NaN.
    """
    rows = []
    for model_name, metrics in model_results.items():
        row = {"model": model_name}
        row.update({k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    if "CRPSS" in df.columns:
        df = df.sort_values("CRPSS", ascending=False)

    return df


def compute_sigma_by_season(
    sigma:   pd.Series,
    season_map: Optional[dict[str, list[int]]] = None,
) -> pd.DataFrame:
    """
    Summarises the distribution of predicted forecast uncertainty (sigma)
    by meteorological season.

    A model that correctly represents forecast uncertainty should produce
    higher sigma values in winter (DJF), when synoptic variability is
    greatest, and lower sigma values in summer (JJA), when temperature
    anomalies are smaller and more predictable. If sigma is nearly constant
    across all seasons, the model is not learning anything about the
    time-varying nature of forecast uncertainty.

    Parameters
    ----------
    sigma:
        Predicted standard deviation Series with DatetimeIndex.
    season_map:
        Optional override of the default METEOROLOGICAL_SEASONS mapping.

    Returns
    -------
    pd.DataFrame
        One row per season with columns 'mean_sigma', 'std_sigma',
        'p10_sigma', and 'p90_sigma'.
    """
    if season_map is None:
        season_map = METEOROLOGICAL_SEASONS

    month_to_season = {}
    for season, months in season_map.items():
        for m in months:
            month_to_season[m] = season

    df = pd.DataFrame({
        "sigma":  sigma,
        "season": sigma.index.month.map(month_to_season),
    }).dropna()

    records = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        group = df[df["season"] == season]["sigma"]
        if group.empty:
            continue
        records.append({
            "season":     season,
            "mean_sigma": round(float(group.mean()),             4),
            "std_sigma":  round(float(group.std()),              4),
            "p10_sigma":  round(float(group.quantile(0.10)),     4),
            "p90_sigma":  round(float(group.quantile(0.90)),     4),
            "n_obs":      len(group),
        })

    return pd.DataFrame(records).set_index("season")
