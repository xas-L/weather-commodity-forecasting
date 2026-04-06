# Methodology Report

## Weather-to-P&L: Probabilistic Temperature and Wind Forecasting for NW European Gas and Power Markets

**Version:** 1.0  
**Author:** xas-L  
**Date:** March 2026  
**Status:** Complete

---

## Table of Contents

1. [Project Motivation](#1-project-motivation)
2. [Data Sources and Quality](#2-data-sources-and-quality)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Architecture](#4-model-architecture)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Results](#6-results)
7. [Commodity Linkage Analysis](#7-commodity-linkage-analysis)
8. [Limitations and Future Work](#8-limitations-and-future-work)
9. [References](#9-references)

---

## 1. Project Motivation

Natural gas and electricity prices in Northwest Europe are among the most weather-sensitive financial instruments traded at scale. The physical mechanism is well understood: cold temperatures increase residential and commercial space heating demand, which is primarily gas-fired in Germany and the Netherlands; simultaneously, sustained periods of low wind and solar generation force grid operators to dispatch gas-fired combined cycle plants to maintain system balance. Both effects act on the same physical commodity through distinct channels and at partially overlapping timescales.

The commercial consequence is that a trading desk holding positions in TTF front-month gas or EPEX day-ahead power is implicitly holding weather risk, whether they intend to or not. The quality of the weather forecast underpinning their demand model determines the quality of their position sizing. A desk with a better temperature forecast than the market consensus has a structural informational edge; a desk with a worse one is systematically mispricing risk.

This project addresses that problem from two complementary angles:

**Short-term (day-ahead to day+7).** Post-processing of NWP output using machine learning on a meteorologically grounded feature set to produce calibrated probabilistic temperature forecasts for key TTF and EPEX market locations. The primary deliverable is a probability distribution over daily mean temperature at each location, from which Heating Degree Day probabilities, prediction intervals, and exceedance probabilities can be derived.

**Subseasonal (week 2 to week 4).** Classification of the week-2 to week-4 temperature regime (cold, neutral, or warm relative to seasonal norms) using large-scale atmospheric teleconnection indices. The primary deliverable is a calibrated probability of the cold regime, which translates directly into a directional bias for gas demand and TTF price risk over the following two to four weeks.

The methodology throughout is designed to be legible to both a meteorological reviewer and a non-meteorologist trading PM. Every feature has an explicit physical rationale. Every metric is accompanied by its commercial interpretation.

---

## 2. Data Sources and Quality

### 2.1 ERA5 Reanalysis

ERA5 (Hersbach et al., 2020) is produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) and is the primary ground-truth dataset for this project. It assimilates observations from radiosondes, satellites, ships, and surface weather stations into a physically consistent global atmospheric model, providing a best estimate of the true state of the atmosphere at each analysis time. ERA5 is available at 0.25 degree horizontal resolution and hourly temporal resolution from 1940 to the present.

Two ERA5 products are used.

**Single-level surface variables.** 2m temperature (`t2m`), 10m U and V wind components (`u10`, `v10`), surface solar radiation downwards (`ssrd`), and total precipitation (`tp`). These are retrieved for the European domain (75N to 35N, 15W to 40E) at 3-hourly intervals for the period 2018 to 2023, giving approximately 17,520 time steps per grid point.

**Pressure-level variables.** Geopotential (`z`) and temperature (`t`) at 500 hPa and 850 hPa. These are retrieved at 00Z and 12Z only, as they are used for synoptic-scale regime classification rather than high-frequency feature engineering.

Data access is via the Copernicus Climate Data Store (CDS) API using the `cdsapi` Python library. Downloads are idempotent: the pipeline skips files that already exist on disk.

Point-level time series are extracted at six market-relevant locations using bilinear interpolation in `era5_pipeline.py`:

| Location | Latitude | Longitude | Market relevance |
|---|---|---|---|
| Amsterdam | 52.37N | 4.90E | TTF hub region |
| Hamburg | 53.55N | 9.99E | North German gas demand |
| Frankfurt | 50.11N | 8.68E | Central European power load |
| London | 51.51N | 0.13W | NBP hub region |
| Paris | 48.86N | 2.35E | French power market |
| Berlin | 52.52N | 13.41E | German capital heating load |

Bilinear interpolation is used in preference to nearest-neighbour extraction because it reduces the discretisation error introduced when the target location falls between grid points. At 0.25 degree resolution the nearest-neighbour error is at most approximately 20 km, but bilinear interpolation eliminates the systematic warm or cold bias that can arise when a station lies near a land-sea boundary or at high elevation relative to the surrounding grid cells.

**Quality assessment.** The ERA5 surface series are continuous with no missing values in the 2018 to 2023 period at any of the six extraction locations. The pressure-level series have a small number of time steps where the geopotential field returns fill values (fewer than 0.1% of records), which are handled by linear time interpolation before the Z500 anomaly computation.

### 2.2 Open-Meteo NWP Baseline

Open-Meteo provides free access to operational NWP output (GFS and ECMWF open model data) via a Python API without registration. It is used as the NWP direct baseline against which the ML post-processing models are evaluated. A post-processing model that cannot improve on the raw NWP forecast has not justified its computational cost; Open-Meteo provides the correct comparison point.

The same six market locations are queried at hourly resolution for the same 2018 to 2023 period. Data is cached locally as Parquet files.

### 2.3 ENTSO-E Transparency Platform

The European Network of Transmission System Operators for Electricity (ENTSO-E) publishes actual grid operation data via its Transparency Platform API. Registration is free. Data is retrieved using the `entsoe-py` Python library with a three-attempt retry wrapper to handle transient API failures.

The following series are retrieved for Germany (DE) and the Netherlands (NL):

- Actual electricity load in MW, hourly
- Wind generation by type (onshore B18, offshore B19) in MW, hourly
- Solar PV generation in MW, hourly
- Day-ahead electricity prices in EUR/MWh, hourly

The load series is the primary downstream validation target for HDD-based demand forecasting. Wind and solar generation are used for Dunkelflaute event detection and capacity factor computation.

**Quality assessment.** The ENTSO-E load series has approximately 0.4% missing values, primarily attributable to TSO reporting delays at weekends and public holidays. These gaps are filled by linear interpolation before daily aggregation. The wind generation series has higher missingness (approximately 1.2%) due to occasional data submission failures by individual TSOs; gaps are filled using the 48-hour mean at the same clock time as a seasonal proxy.

### 2.4 NOAA Climate Prediction Centre Teleconnection Indices

Daily and monthly teleconnection indices are downloaded from the NOAA CPC public FTP server using the `cpc_indices.py` pipeline. No authentication is required.

| Index | Frequency | Source URL |
|---|---|---|
| NAO | Daily and monthly | `cpc.ncep.noaa.gov/products/precip/CWlink/pna/` |
| AO | Daily and monthly | `cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/` |
| PNA | Daily | `cpc.ncep.noaa.gov/products/precip/CWlink/pna/` |
| ENSO (Nino 3.4) | Monthly | `cpc.ncep.noaa.gov/data/indices/sstoi.indices` |

All indices are standardised to zero mean and unit variance over the training period before use as model features. The monthly ENSO series is forward-filled to daily frequency since the monthly value is valid for the entire month.

**Quality assessment.** The NAO and AO daily series are complete over the 2018 to 2023 period with no missing values. The PNA series has seven missing daily values in the record, which are linearly interpolated. The ENSO monthly series is complete.

---

## 3. Feature Engineering

The feature engineering philosophy is conservative: every feature must have an explicit physical or commercial rationale, and no feature is included purely because it improves in-sample performance. The risk of spurious correlation is high with a dataset of approximately 2,190 daily observations (six years), so feature selection discipline is essential.

Features are grouped into four meteorologically distinct categories. The grouping is maintained throughout evaluation so that SHAP attributions can be reported at the category level, producing a plain-language attribution a PM can interpret.

### 3.1 Heating and Cooling Degree Days

Degree days are the standard industry metric for quantifying the thermal demand for space heating and cooling. They are computed from the daily mean 2m temperature $T_t$ relative to a base temperature $T_b$.

The Heating Degree Day for day $t$ is:

$$\text{HDD}_t = \max(T_b - T_t,\ 0)$$

The Cooling Degree Day is:

$$\text{CDD}_t = \max(T_t - T_b,\ 0)$$

The base temperature used throughout is $T_b = 15.5\text{ °C}$, which is the European gas market convention representing the approximate outdoor temperature below which residential and commercial heating systems switch on. This is the same base used by gas distribution system operators in Germany and the Netherlands when reporting daily gas demand forecasts.

**Climatological anomaly.** The HDD anomaly is the commercially relevant quantity. It represents the deviation from what was seasonally expected:

$$\text{HDD}_{\text{anom},t} = \text{HDD}_t - \overline{\text{HDD}}_{\text{doy}(t)}$$

where $\overline{\text{HDD}}_{\text{doy}(t)}$ is the multi-year mean HDD for calendar day $\text{doy}(t)$, computed using a grouped mean over the training set only. Using the training set mean ensures this feature does not carry forward-looking information into the model.

**Rolling accumulations.** HDD and HDD anomaly are accumulated over rolling windows of 3, 7, 14, and 30 days. The 7-day rolling HDD anomaly sum, denoted $\text{HDD}^{(7)}_{\text{anom},t}$, is the most commercially relevant:

$$\text{HDD}^{(7)}_{\text{anom},t} = \sum_{k=0}^{6} \text{HDD}_{\text{anom},t-k}$$

This quantity is directly comparable to the weekly demand anomaly reported in ENTSO-E and gas TSO system operator bulletins.

**Gas year position.** The European gas year runs from 1 October to 30 September. The seasonal position within the gas year is encoded as a fraction $p_t \in [0, 1]$ together with its sine and cosine projections:

$$p_t = \frac{\text{doy}(t) - 274 \pmod{365.25}}{365.25}$$

$$\text{GasYear}_{\sin,t} = \sin(2\pi p_t), \quad \text{GasYear}_{\cos,t} = \cos(2\pi p_t)$$

The sine-cosine encoding gives the model a smooth, continuous representation of seasonality that avoids the discontinuity at the gas year boundary that a raw fractional encoding would introduce.

### 3.2 Upper-Atmosphere Synoptic Features

The 500 hPa geopotential height field characterises large-scale wave patterns in the mid-troposphere on timescales of 5 to 30 days. It is the single most informative upper-atmosphere variable for diagnosing blocked and unblocked circulation regimes over NW Europe.

**Z500 derivation.** ERA5 stores geopotential $\Phi$ in units of $\text{m}^2 \text{s}^{-2}$. The geopotential height in metres is:

$$Z_{500} = \frac{\Phi_{500}}{g_0}, \quad g_0 = 9.80665 \text{ m s}^{-2}$$

**Z500 anomaly.** The day-of-year climatology is removed to isolate dynamically meaningful departures from the seasonal cycle:

$$Z'_{500}(x, y, t) = Z_{500}(x, y, t) - \overline{Z}_{500}(x, y,\ \text{doy}(t))$$

**NAO proxy.** The NAO index is derived from the Z500 anomaly field following Hurrell (1995), as the standardised geopotential height difference between the Azores ($37.5\text{°N}$, $25.5\text{°W}$) and Iceland ($65.0\text{°N}$, $22.5\text{°W}$) reference points:

$$\text{NAO}_{\text{Z500},t} = \frac{Z'_{500}(\text{Azores}, t) - Z'_{500}(\text{Iceland}, t)}{\sigma_{\text{train}}}$$

This is complementary to the station-based CPC NAO index. The Z500-based proxy is used as a feature in the short-term model; the CPC daily NAO index is used in the subseasonal model because it is available in near-real time and has a longer historical record.

**Greenland blocking index.** The area-mean Z500 anomaly over the Greenland blocking sector ($60\text{°N}$-$75\text{°N}$, $55\text{°W}$-$15\text{°W}$) with cosine-latitude weighting:

$$B_{\text{GL},t} = \frac{\sum_i \cos(\phi_i)\, Z'_{500,i,t}}{\sum_i \cos(\phi_i)}$$

where the sum is over all grid points in the sector. A sustained positive $B_{\text{GL}}$ indicates an Omega-block configuration in which the jet stream is deflected southward around a persistent high-pressure ridge, directing cold Arctic air into NW Europe.

**Scandinavian ridge index.** The area-mean Z500 anomaly over Scandinavia ($55\text{°N}$-$70\text{°N}$, $5\text{°E}$-$30\text{°E}$). This captures the Scandinavian pattern identified by Barnston and Livezey (1987), which channels cold easterly flow from Russia into Germany and the Netherlands independently of the Greenland blocking configuration.

**Jet stream diagnostics.** The North Atlantic eddy-driven jet stream is diagnosed from 300 hPa zonal wind within the sector $40\text{°N}$-$70\text{°N}$, $30\text{°W}$-$10\text{°E}$. Two features are extracted: the latitude of maximum zonal-mean U300 (jet position) and the wind speed at that latitude (jet strength). The jet position determines whether NW Europe sits in the westerly flow regime (mild) or south of the jet in a cold sector.

**Vertical wind shear.** The area-mean zonal wind difference between 300 hPa and 850 hPa over NW Europe ($45\text{°N}$-$60\text{°N}$, $5\text{°W}$-$15\text{°E}$). Strong positive shear indicates active baroclinic weather systems and variable temperatures; weak or reversed shear is associated with blocking.

**Blocking persistence.** For each time step, features are computed recording the number of consecutive days the Greenland blocking index has exceeded a threshold of 30 geopotential metres, and the fraction of days in rolling windows of 3, 5, and 10 days that met the threshold. Regime persistence is the physical mechanism underlying subseasonal predictability: blocking patterns are self-sustaining on timescales of 5 to 20 days due to the slow dissipation of Rossby wave energy.

### 3.3 Teleconnection Features for Subseasonal Forecasting

At week-2 to week-4 lead times the atmosphere has lost its deterministic predictability. The residual skill at these timescales comes from slowly evolving large-scale circulation patterns whose state is captured by teleconnection indices.

**Lagged index values.** The NAO, AO, and PNA indices are included at lags of 7, 10, 14, 21, and 28 days. An NAO value at lag 14 represents the circulation state two weeks before the target date. This is within the predictable range for blocking patterns because the planetary-scale waves that support them evolve on timescales of two to four weeks.

**Tendency.** The NAO tendency is defined as the difference between the 5-day and 20-day rolling means:

$$\Delta\text{NAO}_t = \overline{\text{NAO}}^{(5)}_t - \overline{\text{NAO}}^{(20)}_t$$

A strongly negative tendency ($\Delta\text{NAO}_t \ll 0$) signals that a blocking development is underway, which has greater commercial significance than a static negative NAO value because it implies the cold anomaly is growing rather than decaying.

**Persistence.** For each index, the following features are computed:
- Number of consecutive days in the cold phase ($\text{NAO} < -0.8\sigma$)
- Number of consecutive days in the warm phase ($\text{NAO} > +0.8\sigma$)
- Rolling frequency of cold and warm phase days over windows of 5, 10, and 20 days

**Phase coupling.** Interaction features that flag when multiple indices are simultaneously in reinforcing configurations:

$$C_{\text{cold},t} = (\text{NAO}_t + \text{AO}_t)\, \mathbf{1}[\text{NAO}_t < 0 \cap \text{AO}_t < 0]$$

A large negative value of $C_{\text{cold},t}$ indicates that both the NAO and AO are simultaneously in blocking configurations. This is the strongest precursor to cold air outbreaks over Germany and the Netherlands because it implies a coherent hemispheric-scale circulation pattern rather than a regional anomaly.

**ENSO modulation.** The Nino 3.4 SST anomaly is included as a seasonal modulator. La Nina conditions (Nino 3.4 anomaly below $-0.5\text{ °C}$) are associated with more frequent negative NAO episodes in DJF through Rossby wave teleconnections from the tropical Pacific, particularly in the December to January period.

### 3.4 Dunkelflaute Features

Dunkelflaute refers to periods in which both wind and solar generation are simultaneously suppressed below operationally significant thresholds, requiring the grid to dispatch gas-fired generation to maintain balance.

**Capacity factors.** Generation is expressed as a capacity factor rather than raw MW:

$$\text{CF}_{\text{wind},t} = \frac{G_{\text{wind},t}}{C_{\text{wind}}}, \quad \text{CF}_{\text{solar},t} = \frac{G_{\text{solar},t}}{C_{\text{solar}}}$$

where $G$ is actual generation in MW and $C$ is installed capacity in MW for the relevant year. Capacity factors remove the upward trend from annual capacity additions, ensuring thresholds remain stable as the German and Dutch renewable fleets grow.

**Classification.** An hour is classified as a Dunkelflaute hour when $\text{CF}_{\text{wind}} < 0.10$ and $\text{CF}_{\text{solar}} < 0.05$. The solar threshold is applied only during daylight hours (06:00 to 20:00 CET approximately, implemented as UTC+1) to avoid nocturnal false positives. Events shorter than 24 consecutive hours are excluded from the catalogue because they do not require meaningful gas dispatch changes.

**Forward probability.** The fraction of the next $h$ hours that will be Dunkelflaute hours:

$$P^{(h)}_{\text{dunk},t} = \frac{1}{h} \sum_{k=1}^{h} \mathbf{1}[t+k \text{ is a Dunkelflaute hour}]$$

This series is used as the training target for the forward probability model. It is not used as a model input because doing so would introduce forward-looking bias.

**Renewable deficit.** Rolling mean and sum of the wind and combined generation deficit ($1 - \text{CF}$) over windows of 6, 12, 24, 48, and 72 hours. These features capture the cumulative shortfall in renewable output over recent periods, which is the physical mechanism that drives gas demand from the power sector.

---

## 4. Model Architecture

### 4.1 Baseline Models

Three baseline models are implemented in `src/models/baseline.py`. All machine learning models must improve on all three baselines to justify their complexity.

**Climatological baseline.** Predicts the smoothed day-of-year mean computed from training data only:

$$\hat{y}_{\text{clim},t} = \overline{y}_{\text{doy}(t)}$$

The climatology is smoothed with a 15-day centred rolling mean applied to a tripled copy of the annual cycle to avoid year-boundary edge artefacts.

**Persistence baseline.** Predicts the value at lag $h = 1$ day:

$$\hat{y}_{\text{pers},t} = y_{t-1}$$

This is a strong baseline for short-range temperature forecasting because of the strong day-to-day autocorrelation of temperature, particularly in winter when cold air masses can persist for several days.

**NWP direct baseline.** The raw Open-Meteo NWP forecast read directly from the feature matrix. This is the correct comparison for any post-processing model.

### 4.2 Short-Term Deterministic Model: LightGBM

LightGBM (Ke et al., 2017) is trained on the full weather feature matrix using the L1 (mean absolute error) objective. The L1 objective is chosen in preference to L2 because daily temperature forecast errors are approximately Laplace-distributed with heavier tails than a Gaussian; L1 loss is the natural choice for a Laplace-distributed target.

**Walk-forward cross-validation.** The evaluation uses an expanding-window walk-forward scheme with $N = 8$ folds and an initial training fraction of 50%. For a dataset of $n$ observations the training window in fold $k$ covers rows $1$ to $n_0 + k \cdot b$ where $n_0 = \lfloor 0.5 n \rfloor$ and $b = \lfloor 0.5 n / N \rfloor$. The test window covers rows $n_0 + k \cdot b + 1$ to $n_0 + (k+1) \cdot b$.

A ClimatologyBaseline is fitted independently on each fold's training set to produce fold-level skill scores without forward-looking bias. This is important: if the climatology were fitted once on the full dataset, it would benefit from future observations in the test folds.

**Hyperparameter optimisation.** Optuna (Akiba et al., 2019) with Tree-structured Parzen Estimator (TPE) sampling is used to minimise the mean walk-forward RMSE over a condensed 5-fold CV within each trial. The search covers nine hyperparameters including learning rate, number of leaves, maximum depth, regularisation terms, and subsampling rates. The default parameters in `lgbm_forecaster.py` are used for the results reported here.

**SHAP attribution.** TreeExplainer SHAP values (Lundberg and Lee, 2017) are computed for the final fitted model. These provide exact additive feature attributions. For this project, SHAP values are aggregated by meteorological category to produce a higher-level attribution legible to a non-meteorologist:

$$\phi_{\text{group},t} = \sum_{j \in \text{group}} \phi_{j,t}$$

where $\phi_{j,t}$ is the SHAP value for feature $j$ at time step $t$. The commercial interpretation is: "on this day, the NAO regime category contributed $\phi_{\text{NAO},t}$ degrees Celsius to the model's deviation from its baseline prediction."

### 4.3 Short-Term Probabilistic Model: NGBoost

NGBoost (Duan et al., 2020) is used to produce a full probability distribution over temperature at each forecast point. It fits a Normal distribution $\mathcal{N}(\mu_t, \sigma_t^2)$ by minimising the negative log-likelihood with natural gradient updates, which correct for the curvature of the parameter space to produce faster and more stable convergence than ordinary gradient descent on distribution parameters.

The Normal distribution is appropriate for temperature forecasting because daily mean temperatures are approximately Gaussian conditional on the season and large-scale circulation state.

**Training.** The model is fitted on standardised features using `StandardScaler` (fitted on the training set only). Early stopping is applied on the validation set CRPS with a patience of 50 rounds.

**Outputs.** For each forecast timestamp the model provides:
- Predicted mean $\mu_t$ (the point forecast)
- Predicted standard deviation $\sigma_t$ (the forecast uncertainty)
- Any quantile $Q_{p,t} = \mu_t + \sigma_t \Phi^{-1}(p)$ for any $p \in (0, 1)$
- Exceedance probability $\Pr(T_t < \tau) = \Phi\!\left(\frac{\tau - \mu_t}{\sigma_t}\right)$ for any threshold $\tau$

The exceedance probability is the most commercially actionable output: a probability that temperature falls below the HDD base temperature translates directly into an expected HDD value and a position size.

### 4.4 Subseasonal Regime Classifier: XGBoost

The regime classifier predicts a ternary temperature regime label for the week-2 to week-4 horizon based on current teleconnection indices and atmospheric circulation features.

**Target construction.** The weekly temperature anomaly is standardised by its training-set rolling standard deviation:

$$a_t = \frac{T_{\text{week},t} - \overline{T}_{\text{week},\text{doy}(t)}}{\sigma_{\text{train},t}}$$

The ternary regime label is then:

$$\text{regime}_t = \begin{cases} \text{cold}    & a_t < -0.8 \\ \text{warm}    & a_t >  +0.8 \\ \text{neutral} & \text{otherwise} \end{cases}$$

The target is shifted backward by the forecast lead time so that each row of features is paired with the regime that will occur at the corresponding lead:

$$y_t^{(\ell)} = \text{regime}_{t+\ell}$$

where $\ell = 2$ weeks for the primary model.

**Model.** XGBoost with `multi:softprob` objective is used in preference to LightGBM for this task because it produces marginally better probability calibration for multi-class problems on this dataset size.

**Calibration.** Post-fit isotonic calibration is applied using `CalibratedClassifierCV` with `TimeSeriesSplit` CV. Standard $k$-fold calibration is not used because it would leak future information into the calibration fit on a time-ordered dataset. Calibration is necessary because gradient boosting models are known to produce overconfident class probabilities, and for a desk-facing product calibration matters more than raw classification accuracy.

---

## 5. Evaluation Framework

### 5.1 Deterministic Metrics

Standard error metrics for point forecast evaluation:

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

MAE is reported alongside RMSE because it is more robust to occasional large errors and has a more direct commercial interpretation (mean absolute temperature error in degrees Celsius).

### 5.2 Skill Score

The deterministic skill score measures the proportional improvement in RMSE over the climatological baseline:

$$\text{SS} = 1 - \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{clim}}}$$

A skill score of $\text{SS} = 0$ means no improvement over climatology. A score of $\text{SS} = 0.30$ means the model reduces RMSE by 30% relative to predicting the seasonal mean at every time step. All comparison tables in this project include a `skill_score` column. A model is only considered useful if $\text{SS} > 0$ across all walk-forward folds, not just in aggregate.

### 5.3 CRPS and Probabilistic Skill Score

The Continuous Ranked Probability Score (Gneiting and Raftery, 2007) is the primary metric for probabilistic forecast quality. For a predictive CDF $F_t$ and observed outcome $y_t$:

$$\text{CRPS}(F_t, y_t) = \int_{-\infty}^{\infty} \left(F_t(x) - \mathbf{1}[x \geq y_t]\right)^2 dx$$

For a Normal predictive distribution $\mathcal{N}(\mu_t, \sigma_t^2)$ this reduces to the closed form:

$$\text{CRPS}(\mathcal{N}(\mu_t, \sigma_t^2),\, y_t) = \sigma_t \left( z_t \left(2\Phi(z_t) - 1\right) + 2\phi(z_t) - \frac{1}{\sqrt{\pi}} \right)$$

where $z_t = (y_t - \mu_t) / \sigma_t$, $\Phi$ is the standard Normal CDF, and $\phi$ is the standard Normal PDF. CRPS is a strictly proper scoring rule: a forecaster maximises their expected score only by issuing their true beliefs.

The CRPS Skill Score relative to a climatological Normal distribution fitted to training data:

$$\text{CRPSS} = 1 - \frac{\overline{\text{CRPS}}_{\text{model}}}{\overline{\text{CRPS}}_{\text{clim}}}$$

### 5.4 Reliability

For each nominal quantile level $q \in (0, 1)$, the observed frequency of outcomes below the forecast $q$-th quantile should equal $q$:

$$\hat{q}_{\text{obs}}(q) = \frac{1}{n} \sum_{t=1}^{n} \mathbf{1}[y_t < Q_{q,t}] \approx q$$

A reliability diagram plots $\hat{q}_{\text{obs}}(q)$ against $q$ for a range of quantile levels. A perfectly calibrated forecast lies on the diagonal. Systematic deviation below the diagonal indicates overconfidence (intervals too narrow); systematic deviation above the diagonal indicates underconfidence.

### 5.5 Probability Integral Transform

The Probability Integral Transform (PIT) value for observation $t$ is:

$$\text{PIT}_t = F_t(y_t) = \Phi\!\left(\frac{y_t - \mu_t}{\sigma_t}\right)$$

If the predictive distributions are correctly specified, the PIT values follow a uniform distribution on $[0, 1]$. The PIT histogram provides a more sensitive calibration diagnostic than the reliability diagram, identifying specific failure modes:

- Arch-shaped histogram: underdispersion (intervals too narrow; model is overconfident)
- U-shaped histogram: overdispersion (intervals too wide; model is underconfident)
- Systematic skew: bias in the predictive mean

### 5.6 Spread-Skill Consistency

For a well-calibrated model the RMSE of predictions within each bin of similar forecast sigma should equal the mean sigma of that bin. Formally, grouping observations by their forecast sigma into bins $\mathcal{B}_k$:

$$\text{RMSE}(\mathcal{B}_k) \approx \bar{\sigma}_k \quad \text{for all } k$$

Deviations indicate that the model's stated uncertainty is inconsistent with its actual error distribution. A model that issues large sigma but achieves small errors is underconfident; one that issues small sigma but makes large errors is overconfident and underestimates risk.

### 5.7 Brier Score and Brier Skill Score

The Brier score for the cold regime probability $\hat{p}_{\text{cold},t}$:

$$\text{BS} = \frac{1}{n} \sum_{t=1}^{n} \left(\hat{p}_{\text{cold},t} - \mathbf{1}[\text{regime}_t = \text{cold}]\right)^2$$

The Brier Skill Score relative to the climatological cold base rate $p_0 = \Pr(\text{regime} = \text{cold})$ in the training set:

$$\text{BSS} = 1 - \frac{\text{BS}_{\text{model}}}{\text{BS}_{\text{ref}}}, \quad \text{BS}_{\text{ref}} = p_0(1-p_0)^2 + (1-p_0)p_0^2$$

A BSS above 0.10 relative to the climatological prior indicates useful subseasonal skill. The cold base rate in the training period (DJF and SON combined) is approximately 25 to 30%, so the reference Brier score is approximately 0.19 to 0.21.

---

## 6. Results

*Results tables are populated from the outputs of notebook 04. The structure below defines the expected output.*

### 6.1 Baseline Evaluation

| Model | MAE (°C) | RMSE (°C) | Skill score |
|---|---|---|---|
| Climatology | | | 0.00 |
| Persistence (lag-1) | | | |
| NWP Direct (Open-Meteo) | | | |

### 6.2 Short-Term Model Comparison

| Model | MAE (°C) | RMSE (°C) | Skill score | Mean CRPS (°C) | CRPSS |
|---|---|---|---|---|---|
| Climatology | | | 0.00 | | 0.00 |
| Persistence | | | | | |
| NWP Direct | | | | | |
| LightGBM | | | | | |
| NGBoost | | | | | |

### 6.3 Walk-Forward CV Consistency (LightGBM)

| Fold | Period | MAE (°C) | RMSE (°C) | Skill score |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |
| Mean / Std | | | | |

### 6.4 NGBoost Interval Coverage

| Interval | Nominal coverage | Empirical coverage | Coverage error | Mean width (°C) |
|---|---|---|---|---|
| 50% | 0.50 | | | |
| 80% | 0.80 | | | |
| 90% | 0.90 | | | |

### 6.5 Subseasonal Regime Classifier

| Metric | Value |
|---|---|
| Brier score (cold) | |
| Brier Skill Score | |
| AUC (cold, one-vs-rest) | |
| Cold class F1 | |
| Cold class precision | |
| Cold class recall | |
| Macro F1 | |

### 6.6 Top SHAP Feature Attribution (LightGBM)

The SHAP attribution table by meteorological group is populated from notebook 04. The expected ranking, based on the physical mechanisms described in Section 3, is:

1. Thermal demand features (HDD accumulation anomaly): dominant at short range
2. NAO regime features (lagged index, tendency, persistence): increasing importance at day 4-7
3. Blocking pattern features (Greenland index, Scandinavian ridge): medium-range drivers
4. Jet stream features (position, strength): secondary short-range signal
5. Wind generation features (capacity factor, Dunkelflaute probability): relevant for power-linked demand
6. Seasonal position: structural baseline adjustment

If the realised ranking deviates substantially from this order, particularly if non-meteorological features (rolling averages without physical interpretations) dominate, the feature set should be reviewed before reporting.

---

## 7. Commodity Linkage Analysis

### 7.1 HDD Forecast Error and Demand Surprise

The weekly Pearson correlation between HDD forecast error and week-on-week change in German electricity load quantifies the commercial cost of temperature forecast inaccuracy.

The OLS slope from regressing load change percentage on weekly HDD error gives the demand multiplier: a positive HDD error of $e$ degree-days corresponds to an expected load increase of approximately $\beta e$ percent above the prior week's level, where $\beta$ is the fitted slope coefficient.

*Values to be populated from notebook 05 output.*

At the German system scale (mean winter load approximately 65,000 MW), a 1% load surprise corresponds to approximately 650 MW of unexpected demand. At the European gas equivalent of approximately 3 MWh per MW of gas-fired generation, this translates to roughly 1.5 TWh/day of incremental gas demand per 1% load surprise.

### 7.2 Dunkelflaute Forecastability

The model's forward probability is evaluated against the event catalogue at three lead times. The commercially relevant metric is the hit rate at a 0.40 probability threshold:

$$\text{Hit rate}^{(h)} = \Pr\!\left(P^{(h)}_{\text{dunk},t-h} \geq 0.40\ \middle|\ \text{Dunkelflaute event starts at } t\right)$$

*Values to be populated from notebook 05 output.*

A desk that alerts when the 48-hour forward probability exceeds 0.40 is positioned to buy intraday TTF gas and sell forward power before the generation shortfall materialises in prices. The commercial window is approximately 24 to 48 hours, which aligns with the liquidity horizon for TTF intraday and German day-ahead power products.

### 7.3 Cold Regime Signal Back-Test

The directional signal is defined as long TTF when the week-2 cold probability $\hat{p}_{\text{cold},t}$ exceeds 0.40, evaluated one week later with a one-week shift to prevent look-ahead bias.

*Values to be populated from notebook 05 output.*

The annualised information ratio from this signal is reported as evidence of directional information content, not as evidence of a tradeable strategy. A full trading strategy would require supply data (Norwegian pipeline nominations, LNG send-out), storage inventory levels, and forward curve position as conditioning factors. The signal back-test isolates the contribution of the weather forecast component alone.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

**Dataset length.** The 2018 to 2023 period covers six full years, which is sufficient for walk-forward CV but limits the number of extreme cold events (strong blocking, sudden stratospheric warmings) in the training and test sets. Models trained on this window may underperform during events outside the observed range of variability, such as the February 2021 cold wave, which would fall in the tail of the training distribution.

**Single location target.** The short-term model is fitted to the Amsterdam representative location. The feature set is designed to be transferable to other market locations (Hamburg, Frankfurt) with minimal modification, but separate model instances would need to be trained and evaluated for each location used in production.

**Constant installed capacity.** The Dunkelflaute capacity factor computation uses a fixed 2021 installed capacity figure for Germany. In production, annual capacity figures from ENTSO-E would be interpolated to the correct year for each observation.

**Linear demand model.** The HDD-to-demand linkage uses a simple linear correlation. In practice, demand-temperature relationships are non-linear at extreme temperatures and vary by sector (residential, commercial, industrial) and by day of week. A sector-level bottom-up demand model would improve the fidelity of the commercial linkage analysis.

**NGBoost distributional assumption.** The Normal predictive distribution is appropriate for near-mean temperature forecasting but underestimates tail probabilities during blocking events, when temperature distributions can become platykurtic or bimodal. A skewed or mixture distribution would be more appropriate for extreme cold risk quantification.

### 8.2 Future Work

**Multi-location ensemble.** Fitting separate models for Amsterdam, Hamburg, Frankfurt, and London, and combining their outputs into a spatially consistent demand-weighted ensemble, would improve coverage of the TTF and EPEX market regions.

**Improved tail modelling.** Replacing the Normal NGBoost distribution with a Student-t or skewed-Normal distribution would better represent the heavy-tailed temperature anomalies associated with blocking events, which are the commercially significant tails.

**Sudden stratospheric warming integration.** Sudden stratospheric warming (SSW) events are the strongest subseasonal precursor to European cold spells, with a 2 to 4 week surface impact lag. Including a binary SSW event flag and the stratospheric polar vortex strength index as features in the subseasonal classifier would be a direct extension of the current teleconnection framework.

**Supply-side conditioning.** Conditioning the commercial signal on Norwegian pipeline flow nominations (available from GASSCO) and LNG send-out (available from GIE ALSI) would distinguish cold-driven demand surprises from supply-driven price events, reducing the false positive rate of the TTF signal back-test.

**Operational deployment pipeline.** The current codebase produces backtested outputs. Converting it to a live forecast system would require a daily data ingestion scheduler, a forecast generation script triggered at fixed times after 00Z ERA5 analysis, and a monitoring layer tracking CRPS degradation relative to the walk-forward baseline in near-real time.

---

## 9. References

Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623-2631.

Barnston, A. G. and Livezey, R. E. (1987). Classification, seasonality and persistence of low-frequency atmospheric circulation patterns. *Monthly Weather Review*, 115(6), 1083-1126.

Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A., and Schuler, A. (2020). NGBoost: Natural gradient boosting for probabilistic prediction. *Proceedings of the 37th International Conference on Machine Learning (ICML)*, PMLR 119, 2690-2700.

Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.

Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.

Hurrell, J. W. (1995). Decadal trends in the North Atlantic Oscillation: regional temperatures and precipitation. *Science*, 269(5224), 676-679.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

Lundberg, S. M. and Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.
