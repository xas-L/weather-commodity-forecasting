# Weather-to-P&L: Probabilistic Temperature and Wind Forecasting for NW European Gas and Power Markets

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

This project develops and evaluates probabilistic short-range temperature forecasts and subseasonal regime classifications for the TTF (Netherlands), German EPEX, and Dutch APX energy markets, using ERA5 reanalysis as ground truth and Open-Meteo NWP output as the operational baseline to post-process. Heating Degree Day forecasts are validated against ENTSO-E realised electricity load, and a subseasonal NAO regime classifier provides week-2 to week-4 directional bias signals grounded in atmospheric teleconnection indices from the NOAA Climate Prediction Centre.

Every result in this project is framed around the question a commodity gas or power trader would ask: what is the probability of a cold or low-wind surprise, how large is it, and what would a desk have done with that view?

The project operates across two forecast horizons that are commercially distinct:

- **Short-term (day-ahead to day+7):** Probabilistic 2m temperature and Heating Degree Day forecasts using LightGBM and NGBoost, evaluated by CRPS, reliability diagrams, and skill score relative to climatology.
- **Subseasonal (week 2 to week 4):** Ternary temperature regime classification (cold / neutral / warm) using XGBoost with isotonic probability calibration, evaluated by Brier score and Brier Skill Score.

---

## Market Context

European natural gas and power prices are among the most weather-sensitive financial instruments in the world. The physical mechanisms driving this sensitivity are specific and tractable:

**Heating Degree Days and gas demand.** Space heating accounts for roughly 40% of European natural gas end-use. Demand is approximately linear in HDD below the base temperature of 15.5°C. A weekly HDD anomaly of 5 degree-days corresponds to a demand increase on the order of 10-15 TWh across the German and Dutch systems combined, which is material relative to typical weekly storage withdrawals during winter.

**Dunkelflaute and gas-for-power demand.** Germany and the Netherlands have installed over 100 GW of wind and solar capacity. When both sources are simultaneously suppressed (wind capacity factor below 10%, solar below 5%), the grid relies disproportionately on gas-fired combined cycle plants. These "dark doldrums" events are forecastable 3-5 days ahead and produce sharp movements in both EPEX day-ahead power prices and TTF intraday prices.

**NAO and subseasonal cold risk.** The North Atlantic Oscillation (NAO) index characterises the meridional pressure gradient between the Azores and Iceland. A persistently negative NAO indicates a weakened or reversed pressure gradient, blocking westerly flow and allowing cold Arctic air to advance southward into NW Europe. Negative NAO regimes have a predictable association with elevated HDD anomalies in Germany and the Netherlands at 2-3 week lead times, providing a subseasonal edge to a gas trading desk positioned on the forward curve.

---

## Data Sources

All data sources used in this project are free and publicly accessible. No proprietary data is required.

| Source | Product | Access | Variables |
|---|---|---|---|
| Copernicus CDS | ERA5 Reanalysis | Free registration | T2m, U10, V10, SSRD, precip, Z500, T850 |
| Open-Meteo | Historical NWP archive | No key required | T2m, wind speed, radiation |
| ENTSO-E Transparency | Actual load, generation, prices | Free registration | MW load, wind, solar, DA prices |
| NOAA CPC | Teleconnection indices | Public FTP | NAO, AO, PNA (daily); ENSO (monthly) |
| Bundesnetzagentur SMARD | German power market | Public download | DA prices, generation mix |

### ERA5 Reanalysis

ERA5, produced by ECMWF, is the primary ground-truth dataset. It assimilates observations from radiosondes, satellites, ships, and surface stations into a physically consistent global atmospheric model at 0.25° horizontal resolution and hourly temporal resolution from 1940 to the present. ERA5 is the standard reference dataset used in peer-reviewed subseasonal forecasting research and in operational NWP post-processing.

Two ERA5 products are used: single-level surface fields (T2m, 10m wind components, surface solar radiation, mean sea-level pressure) and pressure-level fields (geopotential and temperature at 500 hPa and 850 hPa). The pressure-level fields provide the upper-tropospheric circulation diagnostics required for blocking detection and NAO regime characterisation.

Setup requires a free CDS account and the `cdsapi` Python library:

```bash
pip install cdsapi
```

Credentials are placed in `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-uid>:<your-api-key>
```

### Open-Meteo NWP Baseline

Open-Meteo provides free access to operational NWP output (GFS and ECMWF open data) at no cost and without registration. It is used here as the NWP direct baseline: the raw model forecast before statistical post-processing. The ML models in this project are evaluated against this baseline as their primary performance benchmark. A post-processing model that cannot improve on raw NWP output has not justified its complexity.

### ENTSO-E Transparency Platform

ENTSO-E publishes actual grid data for all European transmission system operators at hourly resolution. The primary series used here are German and Dutch electricity load (for validating HDD demand forecasts), wind generation by source (for Dunkelflaute detection and capacity factor computation), and day-ahead power prices (for the commodity linkage signal back-test).

```bash
pip install entsoe-py
```

### NOAA CPC Teleconnection Indices

Daily and monthly teleconnection indices are downloaded from the NOAA CPC public FTP server. These plain-text files require no authentication. The NAO and AO daily indices are primary features in the subseasonal regime classifier. Monthly ENSO (Nino 3.4 SST anomaly) is forward-filled to daily frequency and included as a seasonal modulator of NAO-temperature linkage strength.

---

## Project Structure

```
weather-commodity-forecasting/
├── data/
│   ├── raw/                        # ERA5 NetCDF, ENTSO-E Parquet, CPC text cache
│   ├── processed/                  # Feature-engineered Parquet files, regime fields
│   └── outputs/                    # Model forecasts, evaluation results
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── era5_pipeline.py        # CDS API ingestion, xarray processing, Z500
│   │   ├── entso_pipeline.py       # ENTSO-E API client, load, wind, solar, prices
│   │   └── cpc_indices.py          # NAO/AO/ENSO download, parsing, regime flags
│   ├── features/
│   │   ├── __init__.py
│   │   ├── degree_days.py          # HDD/CDD, climatological anomaly, rolling accumulations
│   │   ├── atmospheric.py          # Z500 anomaly, blocking indices, jet stream, wind shear
│   │   ├── teleconnections.py      # NAO/AO lags, tendency, persistence, phase coupling
│   │   └── dunkelflaute.py         # Capacity factors, event detection, forward probability
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py             # Climatology, persistence, and NWP direct baselines
│   │   ├── lgbm_forecaster.py      # LightGBM, walk-forward CV, Optuna, SHAP attribution
│   │   ├── ngboost_prob.py         # NGBoost probabilistic output, CRPS, reliability
│   │   └── regime_classifier.py    # XGBoost regime classifier, calibration, Brier score
│   └── evaluation/                 # (in progress)
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_eda_atmospheric.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_commodity_linkage.ipynb
├── reports/
│   └── methodology.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Feature Engineering

Features are organised into four meteorologically distinct groups. Every feature has an explicit physical or commercial rationale. If a feature cannot be explained to a non-meteorologist in under 30 seconds, it should not be in the model.

### Heating and Cooling Degree Days

Degree days are the bridge between a temperature forecast and a demand forecast. They are defined relative to a base temperature $T_b$ below which space heating is assumed to be active:

$$\text{HDD}_t = \max(T_b - T_t,\ 0)$$

$$\text{CDD}_t = \max(T_t - T_b,\ 0)$$

where $T_t$ is the daily mean 2m temperature in degrees Celsius and $T_b = 15.5\text{°C}$ is the European gas market standard base temperature.

The HDD anomaly relative to the day-of-year climatological mean is the primary signal used by gas traders to assess demand deviation:

$$\text{HDD}\_{\text{anom},t} = \text{HDD}_t - \overline{\text{HDD}}_{\text{doy}(t)}$$

where $\overline{\text{HDD}}_{\text{doy}(t)}$ is the multi-year mean HDD for that calendar day, computed from training data only to avoid forward-looking bias.

Rolling accumulations are computed over windows of 3, 7, 14, and 30 days. The 7-day rolling HDD sum is the most commercially relevant because it aligns with weekly gas storage reporting cycles and forward contract settlement periods.

### Upper-Atmosphere Synoptic Features

The 500 hPa geopotential height (Z500) is the foundational upper-tropospheric variable. It characterises large-scale atmospheric wave patterns on timescales of 5-30 days. Z500 is derived from the ERA5 pressure-level geopotential field via:

$$Z_{500} = \frac{\Phi_{500}}{g_0}$$

where $\Phi_{500}$ is the geopotential at 500 hPa in $\text{m}^2\,\text{s}^{-2}$ and $g_0 = 9.80665\,\text{m}\,\text{s}^{-2}$ is standard gravity.

The standardised Z500 anomaly is computed by removing the day-of-year climatology:

$$Z'_{500}(x, y, t) = Z_{500}(x, y, t) - \overline{Z}_{500}(x, y, \text{doy}(t))$$

**NAO proxy (Z500-based).** Following Hurrell (1995), the NAO index is derived from the standardised Z500 anomaly difference between the Azores ($37.5\text{°N},\ 25.5\text{°W}$) and Iceland ($65.0\text{°N},\ 22.5\text{°W}$) reference points:

$$\text{NAO}_{\text{Z500}} = \frac{Z'_{500}(\text{Azores}) - Z'_{500}(\text{Iceland})}{\sigma_{\text{train}}}$$

**Greenland blocking index.** The cosine-latitude-weighted area mean Z500 anomaly over the Greenland sector ($60\text{°N}$-$75\text{°N}$, $55\text{°W}$-$15\text{°W}$):

$$B_{\text{GL}} = \frac{\sum_{i} \cos(\phi_i)\, Z'_{500,i}}{\sum_{i} \cos(\phi_i)}$$

A sustained positive $B_{\text{GL}}$ indicates an Omega-block configuration associated with cold air outbreaks over NW Europe.

**Jet stream position and strength.** The latitude of maximum 300 hPa zonal wind within the North Atlantic sector ($40\text{°N}$-$70\text{°N}$, $30\text{°W}$-$10\text{°E}$) and the wind speed at that latitude.

**Vertical wind shear.** Area-mean zonal wind difference between 300 hPa and 850 hPa over NW Europe. Strong positive shear is associated with active baroclinic weather systems; weak or negative shear is associated with blocking.

### Teleconnection Features for Subseasonal Forecasting

At week-2 to week-4 lead times, the atmosphere has lost its deterministic predictability. The residual predictability comes from the slow evolution of large-scale circulation patterns characterised by teleconnection indices.

**Lagged index values.** NAO, AO, and PNA at lags of 7, 10, 14, 21, and 28 days. An NAO value at lag-14 represents the circulation state two weeks before the forecast target, which is within the predictable range for blocking patterns.

**Tendency.** The difference between short-term (5-day) and medium-term (20-day) rolling means of the NAO:

$$\Delta \text{NAO} = \overline{\text{NAO}}_{5\text{d}} - \overline{\text{NAO}}_{20\text{d}}$$

A rapidly falling tendency signals an emerging blocking development, which has greater commercial significance than a static negative value.

**Persistence features.** Consecutive days in the cold ($\text{NAO} < -0.8\sigma$) or warm ($\text{NAO} > +0.8\sigma$) phase, and the rolling frequency of blocking days over windows of 5, 10, and 20 days. Regime persistence is the key physical mechanism that gives subseasonal forecasts their skill: blocking patterns are self-sustaining on 5-20 day timescales.

**Phase coupling.** Interaction features that flag when multiple indices are simultaneously in the cold phase. The combined cold magnitude:

$$C_{\text{cold}} = \text{NAO} + \text{AO} \quad \text{when both} < 0, \text{ else } 0$$

is the most commercially legible single feature: a large negative value indicates that both the NAO and AO are simultaneously in blocking configurations, which is the strongest precursor to cold air outbreaks over Germany and the Netherlands.

### Dunkelflaute Features

The wind and solar capacity factors are:

$$\text{CF}_{\text{wind},t} = \frac{G_{\text{wind},t}}{C_{\text{wind}}} \quad \text{CF}_{\text{solar},t} = \frac{G_{\text{solar},t}}{C_{\text{solar}}}$$

where $G$ is actual generation in MW and $C$ is installed capacity in MW. Capacity factors are used rather than raw MW to remove the upward trend from annual capacity additions and ensure thresholds remain stable across years.

An hour is classified as a Dunkelflaute hour when $\text{CF}_{\text{wind}} < 0.10$ and $\text{CF}_{\text{solar}} < 0.05$ (daytime hours only for solar). Events shorter than 24 consecutive hours are excluded from the catalogue because they do not require meaningful gas dispatch changes.

The forward probability feature is:

$$P_{\text{dunk},t}^{(h)} = \frac{1}{h}\sum_{k=1}^{h} \mathbf{1}[\text{hour } t+k \text{ is Dunkelflaute}]$$

This is the model's prediction target for a given look-ahead horizon $h$ and is only used as a training label, not as a model input, to avoid forward-looking bias.

---

## Models

### Baseline Models

All machine learning models are evaluated against three baselines. A model that does not beat all three baselines does not justify its complexity.

**Climatological baseline.** Predicts the smoothed day-of-year mean computed from training data only. The primary benchmark for skill score computation.

**Persistence baseline.** Predicts tomorrow's value as today's observed value. Competitive for short-range temperature forecasting due to strong day-to-day autocorrelation.

**NWP direct baseline.** The raw Open-Meteo forecast, representing the skill of an uncorrected operational NWP system. The correct benchmark for any post-processing model.

### Short-Term Deterministic Model: LightGBM

LightGBM is trained on the full weather feature matrix using an expanding walk-forward cross-validation scheme. The objective function is L1 loss (mean absolute error), which is more robust than L2 for temperature forecasting because day-to-day temperature distributions are approximately Laplace rather than Gaussian.

Hyperparameters are optimised via Optuna with Tree-structured Parzen Estimator (TPE) sampling, minimising mean walk-forward RMSE over 5 CV folds. The Optuna search covers learning rate, number of leaves, depth, regularisation, and subsampling parameters.

Walk-forward CV uses an expanding training window with $N_{\text{splits}} = 8$ folds and an initial training fraction of 50%. Each fold trains on all data up to the split point and tests on the next contiguous block:

```
Fold 1:  |------ train ------|-- test --|
Fold 2:  |-------- train --------|-- test --|
...
Fold 8:  |-------------- train --------------|-- test --|
```

**SHAP attribution.** TreeExplainer SHAP values are computed for the final fitted model. These provide exact additive feature attributions that allow plain-language explanation of each forecast. Aggregating SHAP values by meteorological group (HDD anomaly features, NAO features, blocking persistence, jet position) produces the higher-level attribution a PM can interpret directly.

### Short-Term Probabilistic Model: NGBoost

NGBoost (Natural Gradient Boosting) fits a full Normal distribution at each forecast point. The predictive mean $\mu_t$ is the point forecast; the predictive standard deviation $\sigma_t$ quantifies forecast uncertainty and is itself a learnable function of the input features.

The Normal distribution parameters are estimated by minimising the negative log-likelihood loss with natural gradient updates, which correct for the curvature of the parameter space.

### Subseasonal Regime Classifier: XGBoost

The regime classifier predicts a ternary temperature regime label for week-2 to week-4 ahead based on current teleconnection indices. The three classes are defined by the standardised weekly temperature anomaly $a_t$:

$$\text{regime}_t = \begin{cases} \text{cold} & a_t < -0.8 \\ \text{warm} & a_t > +0.8 \\ \text{neutral} & \text{otherwise} \end{cases}$$

where $a_t = (T_{\text{week},t} - \overline{T}_{\text{week}}) / \sigma_{\text{train}}$.

Post-fit isotonic calibration is applied using `TimeSeriesSplit` CV to correct the known tendency of gradient boosting models to produce overconfident class probabilities. Standard k-fold calibration is not used because it would leak future information into the calibration fit.

The cold-regime probability is the primary output used by the commodity linkage notebook.

---

## Evaluation Framework

### Deterministic Metrics

Standard error metrics for the short-term point forecast:

- **MAE** (mean absolute error): $\frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|$
- **RMSE** (root mean squared error): $\sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}$

### Skill Score

The deterministic skill score measures improvement over climatology:

$$SS = 1 - \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{climatology}}}$$

A skill score of 0 means no improvement over climatology. A skill score of 0.3 means the model reduces RMSE by 30% relative to predicting the seasonal mean at every time step. All model comparison tables in this project include a `skill_score` column alongside MAE and RMSE.

### Probabilistic Metrics

**CRPS (Continuous Ranked Probability Score).** The primary metric for probabilistic forecast quality:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(x) - \mathbf{1}[x \geq y]\right)^2 dx$$

For a Normal predictive distribution $\mathcal{N}(\mu, \sigma^2)$ this reduces to a closed form (Gneiting and Raftery, 2007):

$$\text{CRPS}(\mathcal{N}(\mu,\sigma^2),\, y) = \sigma\left(\frac{y-\mu}{\sigma}\left(2\Phi\!\left(\frac{y-\mu}{\sigma}\right)-1\right) + 2\phi\!\left(\frac{y-\mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}}\right)$$

where $\Phi$ is the standard Normal CDF and $\phi$ is the standard Normal PDF.

**CRPS Skill Score (CRPSS).** Improvement over a climatological Normal distribution fitted to training data:

$$\text{CRPSS} = 1 - \frac{\overline{\text{CRPS}}_{\text{model}}}{\overline{\text{CRPS}}_{\text{climatology}}}$$

**Reliability (calibration).** For each nominal quantile level $q$, the observed frequency of actual outcomes below the predicted $q$-th quantile should equal $q$. A reliability diagram plots observed frequency against nominal quantile. A perfectly calibrated forecast lies on the diagonal.

**Sharpness.** The mean width of prediction intervals. For the 80% interval:

$$\text{Sharpness}_{80} = \mathbb{E}\left[Q_{0.90} - Q_{0.10}\right]$$

where $Q_p$ denotes the $p$-th quantile of the predictive distribution.

### Probabilistic Classification Metrics

**Brier score** for the cold regime probability $\hat{p}_{\text{cold}}$:

$$\text{BS} = \frac{1}{n}\sum_{t=1}^{n}\left(\hat{p}_{\text{cold},t} - \mathbf{1}[\text{regime}_t = \text{cold}]\right)^2$$

**Brier Skill Score** relative to the climatological cold base rate $p_0$:

$$\text{BSS} = 1 - \frac{\text{BS}_{\text{model}}}{\text{BS}_{\text{reference}}}$$

where $\text{BS}_{\text{reference}} = p_0(1-p_0)^2 + (1-p_0)p_0^2$.

A BSS above 0.1 relative to the climatological prior indicates useful subseasonal skill.

---

## Commodity Linkage

Notebook `05_commodity_linkage.ipynb` answers the question that justifies the project's existence: does weather forecast skill translate into commercial information content for gas and power trading?

The analysis proceeds in three steps:

**Step 1: HDD forecast error and realised demand.** The weekly HDD forecast error (actual minus forecast) is correlated with the week-on-week change in ENTSO-E realised electricity load. A significant positive correlation demonstrates that weather forecast quality directly determines demand forecast quality, quantifying the stakes of each degree Celsius of temperature forecast error in terms of TWh of gas and power demand.

**Step 2: Dunkelflaute event forecastability.** For each catalogued Dunkelflaute event in the test period, the model's forward probability at 24, 48, and 72 hours ahead is recorded. This produces a lead-time forecastability curve showing how far in advance the model provides actionable signal, and at what probability threshold a desk would have traded.

**Step 3: Signal back-test.** A simplified directional back-test measures the information content of the cold-regime probability signal in TTF gas and German day-ahead power prices. The signal is binary: long when the model forecasts a cold regime probability above 40%, flat otherwise. The back-test reports hit rate and annualised information ratio. It is explicitly not a trading strategy. The purpose is to demonstrate that the model's output contains directional information about price movement, quantified in the same terms a PM would use to assess any alpha signal.

---

## How to Read These Results

This section is written for a non-meteorologist reader who wants to interpret the model outputs without engaging with the meteorological detail.

**CRPS of 0.6°C** means that on average, the predictive distribution is displaced from the outcome by approximately 0.6°C, accounting for both the centre of the distribution and its spread. Lower is always better.

**Skill score of 0.28** means the model reduces forecast error by 28% relative to predicting the seasonal climatological mean at every time step. This is the correct comparison for assessing whether the model adds value.

**Cold regime probability of 0.70 at week-2** means the model assigns 70% probability to NW European temperatures being more than 0.8 standard deviations below seasonal norms in two weeks' time. A gas desk can translate this into an expected positive HDD anomaly of approximately 2-4 degree-days per day, corresponding to elevated gas demand above the seasonal baseline.

**Negative NAO blocking event (NAO below -1.5 standard deviations)** is associated with a 1.5 to 2.5°C cold anomaly over Germany and the Netherlands with a 7-14 day lag. This is the forecastable signal that most reliably precedes TTF gas demand spikes in the 1-3 week range.

**80% prediction interval width of 4.2°C** means that on average, the model assigns 80% of its probability mass to a 4.2°C range. A narrower interval means a more confident and commercially useful forecast, provided the calibration is maintained (i.e. 80% of outcomes actually fall within the 80% interval).

---

## Setup and Installation

### Requirements

Python 3.11 or later is required. All dependencies are listed in `requirements.txt`.

```bash
git clone https://github.com/your-username/weather-commodity-forecasting.git
cd weather-commodity-forecasting
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data access credentials

Three credentials are required before the data pipeline can run:

**CDS API key (ERA5).** Register at `https://cds.climate.copernicus.eu`, then create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-uid>:<your-api-key>
```

**ENTSO-E API key.** Register at `https://transparency.entsoe.eu`, then set the environment variable:

```bash
export ENTSO_API_KEY="your-key-here"
```

**CPC indices.** No credentials required. The pipeline downloads directly from NOAA public servers.

### Running the data pipeline

Each pipeline is idempotent: re-running it skips files that already exist on disk. CDS ERA5 downloads can take several hours depending on queue length.

```python
from src.data import run_era5_pipeline, run_entso_pipeline, run_cpc_pipeline
import os

run_era5_pipeline(years=list(range(2018, 2024)))
run_entso_pipeline(api_key=os.environ["ENTSO_API_KEY"])
run_cpc_pipeline()
```

### Running the notebooks

Notebooks are intended to be run sequentially from the project root:

```bash
jupyter lab
```

1. `01_data_pipeline.ipynb`: Data ingestion and quality audit.
2. `02_eda_atmospheric.ipynb`: Z500 composites, HDD climatology, Dunkelflaute event statistics.
3. `03_feature_engineering.ipynb`: Full feature matrix construction and inspection.
4. `04_model_comparison.ipynb`: All models against all baselines, walk-forward CV results.
5. `05_commodity_linkage.ipynb`: HDD error and demand correlation, signal back-test.

---

## Results Summary (TBD)

Results are populated after running the full pipeline. The table structure below is the target output from `04_model_comparison.ipynb`.

(Table creation chore done)

| Model | MAE (°C) | RMSE (°C) | Skill Score | Mean CRPS | CRPSS |
|---|---|---|---|---|---|
| Climatology | | | 0.00 | | |
| Persistence | | | | | |
| NWP Direct | | | | | |
| LightGBM | | | | | |
| NGBoost | | | | | |

| Regime Classifier | Brier Score | BSS | AUC (cold) | Cold F1 |
|---|---|---|---|---|
| Climatological prior | | | | |
| XGBoost + calibration | | | | |

---

## Dependencies

Core dependencies, with minimum versions:

```
cdsapi>=0.6.1
xarray>=2023.1.0
netcdf4>=1.6.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=1.7.0
ngboost>=0.3.12
shap>=0.43.0
optuna>=3.3.0
properscoring>=0.1
scipy>=1.11.0
entsoe-py>=0.5.10
requests>=2.31.0
requests-cache>=1.1.0
retry-requests>=2.0.0
pyarrow>=12.0.0
```

---

## References

Barnston, A. G. and Livezey, R. E. (1987). Classification, seasonality and persistence of low-frequency atmospheric circulation patterns. *Monthly Weather Review*, 115(6), 1083-1126.

Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.

Hurrell, J. W. (1995). Decadal trends in the North Atlantic Oscillation: regional temperatures and precipitation. *Science*, 269(5224), 676-679.

Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.

Duan, T. et al. (2020). NGBoost: Natural gradient boosting for probabilistic prediction. *Proceedings of the 37th International Conference on Machine Learning (ICML)*, PMLR 119.

---

## Licence

This project is licensed under the MIT Licence. See [LICENSE](LICENSE) for details.