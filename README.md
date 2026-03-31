<div align="center">

<img src="https://img.shields.io/badge/Apple_FHI-Forecasting_System-000000?style=for-the-badge" alt="Apple FHI Forecasting System"/>

# Apple Financial Health Index Forecasting

### End-to-End Time Series Pipeline with ARIMAX, LSTM, FinBERT Sentiment, and SHAP Explainability

<br/>

> Built a composite financial health score from scratch across 7 data sources, forecasted it with 4 production-grade models, deployed a live REST API on HuggingFace Spaces, and shipped a full-stack interactive web application on Vercel — with complete protection against data leakage at every stage of the pipeline.

<br/>

[![Live API](https://img.shields.io/badge/Live_API-HuggingFace_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://murtazamajid-apples-fhi-forecasting-analysis.hf.space/predict)
[![Web App](https://img.shields.io/badge/Web_App-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![Notebook](https://img.shields.io/badge/Notebook-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Status](https://img.shields.io/badge/Status-Complete_and_Deployed-brightgreen?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Educational-4B89DC?style=for-the-badge)]()

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14-4B8BBE?style=flat-square)](https://www.statsmodels.org)
[![Optuna](https://img.shields.io/badge/Optuna-4.x-4B89DC?style=flat-square)](https://optuna.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-FF6B6B?style=flat-square)](https://shap.readthedocs.io)
[![HuggingFace](https://img.shields.io/badge/FinBERT-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/ProsusAI/finbert)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Deployment](#live-deployment)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Full Pipeline](#full-pipeline)
- [Financial Health Index](#financial-health-index)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models](#models)
- [Explainability with SHAP](#explainability-with-shap)
- [Model Comparison](#model-comparison)
- [Data Leakage Prevention](#data-leakage-prevention)
- [Sentiment Analysis with FinBERT](#sentiment-analysis-with-finbert)
- [Known Limitation and Honest Science](#known-limitation-and-honest-science)
- [Key Findings](#key-findings)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Project Overview

This project builds a complete, production-grade forecasting system for Apple's financial health. Rather than predicting stock price — a noisy, speculative target dominated by sentiment and market microstructure — it constructs a principled **Financial Health Index (FHI)** from Apple's fundamental accounting ratios, then trains, compares, deploys, and explains four forecasting models against it.

The guiding research question throughout:

> Does adding macroeconomic indicators and NLP-scored news sentiment actually improve forecasts of Apple's financial health, or do Apple's own financial ratios tell the full story?

To answer this, four experiments were run. Two feature sets — Apple's financial ratios alone versus all available features including CPI, crude oil, copper prices, GDP, the Federal Funds Rate, and monthly FinBERT sentiment scores from 1,069 New York Times articles — were crossed with two model classes: ARIMAX (a classical statistical model tuned with Optuna) and LSTM (a sequence neural network with early stopping). All four models were evaluated on the same strictly held-out test period.

The project does not stop at the notebook. The trained models are serialised, served behind a FastAPI endpoint hosted on HuggingFace Spaces, and accessible through a full-stack web application deployed on Vercel, where all four models can be selected and compared interactively.

**The answer to the research question:** Apple's own numbers win, in both model families. Adding macroeconomic data and news sentiment made performance worse in every case.

---

## Live Deployment

This project is fully deployed and publicly accessible — not just a notebook.

| Layer | Technology | Description |
|-------|------------|-------------|
| Model API | FastAPI on HuggingFace Spaces | REST endpoint serving all 4 trained models with /predict, /health, and /models routes |
| Web Application | Lovable + Vercel | Interactive dashboard with 8 sections: Overview, Financial Ratios, Macro and Commodities, FHI Deep Dive, News and Sentiment, Model Forecast, Predict FHI, and Data Explorer |
| Model Storage | HuggingFace Spaces filesystem | `.pkl` ARIMAX models and `.keras` LSTM models, version-controlled and hot-loadable |

**Live API base URL:**
```
https://murtazamajid-apples-fhi-forecasting-analysis.hf.space
```

The web application exposes a dedicated Predict FHI section where users can select any of the four trained models, provide input features, and receive a forecasted FHI value rendered within the interactive dashboard.

---

## Key Results

All four models were evaluated on the same held-out test set covering approximately February 2023 to April 2025 (27 observations). No model saw any part of this data during training or hyperparameter selection.

| Experiment | Model | Feature Set | Test RMSE |
|------------|-------|-------------|-----------|
| 1 | ARIMAX | Financial Ratios Only | 0.0971 |
| 2 | ARIMAX | All Features | 0.1114 |
| 3 | LSTM | Financial Ratios Only | **0.0272** |
| 4 | LSTM | All Features | 0.0396 |

Three patterns emerge immediately. Ratios-only outperforms all-features in both model families. LSTM outperforms ARIMAX in both feature configurations. The performance gap between ratios-only and all-features is larger in ARIMAX than in LSTM, consistent with LSTM having greater capacity to down-weight irrelevant inputs.

The ARIMAX ratios-only result of 0.0971 is treated as the primary trustworthy benchmark. The LSTM's 0.0272, while the lowest absolute error, is partially attributable to a structural data quirk explained in full under [Known Limitation and Honest Science](#known-limitation-and-honest-science).

---

## System Architecture

```
+------------------------------------------------------------------+
|                      DATA LAYER  (7 sources)                     |
|  Apple Ratios  CPI  Crude Oil  Copper  GDP  Fed Funds  NYT News  |
+----------------------------------+-------------------------------+
                                   |
                                   v
+------------------------------------------------------------------+
|                      PIPELINE LAYER                              |
|                                                                  |
|  Load all 7 sources                                              |
|    |                                                             |
|    v                                                             |
|  Clean and standardise each source individually                  |
|    |                                                             |
|    v                                                             |
|  Score 1,069 NYT articles with FinBERT -> monthly average        |
|    |                                                             |
|    v                                                             |
|  Align all sources to 179-row monthly date index                 |
|    |                                                             |
|    v                                                             |
|  TIME-BASED SPLIT  (set once, boundaries never moved)            |
|  Train 70%  /  Validation 15%  /  Test 15%                       |
|    |                                                             |
|    v                                                             |
|  Outlier treatment  (IQR from training data only)                |
|    |                                                             |
|    v                                                             |
|  Scaling  (MinMaxScaler fit on train, applied separately)        |
|    |                                                             |
|    v                                                             |
|  Build FHI -> log transform -> ADF test -> first difference      |
|    |                                                             |
|    v                                                             |
|  Visualisation and EDA                                           |
+----------------------------------+-------------------------------+
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
         +--------------------+       +--------------------+
         |    ARIMAX x 2      |       |    LSTM x 2        |
         |  Optuna 50 trials  |       |  EarlyStopping     |
         |  p search [0, 5]   |       |  Lookback = 12     |
         |  q search [0, 5]   |       |  LSTM(50)+Dense(1) |
         +--------+-----------+       +----------+---------+
                  |                              |
                  +-------------+----------------+
                                |
                                v
                  +-----------------------------+
                  |    SHAP EXPLAINABILITY      |
                  |  Per-feature per-timestep   |
                  |  importance on both LSTMs   |
                  +-------------+---------------+
                                |
                                v
                  +-----------------------------+
                  |    EVALUATION               |
                  |  All 4 models on test set   |
                  |  Same data, same period     |
                  +-------------+---------------+
                                |
                                v
                  +-----------------------------+
                  |    SERIALISATION            |
                  |  .pkl via joblib (ARIMAX)   |
                  |  .keras native (LSTM)       |
                  +-------------+---------------+
                                |
                                v
                  +-----------------------------+
                  |    FASTAPI ENDPOINT         |
                  |  HuggingFace Spaces         |
                  |  /predict /health /models   |
                  +-------------+---------------+
                                |
                                v
                  +-----------------------------+
                  |    WEB APPLICATION          |
                  |  Lovable frontend           |
                  |  Deployed on Vercel         |
                  |  8-section interactive UI   |
                  +-----------------------------+
```

---

## Project Structure

```
Apple-FHI-Forecasting-with-ARIMA-and-LSTM/
|
|-- README.md
|-- requirements.txt
|
|-- Data/
|   |-- Apples Financial Data 2010 - 2025.csv
|   |-- USA CPI 2010 - 2025.csv
|   |-- USA Crude Oil 2010 - 2025.csv
|   |-- USA Copper prices 2010 - 2025.csv
|   |-- USA GDP 2010 - 2025.csv
|   |-- USA Fed Funds 2010 - 2025.csv
|   `-- nyt_apple_related_news.csv              (1,069 articles, FinBERT-scored)
|
|-- Images/
|   |-- Apples_financial_ratios_over_time_2010_-_2025.png
|   |-- Macro_and_commodity_indicators_over_time_2010_-_2025.png
|   |-- Apples_Financial_Health_Index_over_time_2010_-_2025_.png
|   |-- Distribution_of_key_Financial_ratios_and_FHI.png
|   `-- Model_Comparison_RMSE.png
|
|-- Models/
|   |-- arima_ratios.pkl                        (ARIMAX, ratios only)
|   |-- arima_all.pkl                           (ARIMAX, all features)
|   |-- lstm_ratios.keras                       (LSTM, ratios only)
|   `-- lstm_all.keras                          (LSTM, all features)
|
`-- Notebook/
    `-- APPLE_FHI_SUBMISSION.ipynb              (full 15-section pipeline)
```

---

## Data Sources

Seven datasets were collected, cleaned, and merged into a single 179-row monthly panel spanning June 2010 through April 2025.

| Dataset | Source | Native Frequency | Aligned To |
|---------|--------|------------------|------------|
| Apple Financial Ratios (8 metrics) | [Macrotrends](https://www.macrotrends.net/stocks/charts/AAPL/apple) | Quarterly | Monthly via forward fill |
| US Consumer Price Index (CPI) | [FRED — CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | Monthly | Monthly |
| WTI Crude Oil Spot Price | [FRED — WTISPLC](https://fred.stlouisfed.org/series/WTISPLC) | Monthly | Monthly |
| Copper Price (USD / metric ton) | [FRED — PCOPPUSDM](https://fred.stlouisfed.org/series/PCOPPUSDM) | Monthly | Monthly |
| US Real GDP | [FRED — GDP](https://fred.stlouisfed.org/series/GDP) | Quarterly | Monthly via forward fill |
| Federal Funds Effective Rate | [FRED — FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) | Monthly | Monthly |
| NYT Apple-Related News Articles | New York Times API | Daily | Monthly average via FinBERT |

**Alignment notes.** GDP and Apple financial ratios are quarterly. Both were forward-filled to produce monthly rows. Forward fill only — no backward fill was used anywhere in the project. Months with no news coverage were also forward-filled after FinBERT scoring. All 7 datasets share exactly 179 rows after alignment with an identical DatetimeIndex.

---

## Full Pipeline

The pipeline runs in 18 stages. The ordering is deliberate: the train/val/test split is performed before any transformation so that no future information can influence how past data is processed.

```
Stage 1    Load all 7 datasets from disk

Stage 2    Clean each source individually
             Apple     : parse dates, rename columns, filter to project date range
             CPI       : filter to project date range
             Oil       : parse dates, filter to project date range
             Copper    : parse dates, filter to project date range
             GDP       : convert quarterly to monthly with forward fill only
             Fed Funds : parse dates, filter to project date range
             News      : score 1,069 articles with FinBERT, aggregate by month

Stage 3    Verify alignment
             Confirm all 7 sources have 179 rows and an identical DatetimeIndex

Stage 4    Merge into one master DataFrame
             Forward fill any remaining NaN values
             No backward fill used at any point

Stage 5    TIME-BASED SPLIT  (performed here, boundaries never moved)
             Train      : rows 1 to 125    (Jun 2010 to Jan 2021)
             Validation : rows 126 to 152  (Feb 2021 to Apr 2023)
             Test       : rows 153 to 179  (May 2023 to Apr 2025)
             Proportions: 70% / 15% / 15%

Stage 6    Outlier treatment
             Calculate IQR bounds from training rows only
             Apply those bounds to flag outliers in all three splits
             Replace flagged values with NaN, then forward fill
             Applied only to the 8 Apple financial ratio columns

Stage 7    Scale the data
             Fit MinMaxScaler on training rows only
             Transform train, validation, and test splits separately
             Values above 1.0 in val/test splits are expected and correct

Stage 8    Build the Financial Health Index
             Weighted average of 5 scaled ratios
             Log-transform the result
             Run ADF stationarity test: p = 0.76, not stationary
             Apply first differencing
             Rerun ADF test: p = 0.000, stationary
             Final target column: fhi_log_diff

Stage 9    Exploratory data analysis
             Financial ratio trends 2010 to 2025
             Macro and commodity indicator trends
             FHI over time, raw and log-transformed
             Distribution plots for all 8 ratios and the FHI

Stage 10   Train ARIMAX (Ratios Only)
             Optuna study, 50 trials, minimise validation RMSE
             Search space: p in [0, 5], q in [0, 5], d fixed at 1
             Refit best-order model on train plus validation combined
             Evaluate on held-out test set

Stage 11   Train LSTM (Ratios Only)
             12-month lookback window
             Architecture: LSTM(50) -> Dense(1)
             EarlyStopping with patience=10, restore_best_weights=True
             Evaluate on held-out test set

Stage 12   Train ARIMAX (All Features)
             Same Optuna procedure as Stage 10
             Exogenous variables: CPI, Oil, Copper, GDP, Fed Funds, Sentiment
             Evaluate on held-out test set

Stage 13   Train LSTM (All Features)
             Same architecture and training procedure as Stage 11
             All 14 features as input (8 ratios plus 6 macro/sentiment)
             Flattened input dimensionality: 14 features x 12 months = 168
             Evaluate on held-out test set

Stage 14   SHAP explainability on both LSTM models
             Flatten sequences to feature-timestep pairs
             Label each input as FeatureName_t1 through FeatureName_t12
             Generate summary plots and bar charts
             Identify which features and timesteps drive predictions

Stage 15   Model comparison
             All 4 models evaluated on the identical test set
             RMSE bar chart comparing all experiments

Stage 16   Save all models
             ARIMAX: joblib -> .pkl files
             LSTM: Keras native save -> .keras files

Stage 17   FastAPI deployment
             /predict, /health, /models endpoints
             Hosted on HuggingFace Spaces

Stage 18   Web application
             Built in Lovable, deployed on Vercel
             8 interactive sections including live model prediction
```

---

## Financial Health Index

### Motivation

Stock price reflects market sentiment, momentum, liquidity, and factors entirely outside Apple's operational control. A Financial Health Index built from accounting fundamentals is a more honest and interpretable target for a company-level forecasting task. It captures what Apple itself is doing financially, not what the market speculates it will do next quarter.

### Component Ratios and Weights

| Ratio | Weight | Why It Was Chosen |
|-------|--------|-------------------|
| Debt to Equity | 25% | Measures leverage and long-term financial risk |
| Return on Equity | 25% | Measures how efficiently Apple generates profit from shareholder capital |
| Return on Investment | 20% | Measures capital allocation efficiency across all invested capital |
| Return on Assets | 20% | Measures how effectively Apple converts total assets into net profit |
| Current Ratio | 10% | Measures short-term liquidity — whether Apple can cover near-term obligations |

### Formula

```
FHI = (0.25 x Debt_to_Equity_scaled)
    + (0.25 x Return_on_Equity_scaled)
    + (0.20 x Return_on_Investment_scaled)
    + (0.20 x Return_on_Assets_scaled)
    + (0.10 x Current_Ratio_scaled)
```

All five inputs are scaled to [0, 1] using a MinMaxScaler fitted on training data only before the weighted average is computed. This ensures no single ratio dominates due to differences in unit or magnitude.

### Stationarity Treatment

Time series models require a stationary target variable. The raw FHI fails the stationarity test.

```
Raw FHI (0 to 1 scale)
  |
  v  Apply log transform
  |
  v  Augmented Dickey-Fuller test
     H0: series has a unit root (not stationary)
     p-value = 0.76  ->  FAIL, do not reject H0
  |
  v  Apply first differencing
  |
  v  Augmented Dickey-Fuller test
     p-value = 0.000  ->  PASS, reject H0
  |
  v  fhi_log_diff  (final model target)
```

### What the FHI Tells Us About Apple

The FHI remained between 0.20 and 0.45 from 2010 through approximately 2018. It climbed sharply from 2019 onward as Apple accelerated its services revenue and share buyback program, holding above 0.90 from 2021 to the end of the dataset. The index captures precisely the transformation from Apple as a hardware company to Apple as a high-margin platform and services company. The bimodal distribution visible in the histogram confirms that the dataset effectively spans two distinct financial regimes.

---

## Exploratory Data Analysis

### Apple Financial Ratios Over Time (2010–2025)

All 8 financial ratios plotted over the 15-year window. The quarterly reporting structure is clearly visible: each ratio holds constant for 3 months before stepping at the next earnings release. Return on Equity and Return on Investment spike sharply around 2019, reflecting the services acceleration and buyback program. The Current Ratio and Quick Ratio have declined steadily since 2020, consistent with Apple tightening working capital and increasing shareholder returns over holding liquid reserves.

### Macro and Commodity Indicators Over Time (2010–2025)

The 6 external features used in the all-features experiments. CPI shows steady inflation with an unusually sharp acceleration in 2021 and 2022. Crude oil crashed in April 2020 from COVID-19 demand destruction and subsequently recovered. Copper tracks global manufacturing demand and dipped in both 2015 and 2020. GDP contracted sharply in Q2 2020 and bounced back within two quarters. The Federal Funds Rate is the most interesting: it held near zero from 2009 through 2021 and then rose in the fastest hiking cycle in four decades starting in March 2022. Monthly FinBERT sentiment scores are shown in the bottom right panel.

### Financial Health Index Over Time (2010–2025)

The top panel shows the FHI on its natural 0 to 1 scale. The bottom panel shows the log-transformed version. The strong upward trend in the log-transformed FHI makes the case for first differencing visually apparent before running the formal stationarity test. The two-regime structure of Apple's financial history — pre-2019 and post-2019 — is highly visible in the raw FHI panel.

### Distribution of Key Financial Ratios and FHI

Histograms for all 8 ratios and the final FHI. Several ratios are clearly bimodal, with two distinct clusters rather than a single approximately normal distribution. This reflects the two financial eras in Apple's history. The FHI histogram makes this most explicit, with one cluster centred around 0.30 and another around 0.90. Any model trained on this dataset is effectively learning across two fundamentally different operating regimes, which has implications for generalisation.

---

## Models

### ARIMAX

ARIMAX (AutoRegressive Integrated Moving Average with eXogenous inputs) is a classical statistical model for univariate time series with support for external predictor variables. It models the target as a linear combination of its own past values (AR terms), past forecast errors (MA terms), and contemporaneous exogenous covariates. The model operates on the first-differenced log FHI series, which satisfies the stationarity requirement.

Hyperparameters were selected automatically using Optuna, a modern Bayesian hyperparameter optimisation framework, across 50 trials. The objective function was validation RMSE. After the best order was identified, the model was refit on the combined train and validation data before evaluation on the test set.

| Parameter | Value |
|-----------|-------|
| Integration order d | 1 (first difference, confirmed by ADF test) |
| AR order p | Search space [0, 5] via Optuna |
| MA order q | Search space [0, 5] via Optuna |
| Optimisation trials | 50 |
| Selection metric | Validation RMSE |
| Final training data | Train + Validation combined |
| Evaluation | Test set only |
| Serialisation | `.pkl` via joblib |

### LSTM

LSTM (Long Short-Term Memory) is a recurrent neural network architecture specifically designed to model sequential data with long-range dependencies. The cell state and gating mechanism allow the model to selectively retain or forget information across many timesteps, making it well-suited to time series forecasting. The model receives a 12-month sliding window as input and produces a single-step prediction for month 13.

| Parameter | Value |
|-----------|-------|
| Lookback window | 12 months |
| Architecture | LSTM(50 units) -> Dense(1) |
| LSTM activation | Tanh (default) |
| Optimiser | Adam |
| Loss function | Mean Squared Error |
| Maximum epochs | 50 |
| Batch size | 32 |
| Early stopping patience | 10 epochs |
| Weight restoration | Best validation weights restored on stop |
| Input shape | (batch, 12, num_features) |
| Output | Scalar fhi_log_diff prediction |
| Serialisation | `.keras` (native Keras format) |

The LSTM input sequences are shaped as `(batch_size, 12, num_features)`, where num_features is 8 for the ratios-only model and 14 for the all-features model. Early stopping prevents overfitting and ensures the saved weights correspond to the epoch with the lowest validation loss, not the final epoch.

---

## Explainability with SHAP

SHAP (SHapley Additive exPlanations) was applied to both LSTM models after training. SHAP computes each feature's average marginal contribution to each individual prediction using a game-theoretic framework. The key properties of SHAP — local accuracy, consistency, and missingness — make the resulting attributions more trustworthy than simpler permutation-based importance measures.

Because the LSTM input is a 3D tensor of shape `(samples, 12 timesteps, num_features)`, the sequences were flattened to a 2D matrix before SHAP analysis. Each feature at each timestep was assigned a unique label in the format `FeatureName_t1` through `FeatureName_t12`, where t1 is the most recent month and t12 is the month 12 periods ago. This naming convention reveals two independent dimensions of importance:

1. **Feature-level importance** — which financial ratios or macro indicators have the highest overall influence on predictions
2. **Timestep-level importance** — whether recent observations (t1, t2) drive predictions more than distant ones (t11, t12), or vice versa

For example, if `Return_on_Equity_t1` has a high SHAP value while `Return_on_Equity_t12` is near zero, the model is primarily using the most recent ROE reading rather than year-ago ROE. This level of temporal interpretability is rarely demonstrated in LSTM-based financial forecasting work in the public domain.

SHAP summary plots and bar plots for both LSTM experiments are available in Section 15 of the notebook.

---

## Model Comparison

All four models were evaluated on the same held-out test set. The test set covers approximately May 2023 through April 2025 and contains 27 observations. It was never used during training, validation, or hyperparameter selection for any of the four models.

| Experiment | Model | Feature Set | Test RMSE | Rank |
|------------|-------|-------------|-----------|------|
| 1 | ARIMAX | Ratios Only | 0.0971 | 3rd |
| 2 | ARIMAX | All Features | 0.1114 | 4th |
| 3 | LSTM | Ratios Only | 0.0272 | 1st |
| 4 | LSTM | All Features | 0.0396 | 2nd |

**Pattern 1 — Feature set consistently matters more than model family.**
Ratios-only beats all-features in both ARIMAX (+14.7% RMSE reduction) and LSTM (+45.6% RMSE reduction). The hypothesis that macro and sentiment data adds predictive value is rejected.

**Pattern 2 — LSTM outperforms ARIMAX in both configurations.**
LSTM's lower error reflects both greater model capacity and the quarterly repetition effect described in the Known Limitation section.

**Pattern 3 — The feature-set gap is asymmetric across model families.**
ARIMAX degrades by 14.7% when all features are added. LSTM degrades by 45.6%. This is counterintuitive: a neural network should in principle be better at ignoring irrelevant features. One explanation is that 14 features times 12 timesteps produces 168 input dimensions for 124 training samples, a ratio that makes it difficult for the LSTM to reliably identify which dimensions carry signal.

**The recommended benchmark.** ARIMAX ratios-only at 0.0971 is the figure to weight most heavily. It does not benefit from the quarterly repetition effect to the same degree, it has no leakage of any kind, and it represents a disciplined classical approach to the forecasting problem.

---

## Data Leakage Prevention

Data leakage is when information from the future is allowed, directly or indirectly, to influence how past data is processed or how models are trained. It produces models that appear stronger than they actually are on unseen data. Preventing leakage was treated as a first-class engineering concern throughout this project, not an afterthought.

Four distinct leakage vectors were identified, documented, and fixed.

| Leakage Source | The Problem | The Fix Applied |
|----------------|-------------|-----------------|
| Outlier IQR calculation | IQR was computed on all 179 rows. This means values from 2024 and 2025 influenced how 2010 and 2011 outliers were defined and clipped — direct future-to-past contamination | IQR bounds are now computed exclusively from the 125 training rows. Those bounds are then applied identically to train, validation, and test splits |
| MinMaxScaler fitting | The scaler was learning its min and max parameters from all 179 rows, meaning it saw the global range including test data before any model training began | Scaler is fitted on the 125 training rows only. Validation and test sets are transformed using those training parameters. If test values exceed the training range, they scale above 1.0 — which is correct |
| Split timing | The train/val/test split was being reapplied after certain transformations had already been computed on the full dataset, effectively shifting the split boundaries and contaminating the training set with future-period processed values | The split is now performed at Stage 5, immediately after merging the raw sources. The boundary indices are never recalculated or moved at any subsequent stage |
| NaN filling direction | `bfill` (backward fill) was used in several pipeline steps. Backward fill fills a missing value at time t by pulling the value from time t+1 — a direct mechanism for future information to influence past observations | All NaN filling throughout the entire pipeline now uses `ffill` (forward fill) exclusively. A missing value at time t is filled from time t-1 |

**A note on the scaler side effect.** After the scaler fix, some values in the validation and test splits scale to values above 1.0. This is intentional and correct, not an error. It means those splits contain values that exceeded the maximum observed in the training period. In real deployment, any model trained in 2021 and used in 2024 will routinely encounter out-of-range inputs. The pipeline now handles this exactly as a production system would.

**On suspiciously good results.** An early version of this project produced an LSTM test RMSE of 0.015. Rather than accepting that result and moving on, it triggered an investigation. The investigation found two remaining leakage sources and identified the quarterly repetition effect as a third contributing factor. After all fixes were applied, the RMSE rose to 0.0272 — a higher number, but an honest one. This decision to investigate rather than publish is documented explicitly because it reflects how rigorous applied data science should work.

---

## Sentiment Analysis with FinBERT

### Why FinBERT

General-purpose sentiment models trained on social media, product reviews, or movie criticism do not generalise reliably to financial language. FinBERT (Araci, 2019) is a BERT-base model fine-tuned specifically on financial communications — earnings call transcripts, analyst reports, and financial news. It outputs calibrated probability distributions across three classes: Positive, Neutral, and Negative. Unlike lexicon-based approaches, it handles negation, financial idiom, and context-dependent tone correctly.

### Data Collection

1,069 New York Times articles containing the keyword "Apple" were retrieved via the NYT Article Search API, covering June 2010 through April 2025. Each article contributes a headline and a snippet — the lead sentence or editorial abstract provided by the Times API.

### Scoring Pipeline

```
Headline + Snippet from each article
  |
  v  Strip HTML tags, normalise whitespace
  |
  v  Tokenise with FinBERT tokeniser (max 512 tokens, truncate if longer)
  |
  v  Forward pass through ProsusAI/finbert
  |
  v  Softmax output: P(Negative), P(Neutral), P(Positive)
  |
  v  Assign label of winning class and its probability as the article score
  |
  v  Group all articles by calendar month
  |
  v  Compute mean score within each month
  |
  v  Forward-fill months where no articles were retrieved
```

### Sentiment Distribution

| Class | Article Count | Percentage |
|-------|---------------|------------|
| Positive | 601 | 56.2% |
| Neutral | 351 | 32.8% |
| Negative | 117 | 11.0% |

Over half of the 1,069 articles scored positive. This is consistent with Apple's product reception, financial performance, and brand perception over the 2010–2025 period. The negative 11% is concentrated around specific events: the 2013 earnings miss relative to elevated expectations, the 2016 iPhone 6s cycle deceleration, and concerns during the 2018 to 2019 US-China trade dispute.

### Hardware Recommendation

Running FinBERT on 1,069 articles requires approximately 20 minutes on CPU and under 5 minutes on a T4 GPU in Google Colab. Switching to a GPU runtime before executing the sentiment section of the notebook is strongly recommended.

---

## Known Limitation and Honest Science

### The Quarterly Repetition Problem

Apple reports financial results four times per year. The Macrotrends dataset reflects this structure: all 8 financial ratios hold the same value for all 3 months within a given quarter, then step to new values at the next earnings release.

This means that across the 179 monthly rows in the dataset, 67% of all rows show zero change in every financial ratio simultaneously. The value at month t is identical to the value at month t-1 for the majority of observations.

The LSTM learns to exploit this pattern. Its objective is to minimise mean squared error on the training set. A model that learns to predict "current month will look like last month" gets rewarded with a low error score for 67% of all training examples without learning any genuine financial dynamics. The model is effectively doing pattern recognition on repetition rather than forecasting a changing quantity.

ARIMAX handles this differently. Its AR and MA terms model the differenced, stationary series and are less sensitive to the within-quarter flat segments. This is the primary reason ARIMAX at 0.0971 is treated as the more trustworthy benchmark, even though it has a higher absolute error.

### What This Means in Practice

The LSTM's 0.0272 comes from a valid held-out test set with no data leakage. The number is real. But it reflects, in part, the model's ability to recognise quarterly repetition rather than its ability to forecast financial health. A model that scores well partly because it has memorised that Apple's ratios do not change most months is not equivalent to a model that can correctly forecast when the next change will occur and in which direction.

This limitation is structural — it comes from the frequency mismatch between the reporting schedule (quarterly) and the analysis horizon (monthly). It is not a mistake in the pipeline. The correct remediation would be to work with the 60 true quarterly observations directly — eliminating the within-quarter repetition entirely — or to switch to a target variable that changes at higher frequency, such as daily closing stock price or weekly analyst earnings estimate revisions.

---

## Key Findings

**1. Apple's own financial ratios are the strongest predictors of Apple's financial health.**

Adding six external datasets — CPI, crude oil, copper, GDP, the Federal Funds Rate, and 1,069 FinBERT-scored news articles — made both models perform measurably worse. The improvement hypothesis for external features is rejected across both model families and both evaluation metrics examined.

**2. More features makes things worse on small samples.**

With 124 training observations and up to 168 input dimensions (14 features times a 12-month lookback), the all-features LSTM does not have enough data to reliably distinguish signal from noise. ARIMAX shows the same pattern: adding 6 exogenous variables raised the RMSE from 0.0971 to 0.1114. Feature selection and dimensionality are more important than feature richness at this sample size.

**3. Data leakage prevention is harder and more consequential than most practitioners assume.**

Four separate leakage vectors were identified and required four separate engineering fixes. Two of them — the scaler fitting issue and the split timing issue — appear in published financial ML papers with some frequency. The difference between a leaky RMSE and a clean one was substantial.

**4. A classical model from the 1970s is still competitive on well-structured financial data.**

ARIMAX with ratios-only (0.0971) is within striking distance of LSTM with ratios-only (0.0272) when the quarterly repetition effect is accounted for. On a small sample with limited variation and a quarterly step structure, a well-specified classical model with appropriate differencing and Optuna-tuned AR/MA orders performs comparably to a 50-unit neural network. Model complexity should be driven by data characteristics, not by the assumption that more expressive models always win.

**5. Questioning suspiciously good results is methodology, not pessimism.**

An RMSE of 0.015 in an early version triggered an investigation rather than a publication. That investigation led to the identification of two leakage sources and a deeper understanding of the quarterly repetition problem. Scientific honesty produced better science. The final numbers are less impressive in absolute terms but they are defensible.

**6. Deployment is part of the work.**

Training, evaluating, and documenting four models is one part of the problem. Serialising them correctly, wrapping them in a typed REST API with health monitoring and model listing, and surfacing them through a usable web interface with 8 interactive sections are the other parts. All layers are present and publicly accessible.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google account for Google Colab, or a local environment with at least 8 GB of RAM
- GPU access is optional but strongly recommended for the FinBERT scoring step

### Option A — Google Colab (Recommended)

Google Colab provides a free GPU runtime and avoids local dependency conflicts. This is the recommended environment for running the full pipeline including FinBERT.

```
1. Upload all files from the Data/ folder to your Google Drive
2. Open Notebook/APPLE_FHI_SUBMISSION.ipynb in Google Colab
3. In Section 2 of the notebook, update the file path variables to match your Drive location
4. Switch to a GPU runtime via Runtime > Change runtime type > T4 GPU
5. Run all cells from top to bottom in order without skipping any section
```

The FinBERT step in Section 7 takes approximately 5 minutes on GPU and 20 minutes on CPU. Every other section runs in under one minute.

### Option B — Local Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/Apple-FHI-Forecasting-with-ARIMA-and-LSTM.git
cd Apple-FHI-Forecasting-with-ARIMA-and-LSTM

# Install all dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Notebook/APPLE_FHI_SUBMISSION.ipynb
```

### Installing Dependencies Individually

```bash
# Core data and analysis
pip install pandas numpy matplotlib seaborn plotly

# Machine learning and statistical modelling
pip install scikit-learn statsmodels

# Deep learning
pip install tensorflow keras

# Hyperparameter optimisation and explainability
pip install optuna shap

# NLP and sentiment analysis
pip install transformers torch

# API server
pip install fastapi uvicorn

# Model serialisation
pip install joblib
```

### Loading the Saved Models

All four trained models are stored in the `Models/` directory and can be loaded independently of the full training pipeline for inference.

```python
import joblib
from tensorflow.keras.models import load_model

# ARIMAX — ratios only (Experiment 1)
arima_ratios = joblib.load('Models/arima_ratios.pkl')

# ARIMAX — all features (Experiment 2)
arima_all = joblib.load('Models/arima_all.pkl')

# LSTM — ratios only (Experiment 3)
lstm_ratios = load_model('Models/lstm_ratios.keras')

# LSTM — all features (Experiment 4)
lstm_all = load_model('Models/lstm_all.keras')
```

ARIMAX models use statsmodels internals serialised with joblib and can be used with the standard statsmodels predict interface. LSTM models are saved in the native Keras format and load with full architecture, weights, and compilation state. No recompilation is required for inference-only use.

---

## API Reference

The prediction API is live and publicly accessible on HuggingFace Spaces.

**Base URL:**
```
https://murtazamajid-apples-fhi-forecasting-analysis.hf.space
```

### GET /health

Returns the current service status and a list of all loaded models.

```bash
curl https://murtazamajid-apples-fhi-forecasting-analysis.hf.space/health
```

Example response:

```json
{
  "status": "ok",
  "models_loaded": ["arima_ratios", "arima_all", "lstm_ratios", "lstm_all"]
}
```

### GET /models

Returns metadata for all available models including their required feature sets and expected input dimensions.

```bash
curl https://murtazamajid-apples-fhi-forecasting-analysis.hf.space/models
```

### POST /predict

Submits a prediction request and returns a forecasted FHI value.

```bash
curl -X POST "https://murtazamajid-apples-fhi-forecasting-analysis.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lstm_ratios",
    "features": [0.82, 0.74, 0.91, 0.68, 0.55, 0.77, 0.83, 0.61]
  }'
```

**Request body schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | One of `arima_ratios`, `arima_all`, `lstm_ratios`, `lstm_all` |
| `features` | array of floats | Yes | Scaled input features. Ratios-only models expect 8 values; all-features models expect 14 |

**Response schema:**

```json
{
  "model": "lstm_ratios",
  "predicted_fhi_log_diff": -0.0031,
  "input_features_received": 8
}
```

The predicted value is returned in `fhi_log_diff` space — the first-differenced log of the FHI. To convert back to the original FHI scale, apply the inverse difference using the last known FHI value as the base, then apply the inverse log transform.

---

## Future Work

| Direction | Rationale |
|-----------|-----------|
| Switch to a daily or weekly target variable | The most direct fix for the quarterly repetition problem. Daily stock-implied volatility or weekly analyst estimate revisions change at a frequency that eliminates the within-quarter flat segment issue entirely |
| Introduce lagged macro features at 3 to 6 month lags | Current-period macro values are unlikely to be predictive because they are contemporaneous with Apple's own reported ratios. Lagged Federal Funds Rate or lagged CPI would more plausibly represent information available before Apple's quarterly reporting |
| Replace LSTM with Temporal Fusion Transformer | TFT is purpose-built for multi-horizon forecasting with heterogeneous input types. It produces interpretable attention weights natively, which would complement the SHAP analysis already present |
| Expand news coverage to Reuters and Bloomberg | NYT coverage is relatively thin in 2010 through 2013. Adding two wire services would improve sentiment signal consistency across the full 15-year window and reduce the number of months requiring forward-fill |
| Add a naive last-observation baseline | Without a benchmark that predicts "next month equals this month," it is impossible to contextualise whether 0.0971 is a meaningful improvement over doing nothing. Every RMSE table should include this baseline |
| Rebuild as a true quarterly pipeline | Strip the within-quarter repeated rows and work with the 60 actual earnings observations. This reduces sample size but eliminates the structural repetition problem and produces genuine month-over-month variation in every row |
| Extend SHAP analysis to the ARIMAX models | Implement permutation importance or coefficient analysis for the ARIMAX exogenous terms so all four models can be compared on interpretability alongside accuracy |

---

## References

Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv preprint arXiv:1908.10063.* https://arxiv.org/abs/1908.10063

Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 2623–2631). https://arxiv.org/abs/1907.10902

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control.* 5th edition. John Wiley and Sons.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805.* https://arxiv.org/abs/1810.04805

Dickey, D. A. and Fuller, W. A. (1979). Distribution of the Estimators for Autoregressive Time Series With a Unit Root. *Journal of the American Statistical Association,* 74(366), 427–431.

Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation,* 9(8), 1735–1780.

Lim, B., Arik, S. O., Loeff, N., and Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting,* 37(4), 1748–1764.

Lundberg, S. M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017).* https://arxiv.org/abs/1705.07874

---

## License

This project was built for learning and portfolio purposes and is not intended for commercial use.

Financial data sourced from [Macrotrends](https://www.macrotrends.net) and the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org) maintained by the Federal Reserve Bank of St. Louis. News data collected via the [New York Times Article Search API](https://developer.nytimes.com). Sentiment scoring performed using [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) available on HuggingFace.

---

<div align="center">

Built end-to-end. From 7 raw data sources to a live deployed API and interactive web application.
179 monthly observations. 4 models. Zero data leakage.

</div>
