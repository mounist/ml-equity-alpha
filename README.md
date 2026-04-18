# ML Equity Alpha

Machine-learning pipeline for cross-sectional U.S. equity return prediction — from WRDS data ingestion to a transaction-cost-aware long–short backtest with FF4 attribution.

## Overview

This project predicts next-month stock returns for the liquid U.S. equity universe and turns those predictions into a tradeable decile long–short portfolio. The model ingests CRSP, Compustat, and IBES data via WRDS, engineers 28 cross-sectional signals across momentum, volatility, value/profitability, and earnings-quality dimensions, and trains a LightGBM regressor to rank stocks by expected one-month forward return.

The core methodology follows the standard academic/industry recipe for honest cross-sectional alpha research: (1) cross-sectional rank-normalisation per month so the model learns *relative* signals rather than absolute levels; (2) **purged walk-forward cross-validation** with a one-month gap between train and validation to eliminate look-ahead bias; (3) Optuna hyperparameter search optimised on rank IC across all folds; (4) SHAP-based interpretability with sign-consistency checks against economic priors; and (5) a decile long–short backtest with turnover-based transaction costs, Fama–French 4-factor attribution (Newey–West SEs), and TC sensitivity analysis.

Every reported in-sample metric is computed from out-of-fold walk-forward predictions — no row is ever scored by a model that saw it during training. The final OOS evaluation window (2019-01 through 2024-12) is strictly held out from both feature computation and Optuna tuning.

## Project Structure

```
ml_equity_alpha/
├── main.py                          # Pipeline orchestrator (data → features → train → validate → portfolio)
├── config.py                        # Global paths, dates, CV / Optuna / TC hyperparameters
├── requirements.txt                 # Python dependencies
│
├── data/
│   └── wrds_loader.py               # CRSP / Compustat / IBES / FF queries with parquet caching
│
├── features/
│   ├── feature_pipeline.py          # Universe filters, merge logic, rank-normalisation, forward returns
│   ├── price_features.py            # Momentum (12-1, 6-1, 3-1), reversal, realised vol, drawdown, skew
│   ├── fundamental_features.py      # Value, profitability, investment, accruals, growth
│   └── quality_features.py          # Standardised Unexpected Earnings (SUE) from IBES
│
├── models/
│   ├── train.py                     # LightGBM training + early stopping + feature importance
│   ├── hyperparameter_tuning.py     # Optuna search on walk-forward folds (rank IC objective)
│   └── predict.py                   # Prediction helpers
│
├── validation/
│   ├── walk_forward_cv.py           # Purged expanding-window CV fold generator
│   ├── ic_analysis.py               # Monthly rank IC, ICIR, IC decay, sub-model decomposition
│   ├── shap_analysis.py             # SHAP values, sign consistency, temporal stability
│   └── factor_regression.py         # FF4 alpha regression with Newey–West standard errors
│
├── portfolio/
│   └── backtest.py                  # Decile sorts, turnover, transaction costs, performance stats
│
├── visualization/
│   └── plots.py                     # Equity curve, IC time series, SHAP, decile bar charts
│
└── artifacts/                       # Generated outputs (git-ignored)
    ├── data/                        # Cached WRDS parquet files
    ├── results/                     # Model pickle, predictions, IC/perf JSON/CSV
    └── figures/                     # PNG plots
```

## Key Features

- **Feature engineering** — 28 cross-sectionally rank-normalised features across four blocks: momentum (12-1, 6-1, 3-1, short reversal), risk (realised vol, downside vol, skew, drawdown), value & quality (B/M, earnings yield, gross profitability, ROA/ROE, accruals, asset growth), and earnings surprise (IBES SUE).
- **Universe filtering** — monthly liquidity screen: drop missing `ret`/`prc`, require `|prc| ≥ $5`, and require market cap above the NYSE 20th-percentile breakpoint for that month.
- **Purged walk-forward CV** — expanding window with a 5-year minimum training history, 12-month validation window, annual refit, and a 1-month purge gap to prevent leakage from overlapping forward returns.
- **Optuna hyperparameter tuning** — 15 trials optimising mean rank IC across all CV folds.
- **SHAP interpretability** — mean-|SHAP| importance, temporal stability of importance rankings, and sign-consistency checks that verify learned directional effects match economic priors (momentum positive, short reversal negative, accruals negative, etc.).
- **IC decomposition** — sub-model benchmarks (price-only, fundamental-only, SUE-only) isolate the marginal contribution of each feature block vs. the full LightGBM model.
- **Backtest** — equal-weighted decile long–short, sector-neutral variant (2-digit SIC), monthly turnover computation, and two-sided transaction costs applied at configurable basis-point levels.
- **Performance reporting** — annualised return/vol, Sharpe, Sortino, Calmar, max drawdown, TC sensitivity grid (10 / 25 / 50 bps), and Fama–French 4-factor regression with Newey–West standard errors.

## Data

All inputs come from **Wharton Research Data Services (WRDS)**:

- **CRSP** — monthly stock file (`crsp.msf`) for returns, prices, share codes, exchange codes, and the market benchmark (`crsp.msi`).
- **Compustat** — annual fundamentals (`comp.funda`) linked via `crsp.ccmxpf_linktable`.
- **IBES** — analyst actuals (`ibes.actu_epsus`) and summary statistics (`ibes.statsum_epsus`) for earnings surprise construction.
- **Fama–French** — monthly research factors (`ff.factors_monthly`) for attribution.

No credentials, data dumps, or proprietary files are included in this repository. Users must have their own WRDS subscription. The WRDS username is read from `config.py`; authentication follows the standard `wrds` Python package flow (via `.pgpass` or interactive prompt) — credentials are never committed. All downloaded data is cached locally as Parquet under `artifacts/data/`, which is git-ignored.

## Setup & Usage

### Prerequisites

- Python 3.10+
- WRDS account with CRSP, Compustat, and IBES access
- Optional: CUDA-enabled GPU (LightGBM defaults to `device="gpu"`; override in `config.LGBM_PARAMS` for CPU)

### Installation

```bash
git clone https://github.com/mounist/ml-equity-alpha.git
cd ml-equity-alpha
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit `config.py` to set your WRDS username and (optionally) adjust the date boundaries, CV settings, or transaction-cost assumption.

### Run

```bash
# Full pipeline end-to-end
python main.py --stage all

# Or run any stage individually (stages cache intermediate state to disk)
python main.py --stage data         # Pull / cache WRDS tables
python main.py --stage features     # Build the feature panel
python main.py --stage train        # Walk-forward CV, Optuna, final model fit
python main.py --stage validate     # IC, SHAP, sub-model comparison
python main.py --stage portfolio    # Decile backtest + FF4 attribution

# Force re-download from WRDS (wipes artifacts/data/)
python main.py --stage all --refresh
```

## Results

Out-of-sample evaluation: **2019-01 → 2024-12** (71 months), held out from Optuna tuning and the final model's training set.

### Portfolio performance (decile long–short, equal-weighted)

| Metric                         | Gross     | Net (10 bps one-way) | Net (50 bps one-way) |
| ------------------------------ | --------- | -------------------- | -------------------- |
| Annualised return              | **28.0%** | 26.3%                | 19.4%                |
| Annualised volatility          | 13.4%     | 13.4%                | 13.5%                |
| **Sharpe ratio**               | **2.09**  | **1.96**             | **1.44**             |
| Sortino ratio                  | 6.03      | 5.66                 | 3.98                 |
| Max drawdown                   | −8.7%     | −9.6%                | −14.2%               |
| Avg monthly turnover (top/bot) | —         | 72.4%                | 72.4%                |

Benchmark (CRSP value-weighted market, same OOS window): 16.7% ann. return, Sharpe 0.94, max DD −24.7%.

### Predictive power

| Metric                     | Value  |
| -------------------------- | ------ |
| OOS monthly rank IC (mean) | 0.024  |
| OOS ICIR                   | 0.22   |
| Walk-forward CV mean IC    | 0.032  |

### Fama–French 4-factor attribution (OOS, net)

| Coefficient       | Value  | t-stat (NW, 5 lags) |
| ----------------- | ------ | ------------------- |
| α (monthly)       | 0.0210 | **4.11**            |
| α (annualised)    | 25.2%  | —                   |
| β<sub>MKT</sub>   | −0.083 | −0.90               |
| β<sub>SMB</sub>   | 0.069  | 0.43                |
| β<sub>HML</sub>   | −0.059 | −0.45               |
| β<sub>UMD</sub>   | 0.079  | 0.88                |
| R²                | 0.032  | —                   |

Alpha is highly statistically significant and not explained by market, size, value, or momentum exposures.

### Figures

Plots are regenerated on each run under `artifacts/figures/`:

| File                        | Content                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `equity_curve.png`          | Cumulative gross/net L-S return vs. market benchmark          |
| `decile_returns.png`        | Mean monthly return per prediction decile (monotonicity test) |
| `ic_time_series.png`        | Monthly rank IC with 12-month rolling mean                    |
| `cv_ic_bar.png`             | Mean IC per walk-forward CV fold                              |
| `shap_summary.png`          | SHAP beeswarm — feature effect distribution                   |
| `shap_importance_bar.png`   | Mean \|SHAP\| feature ranking                                 |
| `shap_dependence_top5.png`  | SHAP dependence plots for the top-5 features                  |

### Artifacts (`artifacts/results/`)

- `final_model.pkl` — trained LightGBM booster
- `oof_predictions.parquet` — out-of-fold predictions (honest IS signal)
- `oos_ic_monthly.csv`, `cv_fold_ic.csv` — IC diagnostics
- `portfolio_stats.json`, `portfolio_monthly.parquet` — backtest metrics and monthly P&L
- `tc_sensitivity.json` — performance at 10 / 25 / 50 bps TC
- `ff4_oos.json` — factor regression output
- `feature_importance_gain.csv`, `feature_importance_shap.csv`, `shap_stability.csv` — interpretability
- `ic_comparison.csv` — full model vs. price/fundamental/SUE sub-models
- `best_params.json` — Optuna-selected LightGBM hyperparameters
- `feature_coverage.csv` — per-feature non-null coverage by year

## Tech Stack

- **Python 3.10+**, **pandas**, **NumPy**, **PyArrow**
- **LightGBM** (primary model, GPU-enabled), **XGBoost** (available for sub-model comparison)
- **Optuna** — hyperparameter optimisation
- **SHAP** — model interpretability
- **scikit-learn**, **scipy**, **statsmodels** (Newey–West SEs)
- **matplotlib**, **seaborn** — plotting
- **wrds** — Wharton data access

## License & Data Notice

Code is provided for research and educational purposes. WRDS data is proprietary; **no data, credentials, or cached tables are distributed with this repository**. Users are responsible for complying with their own WRDS data-use agreement.
