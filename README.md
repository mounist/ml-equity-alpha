# ML Equity Return Prediction Pipeline

End-to-end machine learning pipeline for cross-sectional equity return prediction using WRDS data (CRSP, Compustat, IBES). The model predicts next-month stock returns and constructs a decile-sorted long-short portfolio with statistically significant alpha after transaction costs.

## Architecture

```
WRDS Data Ingestion ──► Feature Engineering (28 features) ──► LightGBM Model
       │                        │                                    │
  CRSP / Compustat /     Momentum, Volatility,            Purged Walk-Forward CV
  IBES / FF Factors      Fundamental, Earnings            Optuna Hyperparameter Tuning
                                                                     │
                                                          Decile Long-Short Portfolio
                                                          SHAP Analysis & Validation
```

**Pipeline stages** (run sequentially or individually via `--stage`):

1. **Data** — Pull CRSP monthly stock files, Compustat fundamentals, IBES analyst estimates, and Fama-French factors from WRDS. Results are cached locally as Parquet files.
2. **Features** — Engineer 28 cross-sectional features, winsorize, rank-transform, and merge into a single panel.
3. **Train** — Purged walk-forward cross-validation, Optuna hyperparameter tuning, final LightGBM model training, and out-of-sample prediction.
4. **Validate** — Information coefficient analysis, SHAP feature importance and stability, IC decay, sub-model comparison.
5. **Portfolio** — Decile long-short backtest with transaction costs, Fama-French 4-factor regression, equity curve and performance reporting.

## Key Results (Out-of-Sample: 2019-2024)

| Metric | Value |
|--------|-------|
| OOS Rank IC | 0.024 |
| Net Sharpe Ratio (50bps TC) | 1.44 |
| FF4 Alpha (annualized) | 25.2% (t = 4.11) |

## Feature Categories

The pipeline constructs 28 features across four dimensions, all rank-transformed cross-sectionally each month:

### Momentum
- 12-1 month momentum, 6-1 month momentum, 3-1 month momentum
- Short-term reversal (1-month)
- Industry-relative momentum

### Volatility
- Idiosyncratic volatility (CAPM residual)
- Total volatility, downside volatility
- Maximum daily return, return skewness
- Amihud illiquidity, turnover

### Fundamental
- Book-to-market, earnings yield, sales yield
- Gross profitability (Novy-Marx), ROA, ROE
- Asset growth (1-year), investment-to-assets
- Debt-to-equity, current ratio, accruals

### Earnings
- Standardized unexpected earnings (SUE)
- Earnings revision (mean and breadth)
- Analyst forecast dispersion
- Number of analyst estimates

## Validation Framework

- **Purged Walk-Forward CV**: 14 annual folds with 1-month purge gap between training and validation sets to prevent look-ahead bias
- **Optuna Hyperparameter Tuning**: 15 trials optimizing rank IC across all CV folds
- **SHAP Analysis**: Feature importance, sign consistency with economic priors, and temporal stability of SHAP values
- **IC Decomposition**: Sub-model comparison isolating contributions from price, fundamental, and earnings features
- **Transaction Cost Sensitivity**: Performance evaluation across multiple TC assumptions
- **Factor Attribution**: Fama-French 4-factor regression with Newey-West standard errors (5 lags)

## Setup

### Prerequisites

- Python 3.10+
- WRDS account with access to CRSP, Compustat, and IBES databases
- GPU recommended (LightGBM configured for GPU by default; falls back to CPU)

### Installation

```bash
git clone https://github.com/mounist/ml-equity-alpha.git
cd ml-equity-alpha
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Configuration

Edit `config.py` to set your WRDS username and adjust parameters:

```python
WRDS_USERNAME = "your_username"
START_DATE = "2000-01-01"
TEST_START = "2019-01-01"
END_DATE = "2024-12-31"
```

### Running the Pipeline

```bash
# Run the full pipeline
python main.py --stage all

# Run individual stages
python main.py --stage data        # Pull data from WRDS
python main.py --stage features    # Build feature panel
python main.py --stage train       # Train model with walk-forward CV
python main.py --stage validate    # IC analysis, SHAP, model comparison
python main.py --stage portfolio   # Backtest and performance reporting

# Force re-download from WRDS (clear parquet cache)
python main.py --stage all --refresh
```

## Project Structure

```
ml_equity_alpha/
├── main.py                          # Pipeline orchestrator (stage runner)
├── config.py                        # Global configuration and hyperparameters
├── requirements.txt                 # Python dependencies
├── data/
│   ├── __init__.py
│   └── wrds_loader.py               # WRDS data ingestion (CRSP, Compustat, IBES, FF)
├── features/
│   ├── __init__.py
│   ├── feature_pipeline.py          # Feature panel builder and merge logic
│   ├── price_features.py            # Momentum, volatility, liquidity features
│   ├── fundamental_features.py      # Value, profitability, investment features
│   └── quality_features.py          # SUE, analyst revision, dispersion features
├── models/
│   ├── __init__.py
│   ├── train.py                     # LightGBM training and feature importance
│   ├── predict.py                   # Prediction utilities
│   └── hyperparameter_tuning.py     # Optuna-based hyperparameter optimization
├── validation/
│   ├── __init__.py
│   ├── walk_forward_cv.py           # Purged walk-forward cross-validation
│   ├── ic_analysis.py               # Information coefficient analysis and decay
│   ├── shap_analysis.py             # SHAP values, stability, sign consistency
│   └── factor_regression.py         # Fama-French factor attribution (NW SE)
├── portfolio/
│   ├── __init__.py
│   └── backtest.py                  # Decile long-short backtest with TC
├── visualization/
│   ├── __init__.py
│   └── plots.py                     # Equity curve, IC series, SHAP, decile plots
└── artifacts/                       # Generated outputs (git-ignored)
    ├── data/                        # Cached WRDS parquet files
    ├── results/                     # Model, predictions, statistics (JSON/CSV)
    └── figures/                     # PNG plots
```

## WRDS Data License Notice

This project requires access to the Wharton Research Data Services (WRDS) database. WRDS data is proprietary and subject to licensing restrictions. **No data is included in this repository.** Users must have their own WRDS subscription and credentials to run the data ingestion stage. Please comply with your institution's WRDS data use agreement.
