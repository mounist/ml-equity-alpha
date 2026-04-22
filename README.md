# ML Equity Alpha

A cross-sectional machine-learning pipeline for U.S. equity return prediction that ingests CRSP, Compustat, and IBES data from WRDS, engineers 28 monthly signals across momentum, volatility, value/profitability, and earnings-quality dimensions, and trains a LightGBM regressor to rank stocks by expected one-month forward return. The resulting decile long–short portfolio produces an out-of-sample gross Sharpe of **2.09** (2019-01 to 2024-12, 71 months) and a 50 bps one-way TC stress Sharpe of **1.44**; the 10 bps net Sharpe sits between the two at 1.96. The OOS FF4 alpha is approximately **27.0% annualized gross** (monthly α = 0.0225, t = 4.40) and **25.2% annualized net of 10 bps** (monthly α = 0.0210, t = 4.11). Full in-sample and out-of-sample performance at 10 / 25 / 50 bps transaction-cost assumptions is reported below.

## Methodology principles

The universe is filtered each month to the liquid U.S. equity cross-section — price at least \$5 and market capitalization above the NYSE 20th-percentile breakpoint — so that reported performance is not mechanically inflated by micro-cap noise. Features are constructed across four blocks (momentum 12-1 / 6-1 / 3-1 plus short reversal; risk via realized and downside vol, max drawdown, and skew; value and profitability via B/M, earnings yield, gross profitability, ROA/ROE, accruals, and asset growth; earnings surprise via IBES standardized unexpected earnings) and cross-sectionally rank-normalized each month so the model learns relative signals rather than absolute levels.

Cross-validation is a purged walk-forward scheme with a five-year minimum training history, a twelve-month validation window, annual refit, and a one-month purge gap that eliminates look-ahead from overlapping forward returns. Optuna tunes LightGBM hyperparameters against mean rank IC across folds, and every in-sample metric reported below is computed from out-of-fold predictions — no row is ever scored by a model that saw it during training. The 2019-01 through 2024-12 OOS window is strictly held out from feature construction, CV, and tuning. SHAP-based feature importance is used not only for ranking but for sign-consistency checks against economic priors (momentum positive, short reversal negative, accruals negative, and so on), and the decile long–short backtest applies two-sided transaction costs at 10, 25, and 50 bps one-way, with FF4 attribution reported using Newey–West standard errors at five lags.

## Project structure

```
data/           WRDS loaders for CRSP, Compustat, IBES, and FF factors
features/       Cross-sectional signal construction (price, fundamental, quality blocks)
models/         LightGBM training and Optuna hyperparameter search
validation/     Walk-forward CV, IC diagnostics, SHAP analysis, FF4 regression
portfolio/      Decile sort, turnover, transaction-cost engine, performance stats
visualization/  Equity curves, IC time series, SHAP and decile bar plots
```

## Headline results

| Metric | IS gross | IS 10bps | IS 25bps | IS 50bps | OOS gross | OOS 10bps | OOS 25bps | OOS 50bps |
|---|---|---|---|---|---|---|---|---|
| Ann. return | 20.9% | 19.3% | 17.0% | 13.0% | **28.0%** | 26.3% | 23.7% | **19.4%** |
| Ann. vol | 10.6% | 10.6% | 10.6% | 10.6% | 13.4% | 13.4% | 13.4% | 13.5% |
| Sharpe | 1.98 | 1.83 | 1.60 | 1.23 | **2.09** | 1.96 | 1.76 | **1.44** |
| Max drawdown | −15.4% | −17.4% | −20.3% | −25.0% | −8.7% | −9.6% | −11.0% | −14.2% |
| FF4 α (monthly) | N/A | N/A | N/A | N/A | 0.0225 | 0.0210 | N/A | N/A |
| FF4 α t-stat (NW, 5 lags) | N/A | N/A | N/A | N/A | 4.40 | 4.11 | N/A | N/A |

IS = in-sample out-of-fold walk-forward predictions (2005-01 to 2018-12, 168 months); OOS = strict held-out window (2019-01 to 2024-12, 71 months). Basis-point values are one-way transaction costs applied two-sided to monthly turnover. FF4 regression was run on OOS returns only and only for the gross and 10 bps net series; the remaining FF4 cells are marked `N/A` to avoid reporting numbers that were not computed. Annualized α ≈ monthly α × 12.

## Known limitations

The OOS Sharpe of 2.09 is not a single-strategy live-trading Sharpe. It is the risk-adjusted return of an equal-weighted decile long–short portfolio whose two legs each contain roughly 160 names; idiosyncratic noise averages out at that breadth, and a concentrated implementation — say, thirty names per leg — would deliver a materially lower risk-adjusted return. More importantly, the 10 bps one-way TC assumption is optimistic. Even after the NYSE 20th-percentile filter, bottom-decile names trade with wider spreads and meaningful market impact once strategy AUM exceeds roughly \$100M, and the short leg carries an unmodeled borrow cost that empirically runs another 30–100 bps annualized for hard-to-borrow names. The 50 bps stress row (OOS Sharpe 1.44) is the more realistic production anchor; 10 bps is useful as an upper bound, not a planning assumption.

A related concern is factor overlap. SHAP importance is dominated by volatility and momentum proxies (realized vol, positive-months ratio, short reversal, max drawdown), and the FF4 model used for attribution does not include a low-volatility factor. Some portion of the reported alpha therefore likely reflects BAB-style and quality-minus-junk exposures rather than genuine new information, and a richer attribution model (FF5 augmented with BAB, or a full Barra-style decomposition) would be expected to shrink the residual α toward its true specific component.

Finally, the OOS window spans the COVID-19 drawdown and recovery, a regime that was unusually favorable for low-volatility and quality-tilted strategies. A regime-conditional robustness check splitting the OOS sample into crisis and non-crisis subsamples is a natural next step, and the absence of one here is a real gap rather than a rhetorical one.

## What I would do differently in production

Transaction costs should be calibrated against actual order-book depth using an Almgren–Chriss or equivalent impact model, with participation constraints sized to the target AUM, rather than a flat bps assumption. The short-leg borrow cost should be sourced from the prime broker's rate schedule rather than ignored. A capacity analysis would then answer the question the current backtest cannot: at what AUM does expected net return fall below the hurdle rate? Separately, a simple regime filter on VIX or the yield-curve slope would allow the strategy to reduce turnover during crisis periods where alpha decays and transaction costs widen.

## Reproducing

```bash
git clone <repo-url>
cd ml-equity-alpha
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set the `WRDS_USERNAME` environment variable before running the pipeline. The `wrds` package will prompt for the password on first run and cache it via `.pgpass`.

```bash
# Windows PowerShell
$env:WRDS_USERNAME = "your_username"

# macOS / Linux
export WRDS_USERNAME=your_username
```

Run the full pipeline end-to-end:

```bash
python main.py --stage all
```

Individual stages (`data`, `features`, `train`, `validate`, `portfolio`) can be run separately; each caches intermediate state to `artifacts/`.

---

WRDS data is proprietary; no data or credentials are distributed with this repository.

