"""
Feature pipeline: merge all feature sets, apply rank normalisation,
compute forward returns, and produce the final modelling panel.

Look-ahead bias control: Forward returns are computed by shifting
realised returns forward. Rank normalisation is purely cross-sectional
(per date) so no future information leaks into features.
"""

import logging

import numpy as np
import pandas as pd

from config import WINSOR_LIMITS, RESULTS_DIR
from features.price_features import compute_price_features
from features.fundamental_features import compute_fundamental_features
from features.quality_features import compute_sue

logger = logging.getLogger(__name__)


# ── Universe filter ────────────────────────────────────────────────────────

def apply_universe_filters(crsp: pd.DataFrame) -> pd.DataFrame:
    """Apply monthly tradability filters to the CRSP universe.

    Filters applied each month **before** feature construction:

    1. Drop stocks with missing ``ret`` or ``prc``.
    2. Minimum share price: ``abs(prc) >= 5``.
    3. Minimum market cap: >= NYSE 20th-percentile market cap for that
       month (NYSE identified by ``exchcd == 1``; threshold applied to
       all stocks).

    Parameters
    ----------
    crsp : pd.DataFrame
        Raw CRSP monthly file with columns ``permno``, ``date``, ``ret``,
        ``prc``, ``market_cap``, ``exchcd``.

    Returns
    -------
    pd.DataFrame
        Filtered CRSP data containing only tradeable stocks.
    """
    n_start = len(crsp)

    # 1. Drop missing ret or prc
    crsp = crsp.dropna(subset=["ret", "prc"]).copy()
    n_after_missing = len(crsp)

    # 2. Minimum share price
    crsp = crsp[crsp["prc"].abs() >= 5.0]
    n_after_price = len(crsp)

    # 3. Minimum market cap: NYSE 20th percentile per month
    nyse = crsp[crsp["exchcd"] == 1]
    nyse_p20 = nyse.groupby("date")["market_cap"].quantile(0.20)
    nyse_p20.name = "_nyse_mcap_p20"
    crsp = crsp.merge(nyse_p20, on="date", how="left")
    crsp = crsp[crsp["market_cap"] >= crsp["_nyse_mcap_p20"]]
    crsp = crsp.drop(columns=["_nyse_mcap_p20"])
    n_after_mcap = len(crsp)

    logger.info(
        "Universe filters: %d -> %d rows "
        "(missing ret/prc: -%d, price<$5: -%d, mcap<NYSE-p20: -%d)",
        n_start, n_after_mcap,
        n_start - n_after_missing,
        n_after_missing - n_after_price,
        n_after_price - n_after_mcap,
    )

    # Per-month summary
    monthly_counts = crsp.groupby("date")["permno"].nunique()
    logger.info(
        "Median stocks/month after filters: %d  (range: %d – %d)",
        int(monthly_counts.median()),
        int(monthly_counts.min()),
        int(monthly_counts.max()),
    )

    return crsp.reset_index(drop=True)


# ── Feature column lists ────────────────────────────────────────────────────

# Excluded from model features (remain as raw columns in the panel):
#   - log_market_cap, log_dollar_volume: pure size, model would just
#     trade a size bet.
#   - amihud_illiquidity: illiquidity risk compensation, not tradeable
#     alpha — disappears with realistic transaction costs.
PRICE_FEATURE_COLS = [
    "mom_12_1", "mom_6_1", "mom_3_1", "short_reversal",
    "realized_vol_12", "realized_vol_3", "vol_ratio",
    "downside_vol_12", "max_drawdown_12",
    "skewness_12", "kurtosis_12", "positive_months_ratio",
]

FUNDAMENTAL_FEATURE_COLS = [
    "book_to_market", "earnings_yield", "sales_to_price", "cash_flow_yield",
    "gross_profitability", "roa", "roe", "operating_leverage",
    "leverage", "current_ratio", "accruals", "capex_intensity",
    "sales_growth_1y", "earnings_growth_1y", "asset_growth_1y",
]

QUALITY_FEATURE_COLS = ["sue"]

FEATURE_COLS = PRICE_FEATURE_COLS + FUNDAMENTAL_FEATURE_COLS + QUALITY_FEATURE_COLS


def _rank_normalize(series: pd.Series) -> pd.Series:
    """
    Map values to the uniform [0, 1] interval via ranks.

    Result: (rank - 0.5) / count. Ties receive average rank.
    """
    ranks = series.rank(method="average", na_option="keep")
    count = series.notna().sum()
    if count == 0:
        return series * np.nan
    return (ranks - 0.5) / count


def get_feature_cols() -> list[str]:
    """Return the list of rank-normalised feature column names used for
    modelling (suffix ``_rank``)."""
    return [f"{c}_rank" for c in FEATURE_COLS]


# ── main builder ────────────────────────────────────────────────────────────


def build_feature_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the full modelling panel by applying universe filters, merging
    price, fundamental, and quality features, computing forward returns,
    and applying rank normalisation.

    Universe filters (applied monthly before feature construction):
    - Drop stocks with missing ``ret`` or ``prc``
    - Minimum share price: ``abs(prc) >= 5``
    - Minimum market cap: >= NYSE 20th-percentile for that month

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Must contain keys: ``crsp_msf``, ``compustat``, ``ccm_link``,
        ``ibes_actuals``, ``ibes_statsum``, ``ibes_crsp_link``.

    Returns
    -------
    pd.DataFrame
        Panel indexed by (permno, date) with raw features, rank-normalised
        features (``*_rank``), and forward return columns.
    """

    # ── 0. Apply universe filters ──────────────────────────────────────
    logger.info("Applying universe filters ...")
    crsp_filtered = apply_universe_filters(data["crsp_msf"])

    # ── 1. Compute feature blocks ───────────────────────────────────────
    logger.info("Computing price features ...")
    price_feat = compute_price_features(crsp_filtered)

    logger.info("Computing fundamental features ...")
    fund_feat = compute_fundamental_features(
        data["compustat"], data["ccm_link"], crsp_filtered,
    )

    logger.info("Computing SUE ...")
    sue_feat = compute_sue(
        data["ibes_actuals"], data["ibes_statsum"], data["ibes_crsp_link"],
    )

    # ── 2. Merge on (permno, date) ──────────────────────────────────────
    panel = price_feat.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["permno"] = panel["permno"].astype(int)

    for other in [fund_feat, sue_feat]:
        if other.empty:
            continue
        other = other.copy()
        other["date"] = pd.to_datetime(other["date"])
        other["permno"] = other["permno"].astype(int)
        panel = panel.merge(other, on=["permno", "date"], how="left")

    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    # ── 3. Forward returns ──────────────────────────────────────────────
    crsp_ret = crsp_filtered[["permno", "date", "ret"]].copy()
    crsp_ret["date"] = pd.to_datetime(crsp_ret["date"])
    crsp_ret["permno"] = crsp_ret["permno"].astype(int)
    crsp_ret = crsp_ret.drop_duplicates(subset=["permno", "date"])

    panel = panel.merge(crsp_ret, on=["permno", "date"], how="left")
    panel = panel.sort_values(["permno", "date"])

    # fwd_ret_1m: next month's return
    panel["fwd_ret_1m"] = panel.groupby("permno")["ret"].shift(-1)

    # fwd_ret_3m / fwd_ret_6m: cumulative returns over the next 3 / 6
    # months (t+1 through t+3 and t+1 through t+6 respectively).
    # Reverse the series within each group so that a backward rolling
    # window becomes a forward window.
    def _fwd_cumret(g, months):
        lr = np.log1p(g["ret"]).iloc[::-1]
        cum = lr.rolling(months, min_periods=months).sum().iloc[::-1]
        # shift so the window starts at t+1, not t
        return np.expm1(cum.shift(-1))

    panel["fwd_ret_3m"] = panel.groupby("permno", group_keys=False).apply(
        _fwd_cumret, months=3,
    )
    panel["fwd_ret_6m"] = panel.groupby("permno", group_keys=False).apply(
        _fwd_cumret, months=6,
    )

    # Drop the raw ret column (not a feature)
    panel = panel.drop(columns=["ret"], errors="ignore")

    # ── 4. Winsorize forward returns cross-sectionally ──────────────────
    fwd_cols = ["fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m"]
    lo, hi = WINSOR_LIMITS
    for col in fwd_cols:
        bounds = panel.groupby("date")[col].quantile([lo, hi]).unstack()
        bounds.columns = ["lb", "ub"]
        tmp = panel[["date"]].merge(bounds, left_on="date",
                                     right_index=True, how="left")
        panel[col] = panel[col].clip(lower=tmp["lb"].values,
                                      upper=tmp["ub"].values)

    # ── 5. Rank-normalise features ──────────────────────────────────────
    for col in FEATURE_COLS:
        rank_col = f"{col}_rank"
        if col in panel.columns:
            panel[rank_col] = panel.groupby("date")[col].transform(
                _rank_normalize,
            )
            panel[rank_col] = panel[rank_col].fillna(0.5)
        else:
            panel[rank_col] = 0.5

    # ── 6. Log feature coverage ─────────────────────────────────────────
    panel["year"] = panel["date"].dt.year
    coverage_records = []
    for col in FEATURE_COLS:
        if col not in panel.columns:
            continue
        cov = panel.groupby("year")[col].apply(
            lambda s: s.notna().mean(),
        ).rename("coverage")
        cov = cov.reset_index()
        cov["feature"] = col
        coverage_records.append(cov)
    if coverage_records:
        coverage = pd.concat(coverage_records, ignore_index=True)
        coverage = coverage.pivot(index="feature", columns="year",
                                   values="coverage")
        coverage.to_csv(RESULTS_DIR / "feature_coverage.csv")
        logger.info("Feature coverage saved to %s",
                     RESULTS_DIR / "feature_coverage.csv")
    panel = panel.drop(columns=["year"], errors="ignore")

    logger.info(
        "Feature panel built: %d rows, %d columns", len(panel), len(panel.columns),
    )
    return panel
