"""
Portfolio construction and backtesting.

Constructs decile portfolios from model predictions, computes returns
net of transaction costs, and reports performance metrics for both
in-sample and out-of-sample periods.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Decile portfolio returns
# ---------------------------------------------------------------------------

def decile_portfolio_returns(
    panel: pd.DataFrame,
    score_col: str,
    ret_col: str = "fwd_ret_1m",
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Equal-weighted quantile portfolio returns per date.

    For each date, rank stocks by *score_col* into *n_quantiles* buckets.
    Compute the equal-weighted mean return for each bucket and a long-short
    spread (top quantile minus bottom quantile).

    Returns
    -------
    DataFrame with columns: date, q1 .. q{n_quantiles}, long_short, n_stocks.
    """

    def _quantile_rets(grp: pd.DataFrame) -> pd.Series:
        valid = grp.dropna(subset=[score_col, ret_col])
        if len(valid) < n_quantiles:
            return pd.Series(dtype=float)
        valid = valid.copy()
        valid["qbin"] = pd.qcut(
            valid[score_col], q=n_quantiles, labels=False, duplicates="drop"
        )
        actual_bins = int(valid["qbin"].max()) + 1
        means = valid.groupby("qbin")[ret_col].mean()
        out = {f"q{int(q) + 1}": means.get(q, np.nan) for q in range(n_quantiles)}
        # Use actual top/bottom bins for long_short when fewer bins exist
        out["_actual_top"] = float(means.iloc[-1])
        out["_actual_bot"] = float(means.iloc[0])
        out["n_stocks"] = len(valid) // actual_bins
        return pd.Series(out)

    result = panel.groupby("date").apply(_quantile_rets).reset_index()
    top = f"q{n_quantiles}"
    # Use actual top/bottom when full decile columns have NaN
    if top in result.columns:
        result["long_short"] = result[top].fillna(result["_actual_top"]) - result["q1"].fillna(result["_actual_bot"])
    else:
        result["long_short"] = result["_actual_top"] - result["_actual_bot"]
    result.drop(columns=["_actual_top", "_actual_bot"], inplace=True, errors="ignore")
    return result


# ---------------------------------------------------------------------------
# 2. Sector-neutral portfolio
# ---------------------------------------------------------------------------

def sector_neutral_portfolio(
    panel: pd.DataFrame,
    score_col: str,
    ret_col: str = "fwd_ret_1m",
    sector_col: str = "siccd",
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Rank within 2-digit SIC sector, then form overall decile portfolios."""

    df = panel.copy()
    df["sector2"] = df[sector_col] // 100

    def _sector_rank(grp: pd.DataFrame) -> pd.Series:
        valid = grp.dropna(subset=[score_col])
        if len(valid) < 2:
            return pd.Series(np.nan, index=valid.index)
        return valid[score_col].rank(pct=True)

    df["sector_pctrank"] = df.groupby(["date", "sector2"]).apply(
        lambda g: _sector_rank(g)
    ).reset_index(level=[0, 1], drop=True)

    return decile_portfolio_returns(
        df, score_col="sector_pctrank", ret_col=ret_col, n_quantiles=n_quantiles
    )


# ---------------------------------------------------------------------------
# 3. Turnover
# ---------------------------------------------------------------------------

def compute_turnover(
    panel: pd.DataFrame,
    score_col: str,
    n_quantiles: int = 10,
) -> pd.Series:
    """Monthly turnover rate for the top and bottom decile.

    Turnover is defined as the fraction of stocks that changed membership in
    the extreme deciles relative to the prior month.
    """

    dates = sorted(panel["date"].unique())
    turnovers = {}

    prev_top: set = set()
    prev_bot: set = set()

    for dt in dates:
        sub = panel.loc[panel["date"] == dt].dropna(subset=[score_col]).copy()
        if len(sub) < n_quantiles:
            continue
        sub["qbin"] = pd.qcut(
            sub[score_col], q=n_quantiles, labels=False, duplicates="drop"
        )
        cur_top = set(sub.loc[sub["qbin"] == n_quantiles - 1, "permno"])
        cur_bot = set(sub.loc[sub["qbin"] == 0, "permno"])

        if prev_top or prev_bot:
            top_union = cur_top | prev_top
            bot_union = cur_bot | prev_bot
            top_turn = (
                len(cur_top.symmetric_difference(prev_top)) / max(len(top_union), 1)
            )
            bot_turn = (
                len(cur_bot.symmetric_difference(prev_bot)) / max(len(bot_union), 1)
            )
            turnovers[dt] = (top_turn + bot_turn) / 2.0

        prev_top, prev_bot = cur_top, cur_bot

    return pd.Series(turnovers, name="turnover").sort_index()


# ---------------------------------------------------------------------------
# 4. Transaction costs
# ---------------------------------------------------------------------------

def apply_transaction_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    tc_bps: Optional[float] = None,
) -> pd.Series:
    """Net returns after accounting for two-sided transaction costs.

    net = gross - turnover * 2 * tc_bps / 10_000
    """
    if tc_bps is None:
        tc_bps = config.TC_ONE_WAY_BPS
    aligned_turnover = turnover.reindex(gross_returns.index, fill_value=0.0)
    return gross_returns - aligned_turnover * 2.0 * tc_bps / 10_000


# ---------------------------------------------------------------------------
# 5. Performance statistics
# ---------------------------------------------------------------------------

def performance_stats(
    returns: pd.Series,
    period: str = "monthly",
) -> Dict[str, float]:
    """Compute standard performance metrics from a return series.

    Assumes monthly returns.  Annualisation: ret * 12, vol * sqrt(12).
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {k: np.nan for k in [
            "ann_ret", "ann_vol", "sharpe", "sortino",
            "max_drawdown", "calmar", "avg_monthly_ret", "n_months",
        ]}

    mean_m = r.mean()
    std_m = r.std(ddof=1)
    ann_ret = mean_m * 12
    ann_vol = std_m * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0].std(ddof=1)
    sortino = ann_ret / (downside * np.sqrt(12)) if (downside > 0) else np.nan

    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "avg_monthly_ret": float(mean_m),
        "n_months": int(n),
    }


# ---------------------------------------------------------------------------
# 6. Combine OOF (IS) + final model (OOS) predictions
# ---------------------------------------------------------------------------

def build_full_predictions_panel(
    panel: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    oos_preds: pd.Series,
    oos_mask: pd.Series,
) -> pd.DataFrame:
    """Build a predictions panel using OOF for IS and final model for OOS.

    IS predictions come from walk-forward CV out-of-fold predictions
    (each row predicted by a model that never saw that row's data).
    OOS predictions come from the final model trained on all IS data.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel (must contain permno, date columns).
    oof_predictions : pd.DataFrame
        OOF predictions with columns: permno, date, oof_pred, actual.
    oos_preds : pd.Series
        Final model predictions for OOS rows, aligned with panel index.
    oos_mask : pd.Series
        Boolean mask for OOS rows in panel.

    Returns
    -------
    pd.DataFrame
        Panel with a ``pred`` column containing OOF preds for IS and
        final model preds for OOS.
    """
    panel = panel.copy()
    panel["pred"] = np.nan

    # OOS: use final model predictions
    panel.loc[oos_mask, "pred"] = oos_preds.values

    # IS: merge OOF predictions
    if not oof_predictions.empty:
        oof = oof_predictions[["permno", "date", "oof_pred"]].copy()

        # CRITICAL: enforce identical dtypes before merge
        panel["permno"] = panel["permno"].astype(np.int64)
        panel["date"] = pd.to_datetime(panel["date"])
        oof["permno"] = oof["permno"].astype(np.int64)
        oof["date"] = pd.to_datetime(oof["date"])

        # Use year-month key for merge to handle date misalignment
        # (panel may have last-trading-day dates, OOF may have month-end)
        panel["_ym"] = panel["date"].dt.to_period("M")
        oof["_ym"] = oof["date"].dt.to_period("M")

        # Remove duplicates from OOF before merge
        oof = oof.drop_duplicates(subset=["permno", "_ym"], keep="last")

        # Diagnostic: print coverage before merge
        is_mask_local = panel["date"] <= pd.Timestamp(config.TRAIN_END)
        is_panel_months = panel.loc[is_mask_local, "_ym"].nunique()
        oof_months = oof["_ym"].nunique()
        print(f"[build_predictions] IS panel months: {is_panel_months}")
        print(f"[build_predictions] OOF months: {oof_months}")

        # Merge on (permno, year-month) to avoid date alignment issues
        oof_merge = oof[["permno", "_ym", "oof_pred"]]
        n_before = len(panel)
        panel = panel.merge(oof_merge, on=["permno", "_ym"], how="left")
        n_after = len(panel)
        if n_after != n_before:
            print(f"[build_predictions] WARNING: merge changed row count "
                  f"{n_before} -> {n_after}, dropping duplicates")
            panel = panel.drop_duplicates(
                subset=["permno", "date"], keep="first",
            )
            panel = panel.reset_index(drop=True)

        # Fill IS pred from oof_pred
        is_fill = panel["pred"].isna() & panel["oof_pred"].notna()
        panel.loc[is_fill, "pred"] = panel.loc[is_fill, "oof_pred"]
        panel.drop(columns=["oof_pred", "_ym"], inplace=True)

        # Diagnostic: check coverage after merge (recompute mask)
        is_mask_local = panel["date"] <= pd.Timestamp(config.TRAIN_END)
        is_with_pred = panel.loc[is_mask_local, "pred"].notna().mean()
        print(f"[build_predictions] IS pred coverage after merge: {is_with_pred:.1%}")

        # Verify no months are completely missing predictions
        is_panel = panel.loc[is_mask_local]
        months_with_preds = is_panel.groupby(is_panel["date"].dt.to_period("M"))["pred"].apply(
            lambda x: x.notna().any()
        )
        missing_months = months_with_preds[~months_with_preds].index.tolist()
        if missing_months:
            print(f"[build_predictions] WARNING: {len(missing_months)} IS months "
                  f"still have no predictions: {missing_months[:5]}...")
        else:
            print(f"[build_predictions] All IS months have predictions")

    return panel


# ---------------------------------------------------------------------------
# 7. Full backtest driver
# ---------------------------------------------------------------------------

def run_backtest(
    panel: pd.DataFrame,
    score_col: str,
    ff_factors: Optional[pd.DataFrame] = None,
) -> dict:
    """End-to-end backtest: decile sorts, turnover, TC, perf stats.

    Parameters
    ----------
    panel : DataFrame
        Must contain columns: date, permno, siccd, *score_col*, fwd_ret_1m.
        Optionally includes ``vwretd`` for market benchmark.
    score_col : str
        Name of the signal / prediction column.
    ff_factors : DataFrame, optional
        Fama-French factors (unused here but reserved for factor regression).

    Returns
    -------
    dict  with keys: decile_rets, sector_neutral_rets, perf (nested IS/OOS).
    """

    train_end = pd.Timestamp(config.TRAIN_END)
    test_start = pd.Timestamp(config.TEST_START)

    # --- Decile sorts -------------------------------------------------------
    decile_df = decile_portfolio_returns(panel, score_col)
    sn_df = sector_neutral_portfolio(panel, score_col)
    turnover = compute_turnover(panel, score_col)

    ls_gross = decile_df.set_index("date")["long_short"]
    ls_net = apply_transaction_costs(ls_gross, turnover)

    # --- IS / OOS split -----------------------------------------------------
    is_mask = ls_gross.index <= train_end
    oos_mask = ls_gross.index >= test_start

    perf: Dict[str, dict] = {
        "is_gross": performance_stats(ls_gross[is_mask]),
        "is_net": performance_stats(ls_net[is_mask]),
        "oos_gross": performance_stats(ls_gross[oos_mask]),
        "oos_net": performance_stats(ls_net[oos_mask]),
    }

    # --- Market benchmark ---------------------------------------------------
    if "vwretd" in panel.columns:
        mkt = panel.groupby("date")["vwretd"].first().sort_index()
        perf["market_is"] = performance_stats(mkt[mkt.index <= train_end])
        perf["market_oos"] = performance_stats(mkt[mkt.index >= test_start])

    # --- TC sensitivity analysis ---------------------------------------------
    tc_levels = [10.0, 25.0, 50.0]
    tc_sensitivity: Dict[str, dict] = {}
    for tc_bps in tc_levels:
        ls_net_tc = apply_transaction_costs(ls_gross, turnover, tc_bps=tc_bps)
        key = f"{int(tc_bps)}bps"
        tc_sensitivity[key] = {
            "is": performance_stats(ls_net_tc[is_mask]),
            "oos": performance_stats(ls_net_tc[oos_mask]),
        }

    tc_sens_path = config.RESULTS_DIR / "tc_sensitivity.json"
    with open(tc_sens_path, "w") as f:
        json.dump(tc_sensitivity, f, indent=2)

    # Log summary table
    log.info("TC Sensitivity Analysis:")
    log.info("  %-8s  %-12s  %-12s  %-12s  %-12s", "TC (bps)",
             "IS Sharpe", "IS Ann Ret", "OOS Sharpe", "OOS Ann Ret")
    for tc_bps in tc_levels:
        key = f"{int(tc_bps)}bps"
        is_s = tc_sensitivity[key]["is"]
        oos_s = tc_sensitivity[key]["oos"]
        log.info("  %-8s  %-12.2f  %-+12.2f%%  %-12.2f  %-+12.2f%%",
                 key,
                 is_s.get("sharpe", 0),
                 is_s.get("ann_ret", 0) * 100,
                 oos_s.get("sharpe", 0),
                 oos_s.get("ann_ret", 0) * 100)

    # --- Persist ------------------------------------------------------------
    out_json = config.RESULTS_DIR / "portfolio_stats.json"
    with open(out_json, "w") as f:
        json.dump(perf, f, indent=2)

    monthly_out = config.RESULTS_DIR / "portfolio_monthly.parquet"
    monthly_df = pd.DataFrame({
        "date": ls_gross.index,
        "gross_ls": ls_gross.values,
        "net_ls": ls_net.values,
    })
    monthly_df.to_parquet(monthly_out, index=False)

    return {
        "decile_rets": decile_df,
        "sector_neutral_rets": sn_df,
        "turnover": turnover,
        "perf": perf,
        "monthly_df": monthly_df,
        "tc_sensitivity": tc_sensitivity,
    }
