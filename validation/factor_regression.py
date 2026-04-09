"""
Fama-French 4-factor regression with Newey-West HAC standard errors.

Regresses portfolio excess returns on MKT, SMB, HML, UMD to decompose
performance into systematic exposures and residual alpha.

Limitation — Liquidity Factor
-----------------------------
This module implements a standard Carhart (1997) 4-factor model that does
**not** include a liquidity factor. Strategies that tilt toward small or
illiquid stocks may show alpha that is partially explained by a liquidity
premium. Candidates for a 5th factor include:

- **Pastor-Stambaugh (2003)** traded liquidity factor (available from
  WRDS at ``pastor_stambaugh.ps_liquidity``).
- **Amihud (2002)** illiquidity measure (computable from CRSP daily
  data via ``abs(ret) / dollar_volume``).

If a significant portion of the strategy's returns comes from liquidity
risk, the reported FF4 alpha will overstate true risk-adjusted
performance. See :func:`ff5_liquidity_regression` for an optional
extension that adds the Pastor-Stambaugh liquidity factor when the
data is available.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config

log = logging.getLogger(__name__)


def ff4_regression(
    portfolio_returns: pd.Series,
    ff_factors: pd.DataFrame,
    n_lags: int | None = None,
) -> dict:
    """Run a Fama-French 4-factor regression with Newey-West HAC SEs.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio returns indexed by date.
    ff_factors : pd.DataFrame
        Must contain columns ``mktrf``, ``smb``, ``hml``, ``umd``,
        ``rf`` indexed (or with a column) by date.
    n_lags : int, optional
        Maximum lags for Newey-West HAC. Defaults to ``config.NW_LAGS``.

    Returns
    -------
    dict
        Regression results including ``alpha``, ``alpha_tstat``,
        ``beta_mkt``, ``beta_mkt_tstat``, ``beta_smb``,
        ``beta_smb_tstat``, ``beta_hml``, ``beta_hml_tstat``,
        ``beta_umd``, ``beta_umd_tstat``, ``r_squared``, ``n_obs``.
        All numeric values are plain Python floats.
    """
    n_lags = n_lags if n_lags is not None else config.NW_LAGS

    # Align on date
    ret = portfolio_returns.copy()
    ret.name = "port_ret"
    if isinstance(ret.index, pd.DatetimeIndex):
        ret = ret.reset_index()
        ret.columns = ["date", "port_ret"]

    ff = ff_factors.copy()
    if "date" in ff.columns:
        ff["date"] = pd.to_datetime(ff["date"])
    else:
        ff = ff.reset_index()
        ff.columns = ["date"] + list(ff.columns[1:])

    # Align by year-month (portfolio uses month-end, FF uses month-start)
    ret["ym"] = ret["date"].dt.to_period("M")
    ff["ym"] = ff["date"].dt.to_period("M")
    merged = ret.merge(ff.drop(columns=["date"]), on="ym", how="inner")
    merged = merged.drop(columns=["ym"])
    factor_cols = ["mktrf", "smb", "hml", "umd", "rf"]
    for c in factor_cols + ["port_ret"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.dropna(subset=["port_ret"] + factor_cols)

    # Excess return
    y = (merged["port_ret"] - merged["rf"]).astype(float)
    X = merged[["mktrf", "smb", "hml", "umd"]].astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": n_lags})

    params = model.params
    tstats = model.tvalues

    return {
        "alpha": float(params["const"]),
        "alpha_tstat": float(tstats["const"]),
        "beta_mkt": float(params["mktrf"]),
        "beta_mkt_tstat": float(tstats["mktrf"]),
        "beta_smb": float(params["smb"]),
        "beta_smb_tstat": float(tstats["smb"]),
        "beta_hml": float(params["hml"]),
        "beta_hml_tstat": float(tstats["hml"]),
        "beta_umd": float(params["umd"]),
        "beta_umd_tstat": float(tstats["umd"]),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
    }


def ff5_liquidity_regression(
    portfolio_returns: pd.Series,
    ff_factors: pd.DataFrame,
    liquidity_factor: pd.DataFrame,
    n_lags: int | None = None,
) -> dict:
    """Run a 5-factor regression adding a liquidity factor to FF4.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio returns indexed by date.
    ff_factors : pd.DataFrame
        Standard FF4 factors (mktrf, smb, hml, umd, rf).
    liquidity_factor : pd.DataFrame
        Must contain columns ``date`` and ``liq`` (traded liquidity
        factor, e.g. Pastor-Stambaugh).
    n_lags : int, optional
        Newey-West HAC lags. Defaults to ``config.NW_LAGS``.

    Returns
    -------
    dict
        Same as :func:`ff4_regression` plus ``beta_liq`` and
        ``beta_liq_tstat``.
    """
    n_lags = n_lags if n_lags is not None else config.NW_LAGS

    ret = portfolio_returns.copy()
    ret.name = "port_ret"
    if isinstance(ret.index, pd.DatetimeIndex):
        ret = ret.reset_index()
        ret.columns = ["date", "port_ret"]

    ff = ff_factors.copy()
    if "date" not in ff.columns:
        ff = ff.reset_index()
    ff["date"] = pd.to_datetime(ff["date"])
    ff["ym"] = ff["date"].dt.to_period("M")

    liq = liquidity_factor.copy()
    liq["date"] = pd.to_datetime(liq["date"])
    liq["ym"] = liq["date"].dt.to_period("M")

    ret["ym"] = ret["date"].dt.to_period("M")
    merged = ret.merge(ff.drop(columns=["date"]), on="ym", how="inner")
    merged = merged.merge(liq[["ym", "liq"]], on="ym", how="inner")
    merged = merged.drop(columns=["ym"])

    for c in ["port_ret", "mktrf", "smb", "hml", "umd", "rf", "liq"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.dropna()

    y = (merged["port_ret"] - merged["rf"]).astype(float)
    X = merged[["mktrf", "smb", "hml", "umd", "liq"]].astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": n_lags})
    params = model.params
    tstats = model.tvalues

    return {
        "alpha": float(params["const"]),
        "alpha_tstat": float(tstats["const"]),
        "beta_mkt": float(params["mktrf"]),
        "beta_mkt_tstat": float(tstats["mktrf"]),
        "beta_smb": float(params["smb"]),
        "beta_smb_tstat": float(tstats["smb"]),
        "beta_hml": float(params["hml"]),
        "beta_hml_tstat": float(tstats["hml"]),
        "beta_umd": float(params["umd"]),
        "beta_umd_tstat": float(tstats["umd"]),
        "beta_liq": float(params["liq"]),
        "beta_liq_tstat": float(tstats["liq"]),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
    }


def run_factor_attribution(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    ff_factors: pd.DataFrame,
    liquidity_factor: pd.DataFrame | None = None,
) -> dict:
    """Run FF4 regression on both gross and net-of-cost portfolio returns.

    Optionally runs a 5-factor regression including a liquidity factor
    (e.g. Pastor-Stambaugh) if ``liquidity_factor`` is provided.

    Parameters
    ----------
    gross_returns : pd.Series
        Gross (before transaction cost) portfolio returns indexed by
        date.
    net_returns : pd.Series
        Net (after transaction cost) portfolio returns indexed by date.
    ff_factors : pd.DataFrame
        Fama-French factor data (see :func:`ff4_regression`).
    liquidity_factor : pd.DataFrame, optional
        If provided, must contain ``date`` and ``liq`` columns. Enables
        the 5-factor regression.

    Returns
    -------
    dict
        ``{"gross": {...}, "net": {...}}`` where each sub-dict is the
        output of :func:`ff4_regression`. Saved to
        ``RESULTS_DIR/ff4_oos.json``.
    """
    log.warning(
        "FF4 alpha does not control for liquidity risk. The reported "
        "alpha may be partially explained by a liquidity premium "
        "(Pastor-Stambaugh or Amihud). Interpret with caution."
    )

    results = {
        "gross": ff4_regression(gross_returns, ff_factors),
        "net": ff4_regression(net_returns, ff_factors),
    }

    with open(config.RESULTS_DIR / "ff4_oos.json", "w") as f:
        json.dump(results, f, indent=2)

    # Optional 5-factor regression with liquidity
    if liquidity_factor is not None and not liquidity_factor.empty:
        log.info("Running FF5 (with liquidity factor) regression ...")
        results["gross_ff5_liq"] = ff5_liquidity_regression(
            gross_returns, ff_factors, liquidity_factor,
        )
        results["net_ff5_liq"] = ff5_liquidity_regression(
            net_returns, ff_factors, liquidity_factor,
        )
        log.info("  FF5-liq alpha (gross): %.4f (t=%.2f)",
                 results["gross_ff5_liq"]["alpha"],
                 results["gross_ff5_liq"]["alpha_tstat"])

        with open(config.RESULTS_DIR / "ff5_liq_oos.json", "w") as f:
            json.dump({
                "gross": results["gross_ff5_liq"],
                "net": results["net_ff5_liq"],
            }, f, indent=2)
    else:
        log.info(
            "No liquidity factor data provided. To run a 5-factor "
            "regression, supply Pastor-Stambaugh liquidity data from "
            "WRDS (pastor_stambaugh.ps_liquidity) or compute Amihud "
            "illiquidity from CRSP daily data."
        )

    return results
