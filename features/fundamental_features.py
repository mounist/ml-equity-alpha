"""
Fundamental features from Compustat annual data.

Look-ahead bias control: All Compustat items are lagged 3 months beyond
datadate before being matched to monthly observations. For example, a
fiscal year ending 2017-12-31 (datadate) becomes available for features
only from 2018-04-01 onward. This accounts for 10-K filing delays.
Winsorisation is applied cross-sectionally per date.
"""

import numpy as np
import pandas as pd

from config import WINSOR_LIMITS


def _winsorize_cs(
    df: pd.DataFrame,
    cols: list[str],
    limits: tuple[float, float] = WINSOR_LIMITS,
) -> pd.DataFrame:
    """Winsorize *cols* cross-sectionally (per date) at quantile bounds."""
    lo, hi = limits
    for col in cols:
        bounds = df.groupby("date")[col].quantile([lo, hi]).unstack()
        bounds.columns = ["lb", "ub"]
        merged = df[["date"]].merge(bounds, left_on="date", right_index=True,
                                     how="left")
        df[col] = df[col].clip(lower=merged["lb"].values,
                               upper=merged["ub"].values)
    return df


def compute_fundamental_features(
    compustat: pd.DataFrame,
    ccm_link: pd.DataFrame,
    crsp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fundamental features from Compustat annual data.

    Parameters
    ----------
    compustat : DataFrame with [gvkey, datadate, at, ceq, revt, cogs, ni,
                oancf, dltt, sale, xsga, capx, dp, act, lct].
    ccm_link :  [gvkey, permno, linkdt, linkenddt].
    crsp :      [permno, date, market_cap].

    Returns
    -------
    DataFrame with [permno, date] + 15 fundamental feature columns.
    """
    # ── 1. Link compustat to permno ─────────────────────────────────────
    comp = compustat.copy()
    link = ccm_link[["gvkey", "permno", "linkdt", "linkenddt"]].copy()

    comp["datadate"] = pd.to_datetime(comp["datadate"])
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"].fillna("2099-12-31"))

    merged = comp.merge(link, on="gvkey", how="inner")
    merged = merged[
        (merged["datadate"] >= merged["linkdt"])
        & (merged["datadate"] <= merged["linkenddt"])
    ].copy()

    # ── 2. Compute raw ratios ───────────────────────────────────────────
    at = merged["at"].replace(0, np.nan)
    ceq = merged["ceq"].replace(0, np.nan)

    merged["gross_profitability"] = (merged["revt"] - merged["cogs"]) / at
    merged["roa"] = merged["ni"] / at
    merged["roe"] = merged["ni"] / ceq
    merged["operating_leverage"] = (merged["cogs"] + merged["xsga"].fillna(0)) / at
    merged["leverage"] = merged["dltt"] / at
    merged["current_ratio"] = merged["act"] / merged["lct"].replace(0, np.nan)
    merged["accruals"] = (merged["ni"] - merged["oancf"]) / at
    merged["capex_intensity"] = merged["capx"] / at

    # ── 3. Growth (YoY) ────────────────────────────────────────────────
    merged["permno"] = merged["permno"].astype("int64")
    merged = merged.sort_values(["permno", "datadate"])
    for base, growth in [("sale", "sales_growth_1y"),
                         ("ni", "earnings_growth_1y"),
                         ("at", "asset_growth_1y")]:
        prior = merged.groupby("permno")[base].shift(1)
        merged[growth] = (merged[base] - prior) / prior.abs().replace(0, np.nan)

    # ── 4. Publication lag (3 months) ───────────────────────────────────
    merged["avail_date"] = merged["datadate"] + pd.DateOffset(months=3)

    fund_cols = [
        "ceq", "ni", "sale", "oancf",
        "gross_profitability", "roa", "roe", "operating_leverage",
        "leverage", "current_ratio", "accruals", "capex_intensity",
        "sales_growth_1y", "earnings_growth_1y", "asset_growth_1y",
    ]
    fund = merged[["permno", "avail_date"] + fund_cols].copy()
    fund = fund.rename(columns={"avail_date": "date"})
    fund = fund.sort_values(["permno", "date"]).drop_duplicates(
        subset=["permno", "date"], keep="last",
    )

    # ── 5. Map to CRSP monthly panel via merge_asof ─────────────────────
    crsp_sub = crsp[["permno", "date", "market_cap"]].copy()
    crsp_sub["date"] = pd.to_datetime(crsp_sub["date"])
    crsp_sub["permno"] = crsp_sub["permno"].astype("int64")
    crsp_sub = crsp_sub.sort_values(["permno", "date"]).drop_duplicates(
        subset=["permno", "date"], keep="last",
    )

    fund["permno"] = fund["permno"].astype("int64")
    fund["date"] = pd.to_datetime(fund["date"])

    # Single merge_asof (both sorted by date within permno groups)
    crsp_sub = crsp_sub.sort_values("date")
    fund = fund.sort_values("date")

    panel = pd.merge_asof(
        crsp_sub, fund,
        on="date", by="permno",
        direction="backward",
    )

    # ── 6. Valuation ratios ─────────────────────────────────────────────
    mkt = panel["market_cap"].replace(0, np.nan)
    panel["book_to_market"] = panel["ceq"] / mkt
    panel["earnings_yield"] = panel["ni"] / mkt
    panel["sales_to_price"] = panel["sale"] / mkt
    panel["cash_flow_yield"] = panel["oancf"] / mkt

    panel = panel.drop(columns=["ceq", "ni", "sale", "oancf", "market_cap"],
                       errors="ignore")

    # ── 7. Winsorize ────────────────────────────────────────────────────
    feature_cols = [
        "book_to_market", "earnings_yield", "sales_to_price",
        "cash_flow_yield", "gross_profitability", "roa", "roe",
        "operating_leverage", "leverage", "current_ratio", "accruals",
        "capex_intensity", "sales_growth_1y", "earnings_growth_1y",
        "asset_growth_1y",
    ]
    panel[feature_cols] = panel[feature_cols].replace([np.inf, -np.inf], np.nan)
    panel = _winsorize_cs(panel, feature_cols)

    print(f"  [fund] done: {len(panel):,} rows")
    return panel[["permno", "date"] + feature_cols]
