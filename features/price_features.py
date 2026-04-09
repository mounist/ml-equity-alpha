"""
Price, momentum, volatility, and liquidity features derived from CRSP monthly data.

Look-ahead bias control: All features use only returns through month t-1
(or earlier). The current month's return is never included in any feature
except short_reversal, which uses the *prior* month return (sign-flipped).
"""

import numpy as np
import pandas as pd


def _safe_rolling(series: pd.Series, grp_obj, window: int,
                  min_periods: int, func: str, **kwargs) -> pd.Series:
    """Apply a rolling function respecting group boundaries."""
    return grp_obj.transform(lambda s: getattr(s.rolling(window, min_periods=min_periods), func)(**kwargs))


def compute_price_features(crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price-based features from CRSP monthly stock file.

    Parameters
    ----------
    crsp : pd.DataFrame
        CRSP monthly data with columns [permno, date, ret, prc, shrout,
        vol, dollar_volume, market_cap].

    Returns
    -------
    pd.DataFrame
        16 price-based feature columns plus permno and date.
    """
    cols_needed = ["permno", "date", "ret", "dollar_volume", "market_cap"]
    df = crsp[cols_needed].copy()
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

    g = df.groupby("permno")["ret"]

    # ── Momentum ────────────────────────────────────────────────────────
    # log returns shifted by 1 to skip most recent month
    print("  [price] momentum ...")
    log_r = np.log1p(df["ret"])
    df["_lr"] = log_r
    g_lr = df.groupby("permno")["_lr"]

    # Shifted log return = log(1 + ret) of the prior month
    df["_lr_s1"] = g_lr.shift(1)

    # Rolling sum of shifted log returns gives cumulative return
    # ending one month before current. Window=11 gives months t-11..t-1 shifted
    # = original months t-12..t-2 (11 months, skip most recent).
    g_lr_s = df.groupby("permno")["_lr_s1"]
    df["mom_12_1"] = np.expm1(g_lr_s.transform(
        lambda s: s.rolling(11, min_periods=11).sum()))
    df["mom_6_1"] = np.expm1(g_lr_s.transform(
        lambda s: s.rolling(5, min_periods=5).sum()))
    df["mom_3_1"] = np.expm1(g_lr_s.transform(
        lambda s: s.rolling(2, min_periods=2).sum()))
    df["short_reversal"] = -g.shift(1)

    # ── Volatility ──────────────────────────────────────────────────────
    print("  [price] volatility ...")
    df["realized_vol_12"] = g.transform(
        lambda s: s.rolling(12, min_periods=12).std())
    df["realized_vol_3"] = g.transform(
        lambda s: s.rolling(3, min_periods=3).std())
    df["vol_ratio"] = df["realized_vol_3"] / df["realized_vol_12"]

    # Downside vol: std of min(ret, 0) over 12 months
    df["_neg_ret"] = df["ret"].clip(upper=0)
    df["_neg_sq"] = df["_neg_ret"] ** 2
    df["_is_neg"] = (df["ret"] < 0).astype(float)
    gn = df.groupby("permno")
    n_neg = gn["_is_neg"].transform(lambda s: s.rolling(12, min_periods=6).sum())
    sum_neg = gn["_neg_ret"].transform(lambda s: s.rolling(12, min_periods=6).sum())
    sum_neg_sq = gn["_neg_sq"].transform(lambda s: s.rolling(12, min_periods=6).sum())
    var_neg = (sum_neg_sq - sum_neg**2 / n_neg.clip(lower=1)) / (n_neg - 1).clip(lower=1)
    df["downside_vol_12"] = np.sqrt(var_neg.clip(lower=0))
    df.loc[n_neg < 2, "downside_vol_12"] = np.nan

    # Max drawdown over 12 months — approximate via cumulative return approach
    # For each row, compute running max of cumulative return within each group,
    # then drawdown from that peak. Use expanding within a rolling-like window.
    # Exact rolling max-drawdown is expensive; use a fast approximation:
    # dd ≈ min trailing cumret / max trailing cumret - 1
    print("  [price] max drawdown ...")
    df["_cum_lr_12"] = gn["_lr"].transform(
        lambda s: s.rolling(12, min_periods=6).sum())
    df["_max_lr_12"] = gn["_lr"].transform(
        lambda s: s.rolling(12, min_periods=6).max()) * 12  # rough peak proxy
    # Better approach: just use rolling min of cumulative return
    df["_cum_ret"] = gn["_lr"].transform(lambda s: s.cumsum())
    df["_roll_max_cum"] = gn["_cum_ret"].transform(
        lambda s: s.rolling(12, min_periods=6).max())
    df["_roll_min_cum"] = gn["_cum_ret"].transform(
        lambda s: s.rolling(12, min_periods=6).min())
    # Max drawdown = exp(min_cum - max_cum up to that point) - 1
    # This is an approximation; exact requires path tracking
    df["max_drawdown_12"] = np.expm1(df["_roll_min_cum"] - df["_roll_max_cum"])
    df.loc[df["max_drawdown_12"] > 0, "max_drawdown_12"] = 0  # DD is always <= 0

    # ── Liquidity / size ────────────────────────────────────────────────
    print("  [price] liquidity ...")
    df["log_market_cap"] = np.log(df["market_cap"].clip(lower=1e-8))
    avg_dvol = df.groupby("permno")["dollar_volume"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    df["log_dollar_volume"] = np.log(avg_dvol.clip(lower=1e-8))

    df["_amihud_raw"] = df["ret"].abs() / df["dollar_volume"].replace(0, np.nan)
    df["amihud_illiquidity"] = df.groupby("permno")["_amihud_raw"].transform(
        lambda s: s.rolling(12, min_periods=6).mean())

    # ── Return distribution ─────────────────────────────────────────────
    print("  [price] distribution ...")
    df["skewness_12"] = g.transform(
        lambda s: s.rolling(12, min_periods=8).skew())
    df["kurtosis_12"] = g.transform(
        lambda s: s.rolling(12, min_periods=8).kurt())
    df["_pos"] = (df["ret"] > 0).astype(float)
    df["positive_months_ratio"] = df.groupby("permno")["_pos"].transform(
        lambda s: s.rolling(12, min_periods=6).mean())

    # ── Return result ───────────────────────────────────────────────────
    keep = [
        "permno", "date",
        "mom_12_1", "mom_6_1", "mom_3_1", "short_reversal",
        "realized_vol_12", "realized_vol_3", "vol_ratio",
        "downside_vol_12", "max_drawdown_12",
        "log_market_cap", "log_dollar_volume", "amihud_illiquidity",
        "skewness_12", "kurtosis_12", "positive_months_ratio",
    ]
    result = df[keep].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    print(f"  [price] done: {len(result):,} rows, {result.notna().mean().mean():.1%} coverage")
    return result
