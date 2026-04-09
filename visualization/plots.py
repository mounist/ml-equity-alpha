"""
Visualisation utilities for the ML equity alpha pipeline.

All plots are saved to FIGURES_DIR with dpi=150.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns            # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

import config                    # noqa: E402

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

_DPI = 150
_TEST_START = pd.Timestamp(config.TEST_START)


# ---------------------------------------------------------------------------
# 1. Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> None:
    """Cumulative return chart for gross, net, and (optional) benchmark.

    Parameters
    ----------
    portfolio_returns : DataFrame
        Must contain columns: date, gross_ls, net_ls.
    benchmark_returns : Series, optional
        Market or other benchmark monthly returns indexed by date.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    dates = pd.to_datetime(portfolio_returns["date"])
    gross_cum = (1 + portfolio_returns["gross_ls"]).cumprod()
    net_cum = (1 + portfolio_returns["net_ls"]).cumprod()

    ax.plot(dates, gross_cum, label="L/S Gross", linewidth=1.5)
    ax.plot(dates, net_cum, label="L/S Net", linewidth=1.5, linestyle="--")

    if benchmark_returns is not None:
        bench_aligned = benchmark_returns.reindex(dates)
        bench_cum = (1 + bench_aligned.fillna(0)).cumprod()
        ax.plot(dates, bench_cum, label="Benchmark", linewidth=1.0, color="grey")

    ax.axvline(_TEST_START, color="black", linestyle=":", linewidth=0.8, label="OOS start")
    ax.set_title("Cumulative Returns: Long-Short Portfolio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "equity_curve.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. IC time series
# ---------------------------------------------------------------------------

def plot_ic_time_series(monthly_ic: pd.DataFrame) -> None:
    """Monthly information coefficient bar chart with rolling mean.

    Parameters
    ----------
    monthly_ic : DataFrame
        Columns: date, ic.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    dates = pd.to_datetime(monthly_ic["date"])
    ic = monthly_ic["ic"].values

    ax.bar(dates, ic, width=25, color="lightblue", edgecolor="none", label="Monthly IC")

    rolling_mean = pd.Series(ic, index=dates).rolling(12, min_periods=1).mean()
    ax.plot(dates, rolling_mean, color="red", linewidth=1.5, label="12-month MA")

    mean_ic = np.nanmean(ic)
    t_stat = mean_ic / (np.nanstd(ic, ddof=1) / np.sqrt(np.sum(~np.isnan(ic))))

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axhline(mean_ic, color="navy", linestyle="--", linewidth=0.8, label=f"Mean IC = {mean_ic:.4f}")
    ax.axvline(_TEST_START, color="black", linestyle=":", linewidth=0.8, label="OOS start")

    ax.set_title("Monthly Rank IC")
    ax.text(
        0.02, 0.95, f"Mean IC = {mean_ic:.4f}  |  t = {t_stat:.2f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Spearman IC")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "ic_time_series.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Cross-validation IC bar chart
# ---------------------------------------------------------------------------

def plot_cv_ic_bar(cv_results: pd.DataFrame) -> None:
    """Bar chart of rank IC per CV fold.

    Parameters
    ----------
    cv_results : DataFrame
        Columns: fold_id, val_year, mean_ic.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"Fold {int(r.fold_id)} ({int(r.val_year)})" for _, r in cv_results.iterrows()]
    ics = cv_results["mean_ic"].values

    bars = ax.bar(labels, ics, color="steelblue", edgecolor="white")

    overall_mean = np.nanmean(ics)
    ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1, label=f"Mean = {overall_mean:.4f}")

    ax.set_title("Cross-Validation: Mean Rank IC per Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Mean IC")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cv_ic_bar.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Feature coverage heatmap
# ---------------------------------------------------------------------------

def plot_feature_coverage(coverage: pd.DataFrame) -> None:
    """Heatmap of feature non-NaN rates across years.

    Parameters
    ----------
    coverage : DataFrame
        Rows = features, columns = years, values = fraction in [0, 1].
    """
    n_features = len(coverage)
    fig_h = max(6, 0.4 * n_features)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    sns.heatmap(
        coverage,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Coverage (non-NaN fraction)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_coverage.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Decile returns bar chart
# ---------------------------------------------------------------------------

def plot_decile_returns(decile_stats: pd.DataFrame) -> None:
    """Mean return by decile (Q1 through Q10).

    Parameters
    ----------
    decile_stats : DataFrame
        Must contain columns q1 .. q10.  Mean across all dates is plotted.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    q_cols = [c for c in decile_stats.columns if c.startswith("q") and c[1:].isdigit()]
    q_cols = sorted(q_cols, key=lambda c: int(c[1:]))
    means = decile_stats[q_cols].mean()

    labels = [c.upper() for c in q_cols]
    colors = ["#d62728" if i == 0 else "#2ca02c" if i == len(q_cols) - 1 else "steelblue"
              for i in range(len(q_cols))]

    ax.bar(labels, means.values, color=colors, edgecolor="white")
    ax.set_title("Mean Monthly Return by Decile")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Mean Return")
    ax.axhline(0, color="grey", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "decile_returns.png", dpi=_DPI)
    plt.close(fig)
