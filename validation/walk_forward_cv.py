"""
Purged walk-forward cross-validation for time-series ML.

Implements an expanding-window walk-forward scheme with a purge gap
between training and validation to prevent information leakage from
overlapping return windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config


def generate_cv_folds(
    dates: pd.Series,
    min_train_years: int | None = None,
    val_window: int | None = None,
    refit_freq: int | None = None,
    purge_gap: int | None = None,
    train_end: str | None = None,
) -> list[dict]:
    """Generate expanding-window walk-forward folds with a purge gap.

    Parameters
    ----------
    dates : pd.Series
        Series of datetime dates present in the panel.
    min_train_years : int, optional
        Minimum years of training data required. Defaults to
        ``config.CV_MIN_TRAIN_YEARS``.
    val_window : int, optional
        Length of validation window in months. Defaults to
        ``config.CV_VAL_WINDOW``.
    refit_freq : int, optional
        Refit frequency in months. Defaults to ``config.CV_REFIT_FREQ``.
    purge_gap : int, optional
        Number of months to skip between training end and validation
        start. Defaults to ``config.PURGE_GAP``.
    train_end : str, optional
        Latest date (inclusive) that may appear in any validation fold.
        Defaults to ``config.TRAIN_END``.

    Returns
    -------
    list[dict]
        Each dict contains: ``fold_id``, ``train_start``, ``train_end``,
        ``val_start``, ``val_end``.
    """
    min_train_years = min_train_years or config.CV_MIN_TRAIN_YEARS
    val_window = val_window or config.CV_VAL_WINDOW
    refit_freq = refit_freq or config.CV_REFIT_FREQ
    purge_gap = purge_gap or config.PURGE_GAP
    train_end_dt = pd.Timestamp(train_end or config.TRAIN_END)

    start_year = pd.Timestamp(config.START_DATE).year
    first_val_year = start_year + min_train_years  # e.g. 2005

    folds: list[dict] = []
    fold_id = 0

    for val_year in range(first_val_year, train_end_dt.year + 1):
        val_start = pd.Timestamp(f"{val_year}-01-01")
        val_end = pd.Timestamp(f"{val_year}-12-31")

        # Ensure val_end does not exceed the overall train_end boundary
        if val_end > train_end_dt:
            val_end = train_end_dt

        # Training ends purge_gap months before validation starts
        train_cutoff = val_start - pd.DateOffset(months=purge_gap)
        # Subtract one more day so train_end is strictly before the gap
        train_cutoff = train_cutoff - pd.Timedelta(days=1)

        fold_id += 1
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": pd.Timestamp(config.START_DATE),
                "train_end": train_cutoff,
                "val_start": val_start,
                "val_end": val_end,
            }
        )

    return folds


def run_walk_forward_cv(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_fn,
    folds: list[dict] | None = None,
) -> pd.DataFrame:
    """Execute walk-forward CV and collect per-fold Spearman IC results.

    Parameters
    ----------
    panel : pd.DataFrame
        Full training panel with a ``date`` column, feature columns, and
        the target column.
    feature_cols : list[str]
        Names of the feature columns.
    target_col : str
        Name of the target (forward return) column.
    train_fn : callable
        ``train_fn(X_train, y_train, X_val, y_val)`` -> fitted model
        with a ``.predict`` method and a ``.feature_importances_``
        attribute.
    folds : list[dict], optional
        Pre-computed folds from :func:`generate_cv_folds`. If *None*,
        folds are generated from ``panel["date"]``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Per-fold results with columns: ``fold_id``, ``val_year``,
          ``mean_ic``, ``std_ic``, ``n_dates``, ``feature_importances``.
        - Out-of-fold predictions with columns: ``permno``, ``date``,
          ``oof_pred``, ``actual``.
    """
    if folds is None:
        folds = generate_cv_folds(panel["date"])

    records: list[dict] = []
    oof_parts: list[pd.DataFrame] = []

    for fold in folds:
        train_mask = (panel["date"] >= fold["train_start"]) & (
            panel["date"] <= fold["train_end"]
        )
        val_mask = (panel["date"] >= fold["val_start"]) & (
            panel["date"] <= fold["val_end"]
        )

        train_df = panel.loc[train_mask]
        val_df = panel.loc[val_mask]

        if train_df.empty or val_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        model = train_fn(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)

        # Store out-of-fold predictions
        oof_rows = val_df[["permno", "date"]].copy()
        oof_rows["oof_pred"] = preds
        oof_rows["actual"] = y_val.values
        oof_parts.append(oof_rows)

        # Per-date cross-sectional Spearman IC
        val_dates = val_df["date"].values
        ics: list[float] = []
        for dt in np.unique(val_dates):
            dt_mask = val_dates == dt
            p = preds[dt_mask]
            a = y_val.values[dt_mask]
            if len(p) < 10:
                continue
            rho, _ = spearmanr(p, a)
            if np.isfinite(rho):
                ics.append(rho)

        mean_ic = float(pd.Series(ics).mean()) if ics else 0.0
        std_ic = float(pd.Series(ics).std()) if ics else 0.0

        # Feature importances
        fi = {}
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(feature_cols, model.feature_importances_.tolist()))

        records.append(
            {
                "fold_id": fold["fold_id"],
                "val_year": fold["val_start"].year,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "n_dates": len(ics),
                "feature_importances": fi,
            }
        )

    results = pd.DataFrame(records)
    results.to_csv(config.RESULTS_DIR / "cv_fold_ic.csv", index=False)

    oof_predictions = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(
        columns=["permno", "date", "oof_pred", "actual"]
    )

    return results, oof_predictions
