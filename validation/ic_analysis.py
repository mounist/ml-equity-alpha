"""
Information Coefficient (IC) analysis for model evaluation.

Computes cross-sectional Spearman rank-IC between model predictions
and realised forward returns. All metrics are computed on the OOS
test period only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config


def compute_monthly_ic(
    predictions: pd.Series,
    actuals: pd.Series,
    dates: pd.Series,
) -> pd.DataFrame:
    """Compute per-date cross-sectional Spearman rank-IC.

    Parameters
    ----------
    predictions : pd.Series
        Model predicted scores, aligned with *actuals* and *dates*.
    actuals : pd.Series
        Realised forward returns.
    dates : pd.Series
        Date label for each observation (typically month-end).

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``ic``, ``n_stocks``.
    """
    records: list[dict] = []
    for dt in sorted(dates.unique()):
        mask = dates == dt
        p = predictions.loc[mask]
        a = actuals.loc[mask]
        n = int(mask.sum())
        if n < 10:
            continue
        rho, _ = spearmanr(p, a)
        records.append({"date": dt, "ic": float(rho), "n_stocks": n})
    return pd.DataFrame(records)


def ic_summary(monthly_ic: pd.DataFrame) -> dict:
    """Compute summary statistics for a series of monthly ICs.

    Parameters
    ----------
    monthly_ic : pd.DataFrame
        Output of :func:`compute_monthly_ic` (must contain ``ic``
        column).

    Returns
    -------
    dict
        Keys: ``mean_ic``, ``std_ic``, ``t_stat``, ``icir``,
        ``hit_rate``, ``ci_lower``, ``ci_upper``, ``n_months``.
    """
    ics = monthly_ic["ic"].values
    n = len(ics)
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1)) if n > 1 else 0.0
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit_rate = float(np.mean(ics > 0))

    # Bootstrap 95 % CI (10,000 i.i.d. resamples)
    rng = np.random.RandomState(config.SEED)
    boot_means = np.array(
        [rng.choice(ics, size=n, replace=True).mean() for _ in range(10_000)]
    )
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "t_stat": float(t_stat),
        "icir": float(icir),
        "hit_rate": hit_rate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_months": n,
    }


def ic_decay(
    panel: pd.DataFrame,
    predictions: pd.Series,
    feature_cols_unused: list[str] | None = None,
) -> pd.DataFrame:
    """Compute IC at multiple forward-return horizons.

    Expects the panel to contain ``fwd_ret_1m``, ``fwd_ret_3m``, and
    ``fwd_ret_6m`` columns.

    Parameters
    ----------
    panel : pd.DataFrame
        OOS panel with date, forward return columns, and an index
        aligned with *predictions*.
    predictions : pd.Series
        Model predicted scores.
    feature_cols_unused : list[str], optional
        Unused; kept for call-signature compatibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``horizon``, ``mean_ic``, ``std_ic``, ``t_stat``.
    """
    horizons = {"1m": "fwd_ret_1m", "3m": "fwd_ret_3m", "6m": "fwd_ret_6m"}
    records: list[dict] = []

    for label, col in horizons.items():
        if col not in panel.columns:
            continue
        mic = compute_monthly_ic(predictions, panel[col], panel["date"])
        if mic.empty:
            continue
        ics = mic["ic"].values
        n = len(ics)
        mean = float(np.mean(ics))
        std = float(np.std(ics, ddof=1)) if n > 1 else 0.0
        t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
        records.append(
            {"horizon": label, "mean_ic": mean, "std_ic": std, "t_stat": float(t)}
        )

    return pd.DataFrame(records)


def compare_models_ic(
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target_col: str,
    train_fn,
    folds: list[dict] | None = None,
) -> pd.DataFrame:
    """Train multiple models using walk-forward CV and compare ICs.

    Uses the same walk-forward fold structure as the main pipeline for
    consistent methodology. Also computes OOS IC using a final model
    trained on IS data only (the last year of IS is held out for early
    stopping so that OOS data never leaks into model fitting).

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel with ``date`` column, feature columns, and target.
    feature_sets : dict[str, list[str]]
        Mapping of model name to its feature list.
    target_col : str
        Name of the target column.
    train_fn : callable
        ``train_fn(X_train, y_train, X_val, y_val)`` -> fitted model.
    folds : list[dict], optional
        Walk-forward folds from generate_cv_folds(). If None, generated
        from panel dates.

    Returns
    -------
    pd.DataFrame
        Columns: ``model``, ``cv_mean_ic``, ``cv_icir``, ``cv_hit_rate``,
        ``oos_mean_ic``, ``oos_icir``.
        Saved to ``RESULTS_DIR/ic_comparison.csv``.
    """
    from validation.walk_forward_cv import generate_cv_folds

    if folds is None:
        folds = generate_cv_folds(panel["date"])

    train_end_ts = pd.Timestamp(config.TRAIN_END)
    test_start_ts = pd.Timestamp(config.TEST_START)
    train_mask = panel["date"] <= train_end_ts
    test_mask = panel["date"] >= test_start_ts

    records: list[dict] = []
    for model_name, feat_cols in feature_sets.items():
        use_panel = panel

        # SUE-only model: filter to rows with actual SUE data
        if model_name == "SUE-only":
            if "sue" in panel.columns:
                sue_coverage = panel.loc[test_mask, "sue"].notna().mean()
                print(f"[IC compare] SUE coverage in test: {sue_coverage:.1%}")
                if sue_coverage < 0.20:
                    print(f"[IC compare] Warning: SUE coverage below 20% in test set")
                use_panel = panel[panel["sue"].notna()].copy()
                n_train_rows = use_panel.loc[use_panel["date"] <= train_end_ts].shape[0]
                if n_train_rows < 1000:
                    print(f"[IC compare] Skipping SUE-only: only {n_train_rows} "
                          f"training rows with actual SUE data")
                    continue
            else:
                print("[IC compare] Skipping SUE-only: 'sue' column not found")
                continue

        # Walk-forward CV IC
        all_cv_ics: list[float] = []
        for fold in folds:
            fold_train_mask = (
                (use_panel["date"] >= fold["train_start"])
                & (use_panel["date"] <= fold["train_end"])
            )
            fold_val_mask = (
                (use_panel["date"] >= fold["val_start"])
                & (use_panel["date"] <= fold["val_end"])
            )

            fold_train = use_panel.loc[fold_train_mask]
            fold_val = use_panel.loc[fold_val_mask]

            if fold_train.empty or fold_val.empty:
                continue

            X_tr = fold_train[feat_cols]
            y_tr = fold_train[target_col]
            X_va = fold_val[feat_cols]
            y_va = fold_val[target_col]

            model = train_fn(X_tr, y_tr, X_va, y_va)
            preds = pd.Series(model.predict(X_va), index=fold_val.index)

            mic = compute_monthly_ic(preds, y_va, fold_val["date"])
            all_cv_ics.extend(mic["ic"].tolist())

        if not all_cv_ics:
            continue

        cv_ics = np.array(all_cv_ics)
        cv_mean = float(cv_ics.mean())
        cv_std = float(cv_ics.std(ddof=1)) if len(cv_ics) > 1 else 0.0
        cv_icir = cv_mean / cv_std if cv_std > 0 else 0.0
        cv_hit = float((cv_ics > 0).mean())

        # OOS IC: train on IS data, evaluate on OOS.
        # Split IS into train/val for early stopping — use last year of
        # IS as the validation set so that OOS data never leaks into
        # the training process.
        oos_train_full = use_panel.loc[use_panel["date"] <= train_end_ts]
        oos_test = use_panel.loc[use_panel["date"] >= test_start_ts]

        oos_mean_ic = np.nan
        oos_icir = np.nan
        if not oos_train_full.empty and not oos_test.empty:
            val_year_start = train_end_ts - pd.DateOffset(years=1)
            is_train = oos_train_full.loc[oos_train_full["date"] <= val_year_start]
            is_val = oos_train_full.loc[oos_train_full["date"] > val_year_start]
            if is_val.empty:
                is_val = is_train.tail(len(is_train) // 5)
                is_train = is_train.iloc[: len(is_train) - len(is_val)]
            model = train_fn(
                is_train[feat_cols], is_train[target_col],
                is_val[feat_cols], is_val[target_col],
            )
            oos_preds = pd.Series(model.predict(oos_test[feat_cols]), index=oos_test.index)
            oos_mic = compute_monthly_ic(oos_preds, oos_test[target_col], oos_test["date"])
            if not oos_mic.empty:
                oos_summary = ic_summary(oos_mic)
                oos_mean_ic = oos_summary["mean_ic"]
                oos_icir = oos_summary["icir"]

        records.append({
            "model": model_name,
            "cv_mean_ic": cv_mean,
            "cv_icir": cv_icir,
            "cv_hit_rate": cv_hit,
            "oos_mean_ic": oos_mean_ic,
            "oos_icir": oos_icir,
        })

    results = pd.DataFrame(records)
    results.to_csv(config.RESULTS_DIR / "ic_comparison.csv", index=False)
    return results
