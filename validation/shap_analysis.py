"""
SHAP-based model interpretability analysis.

Uses TreeExplainer to compute SHAP values for the final LightGBM model,
providing feature importance rankings, sign consistency checks, and
stability analysis across time periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr

import config


def compute_shap_values(model, X_sample: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model
        A fitted tree-based model (e.g. LightGBM).
    X_sample : pd.DataFrame
        Feature matrix to explain (typically a subsample of size
        ``config.SHAP_SAMPLE_N``).

    Returns
    -------
    np.ndarray
        SHAP values array of shape ``(n_samples, n_features)``.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # For regressors shap_values is already an ndarray; for classifiers
    # it may be a list — take the positive-class slice if so.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return np.asarray(shap_values)


def shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array ``(n_samples, n_features)``.
    feature_names : list[str]
        Feature names corresponding to the columns of *shap_values*.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``mean_abs_shap``, sorted descending.
        Saved to ``RESULTS_DIR/feature_importance_shap.csv``.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    df.to_csv(config.RESULTS_DIR / "feature_importance_shap.csv", index=False)
    return df


def shap_sign_consistency(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    expected_signs: dict[str, int],
) -> pd.DataFrame:
    """Check whether SHAP values agree with expected directional signs.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values ``(n_samples, n_features)``.
    X_sample : pd.DataFrame
        Feature matrix (unused in current implementation but available
        for conditional analyses).
    feature_names : list[str]
        Feature names aligned with *shap_values* columns.
    expected_signs : dict[str, int]
        Mapping from feature name to expected sign (+1 or -1).  For
        example ``{"mom_12_1_rank": 1, "short_reversal_rank": -1}``.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``expected_sign``, ``sign_consistency``.
    """
    records: list[dict] = []
    for feat, sign in expected_signs.items():
        if feat not in feature_names:
            continue
        idx = feature_names.index(feat)
        sv = shap_values[:, idx]
        if sign == 1:
            consistency = float(np.mean(sv > 0))
        else:
            consistency = float(np.mean(sv < 0))
        records.append(
            {
                "feature": feat,
                "expected_sign": sign,
                "sign_consistency": consistency,
            }
        )
    return pd.DataFrame(records)


def shap_stability(
    shap_values: np.ndarray,
    dates: pd.Series,
    feature_names: list[str],
    split_date: str = "2022-01-01",
) -> pd.DataFrame:
    """Assess stability of SHAP-based feature rankings across sub-periods.

    Splits OOS observations into two sub-periods and compares the
    mean |SHAP| ranking of each feature via Spearman correlation.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values ``(n_samples, n_features)``.
    dates : pd.Series
        Date for each observation, aligned with rows of *shap_values*.
    feature_names : list[str]
        Feature names.
    split_date : str
        Date string that separates the two sub-periods. Defaults to
        ``"2022-01-01"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``rank_period1``, ``rank_period2``.  The
        overall Spearman rank correlation is stored in the DataFrame
        ``attrs["rank_corr"]``.  Also saved to
        ``RESULTS_DIR/shap_stability.csv``.
    """
    split_ts = pd.Timestamp(split_date)
    dates_arr = pd.to_datetime(dates).values

    mask_p1 = dates_arr < np.datetime64(split_ts)
    mask_p2 = ~mask_p1

    mean_abs_p1 = np.mean(np.abs(shap_values[mask_p1]), axis=0)
    mean_abs_p2 = np.mean(np.abs(shap_values[mask_p2]), axis=0)

    rank_p1 = pd.Series(mean_abs_p1).rank(ascending=False).astype(int).values
    rank_p2 = pd.Series(mean_abs_p2).rank(ascending=False).astype(int).values

    rho, _ = spearmanr(rank_p1, rank_p2)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "rank_period1": rank_p1,
            "rank_period2": rank_p2,
        }
    )
    df.attrs["rank_corr"] = float(rho)
    df.to_csv(config.RESULTS_DIR / "shap_stability.csv", index=False)
    return df


def plot_shap(
    model,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    shap_values: np.ndarray | None = None,
) -> None:
    """Generate and save SHAP visualisation plots.

    Produces three figures:

    1. ``FIGURES_DIR/shap_summary.png`` — beeswarm summary plot.
    2. ``FIGURES_DIR/shap_importance_bar.png`` — bar chart of mean |SHAP|.
    3. ``FIGURES_DIR/shap_dependence_top5.png`` — dependence plots for
       the five most important features arranged in a 2x3 grid (last
       subplot left empty).

    Parameters
    ----------
    model
        Fitted tree-based model.
    X_sample : pd.DataFrame
        Feature matrix used for explanation.
    feature_names : list[str]
        Feature names.
    shap_values : np.ndarray, optional
        Pre-computed SHAP values.  Computed on the fly if *None*.
    """
    if shap_values is None:
        shap_values = compute_shap_values(model, X_sample)

    X_display = X_sample.copy()
    X_display.columns = feature_names

    # 1. Beeswarm summary
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close("all")

    # 2. Bar chart of mean |SHAP|
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_importance_bar.png", dpi=150)
    plt.close("all")

    # 3. Dependence plots for top-5 features
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top5_idx = np.argsort(mean_abs)[::-1][:5]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, feat_idx in enumerate(top5_idx):
        ax = axes_flat[i]
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_display,
            ax=ax,
            show=False,
        )
        ax.set_title(feature_names[feat_idx])

    # Leave the last subplot empty
    axes_flat[5].axis("off")

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_dependence_top5.png", dpi=150)
    plt.close("all")
