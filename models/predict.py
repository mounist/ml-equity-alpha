"""
Model prediction utilities.

Generates cross-sectional predictions and computes rank scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_scores(model, X: pd.DataFrame) -> pd.Series:
    """Generate raw model predictions.

    Parameters
    ----------
    model
        A fitted model exposing a ``.predict()`` method (e.g. LightGBM or
        XGBoost regressor).
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.Series
        Raw predictions with the same index as *X*.
    """
    preds = model.predict(X)
    return pd.Series(preds, index=X.index, name="score")


def predict_rank_scores(
    model,
    X: pd.DataFrame,
    dates: pd.Series,
) -> pd.Series:
    """Generate cross-sectionally ranked prediction scores.

    For each unique date in *dates*, the raw predictions are ranked and
    scaled to the ``[0, 1]`` interval (percentile rank).

    Parameters
    ----------
    model
        A fitted model exposing a ``.predict()`` method.
    X : pd.DataFrame
        Feature matrix.
    dates : pd.Series
        Series of dates aligned with the rows of *X* (same index).

    Returns
    -------
    pd.Series
        Rank-percentile scores in ``[0, 1]`` with the same index as *X*.
    """
    raw = predict_scores(model, X)

    rank_scores = pd.Series(np.nan, index=X.index, name="rank_score")

    for date in dates.unique():
        mask = dates == date
        group = raw.loc[mask]
        if len(group) <= 1:
            rank_scores.loc[mask] = 0.5
        else:
            rank_scores.loc[mask] = group.rank(pct=True)

    return rank_scores
