"""
LightGBM and XGBoost model training for cross-sectional return prediction.

The model predicts cross-sectional rank of forward returns. Training uses
rank-normalised features and rank-normalised target to ensure the model
learns relative ordering rather than absolute return magnitudes.
"""

from __future__ import annotations

from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from config import LGBM_PARAMS, SEED


# ---------------------------------------------------------------------------
# GPU verification
# ---------------------------------------------------------------------------

def verify_gpu() -> None:
    """Verify that LightGBM can train on GPU and print device info."""
    try:
        X = np.random.RandomState(0).randn(100, 5)
        y = np.random.RandomState(0).randn(100)
        ds = lgb.Dataset(X, y)
        booster = lgb.train(
            {"device": "gpu", "num_leaves": 4, "verbosity": -1},
            ds, num_boost_round=2,
        )
        print(f"[GPU] LightGBM {lgb.__version__} -- GPU training verified OK")
    except Exception as e:
        print(f"[GPU] WARNING: GPU training failed ({e}), falling back to CPU")


# ---------------------------------------------------------------------------
# Custom evaluation metric
# ---------------------------------------------------------------------------

def spearman_ic_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[str, float, bool]:
    """LightGBM-compatible custom evaluation metric.

    Computes Spearman rank correlation between *y_true* and *y_pred*.

    Returns
    -------
    tuple[str, float, bool]
        ("spearman_ic", ic_value, True) where ``True`` means higher is better.
    """
    ic, _ = stats.spearmanr(y_true, y_pred)
    return ("spearman_ic", float(ic), True)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[dict] = None,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor with early stopping on a validation set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (rank-normalised forward returns).
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target.
    params : dict, optional
        LightGBM parameters. Falls back to ``LGBM_PARAMS`` from config.
    early_stopping_rounds : int
        Number of rounds without improvement before stopping.

    Returns
    -------
    lgb.LGBMRegressor
        Fitted model.
    """
    model_params = dict(LGBM_PARAMS) if params is None else dict(params)
    model_params.setdefault("random_state", SEED)
    model_params.setdefault("n_jobs", -1)
    model_params.setdefault("verbosity", -1)

    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=spearman_ic_eval,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(50),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

_XGB_DEFAULT_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": SEED,
    "verbosity": 0,
}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[dict] = None,
    early_stopping_rounds: int = 50,
) -> xgb.XGBRegressor:
    """Train an XGBoost regressor with early stopping on a validation set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (rank-normalised forward returns).
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target.
    params : dict, optional
        XGBoost parameters. Falls back to built-in defaults.
    early_stopping_rounds : int
        Number of rounds without improvement before stopping.

    Returns
    -------
    xgb.XGBRegressor
        Fitted model.
    """
    model_params = dict(_XGB_DEFAULT_PARAMS) if params is None else dict(params)
    model_params.setdefault("random_state", SEED)
    model_params.setdefault("n_jobs", -1)
    model_params.setdefault("verbosity", 0)

    model = xgb.XGBRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=50,
    )
    return model


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: lgb.LGBMRegressor | xgb.XGBRegressor,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Extract feature importance from a fitted LightGBM or XGBoost model.

    Parameters
    ----------
    model : lgb.LGBMRegressor | xgb.XGBRegressor
        A fitted tree model.
    feature_names : list[str]
        Ordered list of feature names matching the training data columns.
    importance_type : str
        Type of importance (e.g. ``'gain'``, ``'split'`` / ``'weight'``).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame ``[feature, importance]`` sorted descending by
        importance.
    """
    if isinstance(model, lgb.LGBMRegressor):
        importances = model.booster_.feature_importance(importance_type=importance_type)
    elif isinstance(model, xgb.XGBRegressor):
        booster = model.get_booster()
        score_map = booster.get_score(importance_type=importance_type)
        importances = np.array(
            [score_map.get(f"f{i}", score_map.get(name, 0.0))
             for i, name in enumerate(feature_names)]
        )
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df
