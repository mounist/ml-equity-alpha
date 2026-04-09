"""
Optuna hyperparameter tuning for LightGBM.

Searches over tree structure and regularisation parameters, using
mean rank-IC across walk-forward CV folds as the objective.
Only tunes on validation folds -- never on the OOS test period.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy import stats

from config import LGBM_PARAMS, N_OPTUNA_TRIALS, RESULTS_DIR, SEED


def _objective(trial: optuna.Trial, fold_data: list[dict]) -> float:
    """Optuna objective: mean Spearman IC across walk-forward CV folds.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.
    fold_data : list[dict]
        Each dict contains keys ``X_train``, ``y_train``, ``X_val``,
        ``y_val`` (DataFrames / Series for one CV fold).

    Returns
    -------
    float
        Mean Spearman IC across all folds (higher is better).
    """
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }

    # Merge with base params (trial suggestions override)
    model_params = dict(LGBM_PARAMS)
    model_params.update(params)
    model_params.setdefault("random_state", SEED)
    model_params.setdefault("n_jobs", -1)
    model_params.setdefault("verbosity", -1)

    ics: list[float] = []

    for fold in fold_data:
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_val = fold["X_val"]
        y_val = fold["y_val"]

        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0),  # suppress per-round output
            ],
        )

        y_pred = model.predict(X_val)
        ic, _ = stats.spearmanr(y_val, y_pred)
        ics.append(float(ic))

    return float(np.mean(ics))


def run_optuna_tuning(
    fold_data: list[dict],
    n_trials: Optional[int] = None,
) -> dict:
    """Run Optuna hyperparameter search for LightGBM.

    Parameters
    ----------
    fold_data : list[dict]
        Walk-forward CV folds. Each dict must contain ``X_train``,
        ``y_train``, ``X_val``, ``y_val``.
    n_trials : int, optional
        Number of Optuna trials. Defaults to ``N_OPTUNA_TRIALS`` from config.

    Returns
    -------
    dict
        Best parameters merged with the base ``LGBM_PARAMS``.
    """
    if n_trials is None:
        n_trials = N_OPTUNA_TRIALS

    # Suppress verbose Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: _objective(trial, fold_data), n_trials=n_trials)

    # Merge best trial params with base config
    best_params = dict(LGBM_PARAMS)
    best_params.update(study.best_params)

    # Persist to disk
    out_path = Path(RESULTS_DIR) / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    logging.info("Best Optuna IC=%.4f  params=%s", study.best_value, study.best_params)

    return best_params
