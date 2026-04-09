"""
ML Equity Alpha — end-to-end pipeline.

Stages: data → features → train → validate → portfolio → all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── Reproducibility ──────────────────────────────────────────────────────────

def _seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        pass


# ── Stage: data ──────────────────────────────────────────────────────────────

def stage_data(ctx: dict) -> dict:
    """Pull all datasets from WRDS (or parquet cache)."""
    t0 = time.time()
    from data.wrds_loader import load_all_data

    data = load_all_data()
    for k, v in data.items():
        log.info("  %-20s %s rows", k, f"{len(v):,}")
    ctx["data"] = data
    log.info("stage_data finished in %.1fs", time.time() - t0)
    return ctx


# ── Stage: features ──────────────────────────────────────────────────────────

def stage_features(ctx: dict) -> dict:
    """Build the feature panel from raw data."""
    t0 = time.time()
    from features.feature_pipeline import build_feature_panel, get_feature_cols

    panel = build_feature_panel(ctx["data"])
    ctx["panel"] = panel
    ctx["feature_cols"] = get_feature_cols()
    log.info(
        "Panel: %s rows × %d cols, features: %d",
        f"{len(panel):,}", len(panel.columns), len(ctx["feature_cols"]),
    )

    # Persist for re-entry
    panel.to_parquet(config.DATA_DIR / "panel.parquet", index=False)
    log.info("stage_features finished in %.1fs", time.time() - t0)
    return ctx


# ── Stage: train ─────────────────────────────────────────────────────────────

def stage_train(ctx: dict) -> dict:
    """Walk-forward CV, Optuna tuning, final model fit."""
    t0 = time.time()

    from features.feature_pipeline import get_feature_cols
    from models.train import train_lightgbm, get_feature_importance, verify_gpu
    from validation.walk_forward_cv import generate_cv_folds, run_walk_forward_cv
    from models.hyperparameter_tuning import run_optuna_tuning

    verify_gpu()

    panel = ctx["panel"]
    feature_cols = ctx.get("feature_cols", get_feature_cols())
    target = "fwd_ret_1m"

    # Drop rows with missing target
    panel_valid = panel.dropna(subset=[target]).copy()

    # ── Walk-forward CV ──────────────────────────────────────────────────
    log.info("Generating CV folds ...")
    folds = generate_cv_folds(panel_valid["date"])
    log.info("  %d folds generated", len(folds))
    ctx["folds"] = folds

    log.info("Running walk-forward CV ...")
    cv_results, oof_predictions = run_walk_forward_cv(
        panel_valid, feature_cols, target, train_lightgbm, folds=folds,
    )
    mean_cv_ic = cv_results["mean_ic"].mean()
    log.info("  Mean CV IC: %.4f", mean_cv_ic)
    ctx["cv_results"] = cv_results

    # Save out-of-fold predictions for honest IS backtest
    oof_path = config.RESULTS_DIR / "oof_predictions.parquet"
    oof_predictions.to_parquet(oof_path, index=False)
    log.info("  OOF predictions saved: %d rows", len(oof_predictions))
    ctx["oof_predictions"] = oof_predictions

    # ── Optuna tuning ────────────────────────────────────────────────────
    log.info("Running Optuna hyperparameter tuning (%d trials) ...",
             config.N_OPTUNA_TRIALS)

    # Prepare fold data for Optuna
    fold_data = []
    for fold in folds:
        train_mask = (
            (panel_valid["date"] >= fold["train_start"])
            & (panel_valid["date"] <= fold["train_end"])
        )
        val_mask = (
            (panel_valid["date"] >= fold["val_start"])
            & (panel_valid["date"] <= fold["val_end"])
        )
        tr = panel_valid.loc[train_mask]
        va = panel_valid.loc[val_mask]
        if tr.empty or va.empty:
            continue
        fold_data.append({
            "X_train": tr[feature_cols],
            "y_train": tr[target],
            "X_val": va[feature_cols],
            "y_val": va[target],
        })

    # Use all folds for Optuna (reduce N_OPTUNA_TRIALS in config if runtime is a concern)
    best_params = run_optuna_tuning(fold_data)
    log.info("  Best params: %s", json.dumps(
        {k: round(v, 4) if isinstance(v, float) else v
         for k, v in best_params.items()},
        indent=2,
    ))
    ctx["best_params"] = best_params

    # ── Final model: train on all data through TRAIN_END ─────────────────
    log.info("Training final model on all data through %s ...", config.TRAIN_END)
    train_end = pd.Timestamp(config.TRAIN_END)
    test_start = pd.Timestamp(config.TEST_START)

    train_mask = panel_valid["date"] <= train_end
    # Use last CV year as validation for early stopping
    val_year_start = train_end - pd.DateOffset(years=1)
    val_mask_final = (panel_valid["date"] > val_year_start) & train_mask
    train_mask_final = (panel_valid["date"] <= val_year_start)

    X_tr = panel_valid.loc[train_mask_final, feature_cols]
    y_tr = panel_valid.loc[train_mask_final, target]
    X_va = panel_valid.loc[val_mask_final, feature_cols]
    y_va = panel_valid.loc[val_mask_final, target]

    model = train_lightgbm(X_tr, y_tr, X_va, y_va, params=best_params)
    ctx["model"] = model

    # Feature importance
    fi = get_feature_importance(model, feature_cols)
    fi.to_csv(config.RESULTS_DIR / "feature_importance_gain.csv", index=False)
    log.info("  Top-5 features (gain): %s",
             fi.head(5)["feature"].tolist())

    # ── OOS predictions ──────────────────────────────────────────────────
    oos_mask = panel_valid["date"] >= test_start
    X_oos = panel_valid.loc[oos_mask, feature_cols]
    preds = pd.Series(model.predict(X_oos), index=X_oos.index, name="pred")
    ctx["oos_preds"] = preds
    ctx["oos_panel"] = panel_valid.loc[oos_mask].copy()
    ctx["oos_panel"]["pred"] = preds.values

    # Build combined predictions panel: OOF for IS, final model for OOS
    from portfolio.backtest import build_full_predictions_panel
    panel_valid = build_full_predictions_panel(panel_valid, oof_predictions, preds, oos_mask)
    ctx["panel_with_preds"] = panel_valid

    # Save model and predictions for stage re-entry
    import joblib
    joblib.dump(model, config.RESULTS_DIR / "final_model.pkl")
    panel_valid.to_parquet(config.DATA_DIR / "panel_with_preds.parquet", index=False)
    log.info("  Model and predictions saved")

    log.info("stage_train finished in %.1fs", time.time() - t0)
    return ctx


# ── Stage: validate ──────────────────────────────────────────────────────────

def stage_validate(ctx: dict) -> dict:
    """IC analysis, SHAP, factor regression, model comparison."""
    t0 = time.time()

    from features.feature_pipeline import (
        get_feature_cols, PRICE_FEATURE_COLS, FUNDAMENTAL_FEATURE_COLS,
        QUALITY_FEATURE_COLS,
    )
    from models.train import train_lightgbm
    from validation.ic_analysis import (
        compute_monthly_ic, ic_summary, ic_decay, compare_models_ic,
    )
    from validation.shap_analysis import (
        compute_shap_values, shap_feature_importance,
        shap_sign_consistency, shap_stability, plot_shap,
    )
    from visualization.plots import plot_ic_time_series, plot_cv_ic_bar

    # Load model/preds from disk if not in context
    if "model" not in ctx:
        import joblib
        ctx["model"] = joblib.load(config.RESULTS_DIR / "final_model.pkl")
        log.info("Loaded model from disk")
    if "panel_with_preds" not in ctx:
        ctx["panel_with_preds"] = pd.read_parquet(config.DATA_DIR / "panel_with_preds.parquet")
        log.info("Loaded panel_with_preds from disk")
    if "oos_panel" not in ctx:
        panel_wp = ctx["panel_with_preds"]
        ctx["oos_panel"] = panel_wp[panel_wp["date"] >= pd.Timestamp(config.TEST_START)].copy()
    feature_cols = ctx.get("feature_cols", get_feature_cols())

    model = ctx["model"]
    oos_panel = ctx["oos_panel"]
    target = "fwd_ret_1m"

    # ── Monthly IC ───────────────────────────────────────────────────────
    log.info("Computing OOS monthly IC ...")
    monthly_ic = compute_monthly_ic(
        oos_panel["pred"], oos_panel[target], oos_panel["date"],
    )
    monthly_ic.to_csv(config.RESULTS_DIR / "oos_ic_monthly.csv", index=False)
    summary = ic_summary(monthly_ic)
    log.info("  OOS IC: mean=%.4f, t=%.2f, ICIR=%.3f, hit=%.1f%%",
             summary["mean_ic"], summary["t_stat"],
             summary["icir"], summary["hit_rate"] * 100)
    ctx["ic_summary"] = summary

    # ── IC decay ─────────────────────────────────────────────────────────
    log.info("Computing IC decay ...")
    decay = ic_decay(oos_panel, oos_panel["pred"])
    log.info("  IC decay:\n%s", decay.to_string(index=False))

    # ── CV IC bar chart ──────────────────────────────────────────────────
    if "cv_results" in ctx:
        plot_cv_ic_bar(ctx["cv_results"])

    # ── IC time series plot ──────────────────────────────────────────────
    plot_ic_time_series(monthly_ic)

    # ── Model comparison (IC decomposition) ──────────────────────────────
    log.info("Training sub-models for IC decomposition ...")
    panel_valid = ctx["panel_with_preds"].dropna(subset=[target])
    price_rank = [f"{c}_rank" for c in PRICE_FEATURE_COLS]
    fund_rank = [f"{c}_rank" for c in FUNDAMENTAL_FEATURE_COLS]
    sue_rank = [f"{c}_rank" for c in QUALITY_FEATURE_COLS]

    feature_sets = {
        "Full LightGBM": feature_cols,
        "Price-only": price_rank,
        "Fundamental-only": fund_rank,
        "SUE-only": sue_rank,
    }
    # Load folds from context or regenerate for consistent walk-forward eval
    folds = ctx.get("folds")
    if folds is None:
        from validation.walk_forward_cv import generate_cv_folds
        folds = generate_cv_folds(panel_valid["date"])

    ic_comp = compare_models_ic(
        panel_valid, feature_sets, target, train_lightgbm, folds=folds,
    )
    log.info("  IC comparison:\n%s", ic_comp.to_string(index=False))

    # ── SHAP analysis ────────────────────────────────────────────────────
    log.info("Computing SHAP values ...")
    n_shap = min(config.SHAP_SAMPLE_N, len(oos_panel))
    sample_idx = oos_panel.sample(n_shap, random_state=config.SEED).index
    X_sample = oos_panel.loc[sample_idx, feature_cols]

    shap_vals = compute_shap_values(model, X_sample)
    shap_imp = shap_feature_importance(shap_vals, feature_cols)
    log.info("  Top-5 SHAP features: %s", shap_imp.head(5)["feature"].tolist())

    # Sign consistency
    expected_signs = {
        "mom_12_1_rank": 1, "mom_6_1_rank": 1, "mom_3_1_rank": 1,
        "short_reversal_rank": -1,
        "book_to_market_rank": 1, "earnings_yield_rank": 1,
        "gross_profitability_rank": 1, "roa_rank": 1,
        "sue_rank": 1,
        "accruals_rank": -1,
        "asset_growth_1y_rank": -1,
    }
    sign_cons = shap_sign_consistency(
        shap_vals, X_sample, feature_cols, expected_signs,
    )

    # Stability
    sample_dates = oos_panel.loc[sample_idx, "date"]
    stability = shap_stability(shap_vals, sample_dates, feature_cols)

    # Plots
    plot_shap(model, X_sample, feature_cols, shap_vals)

    # Factor attribution moved to stage_portfolio (requires backtest output)

    log.info("stage_validate finished in %.1fs", time.time() - t0)
    return ctx


# ── Stage: portfolio ─────────────────────────────────────────────────────────

def stage_portfolio(ctx: dict) -> dict:
    """Backtest and performance reporting."""
    t0 = time.time()

    from portfolio.backtest import run_backtest
    from visualization.plots import plot_equity_curve, plot_decile_returns

    if "panel_with_preds" not in ctx:
        ctx["panel_with_preds"] = pd.read_parquet(config.DATA_DIR / "panel_with_preds.parquet")
        log.info("Loaded panel_with_preds from disk")

    panel = ctx["panel_with_preds"]

    # Load raw CRSP data if not in context (for siccd and market returns)
    if "data" not in ctx:
        crsp_msf = pd.read_parquet(config.DATA_DIR / "crsp_msf.parquet")
        crsp_msf["date"] = pd.to_datetime(crsp_msf["date"])
        crsp_msi = pd.read_parquet(config.DATA_DIR / "crsp_msi.parquet")
        crsp_msi["date"] = pd.to_datetime(crsp_msi["date"])
        ctx["data"] = {"crsp_msf": crsp_msf, "crsp_msi": crsp_msi}
        # Load FF factors if available (use loader to handle unit scaling)
        ff_path = config.DATA_DIR / "ff_monthly.parquet"
        if ff_path.exists():
            from data.wrds_loader import load_ff_monthly
            ctx["data"]["ff_monthly"] = load_ff_monthly()

    # Merge market returns for benchmark
    if "crsp_msi" in ctx.get("data", {}):
        msi = ctx["data"]["crsp_msi"][["date", "vwretd"]].copy()
        if "vwretd" not in panel.columns:
            panel = panel.merge(msi, on="date", how="left")
            ctx["panel_with_preds"] = panel

    # Merge siccd if not present
    if "siccd" not in panel.columns:
        crsp = ctx["data"]["crsp_msf"][["permno", "date", "siccd"]].copy()
        crsp["permno"] = crsp["permno"].astype(int)
        panel = panel.merge(crsp[["permno", "date", "siccd"]],
                            on=["permno", "date"], how="left")
        ctx["panel_with_preds"] = panel

    # Filter to rows with valid predictions (OOF covers 2005+, OOS covers 2019+)
    panel_bt = panel.dropna(subset=["pred"]).copy()
    bt = run_backtest(panel_bt, score_col="pred")

    log.info("=== Portfolio Performance (OOS) ===")
    for period_key in ["oos_gross", "oos_net"]:
        stats = bt["perf"].get(period_key, {})
        log.info(
            "  %-12s  ret=%+.2f%%  vol=%.2f%%  SR=%.2f  MaxDD=%.1f%%",
            period_key,
            stats.get("ann_ret", 0) * 100,
            stats.get("ann_vol", 0) * 100,
            stats.get("sharpe", 0),
            stats.get("max_drawdown", 0) * 100,
        )
    # IS stats are out-of-fold estimates, not in-sample — log at lower priority
    log.info("  (IS stats below are out-of-fold estimates, not in-sample)")
    for period_key in ["is_gross", "is_net"]:
        stats = bt["perf"].get(period_key, {})
        log.info(
            "  %-12s  ret=%+.2f%%  vol=%.2f%%  SR=%.2f  MaxDD=%.1f%%",
            period_key,
            stats.get("ann_ret", 0) * 100,
            stats.get("ann_vol", 0) * 100,
            stats.get("sharpe", 0),
            stats.get("max_drawdown", 0) * 100,
        )

    # Turnover stats (Bug 4 fix)
    turnover = bt["turnover"]
    test_start_ts = pd.Timestamp(config.TEST_START)
    oos_turnover = turnover.loc[turnover.index >= test_start_ts]
    if len(oos_turnover) > 0:
        avg_turnover = oos_turnover.mean()
        avg_tc_drag = avg_turnover * 2.0 * config.TC_ONE_WAY_BPS / 10_000 * 12
        log.info("[backtest] OOS avg monthly turnover: %.1f%%", avg_turnover * 100)
        log.info("[backtest] OOS avg annualised TC drag: %.2f%%", avg_tc_drag * 100)
        bt["perf"]["turnover_stats"] = {
            "oos_avg_monthly_turnover": float(avg_turnover),
            "oos_avg_annualised_tc_drag": float(avg_tc_drag),
        }
        # Re-save portfolio_stats.json with turnover stats
        import json as _json
        with open(config.RESULTS_DIR / "portfolio_stats.json", "w") as f:
            _json.dump(bt["perf"], f, indent=2)

    # ── Factor attribution (runs AFTER backtest so data is fresh) ─────────
    if "ff_monthly" in ctx.get("data", {}):
        from validation.factor_regression import run_factor_attribution

        monthly_df_bt = bt["monthly_df"]
        monthly_df_bt["date"] = pd.to_datetime(monthly_df_bt["date"])
        oos_monthly = monthly_df_bt[
            monthly_df_bt["date"] >= pd.Timestamp(config.TEST_START)
        ]
        if not oos_monthly.empty:
            log.info("Running FF4 regression on current backtest output ...")
            ff4 = run_factor_attribution(
                oos_monthly.set_index("date")["gross_ls"],
                oos_monthly.set_index("date")["net_ls"],
                ctx["data"]["ff_monthly"],
            )
            log.info("  FF4 alpha (gross): %.4f (t=%.2f)",
                     ff4["gross"]["alpha"], ff4["gross"]["alpha_tstat"])
            log.info("  FF4 alpha (net):   %.4f (t=%.2f)",
                     ff4["net"]["alpha"], ff4["net"]["alpha_tstat"])
            log.info("  FF4 beta_mkt: %.4f (t=%.2f)",
                     ff4["gross"]["beta_mkt"], ff4["gross"]["beta_mkt_tstat"])

    # Equity curve plot
    monthly_df = bt["monthly_df"]
    benchmark = None
    if "vwretd" in panel.columns:
        benchmark = panel.groupby("date")["vwretd"].first().sort_index()
    plot_equity_curve(monthly_df, benchmark)

    # Decile returns bar plot
    decile_df = bt["decile_rets"]
    q_cols = [c for c in decile_df.columns if c.startswith("q")]
    if q_cols:
        decile_means = decile_df[q_cols].mean()
        plot_decile_returns(decile_means.to_frame("mean_return").reset_index())

    log.info("stage_portfolio finished in %.1fs", time.time() - t0)
    ctx["backtest"] = bt
    return ctx


# ── Main ─────────────────────────────────────────────────────────────────────

STAGES = {
    "data": stage_data,
    "features": stage_features,
    "train": stage_train,
    "validate": stage_validate,
    "portfolio": stage_portfolio,
}

STAGE_ORDER = ["data", "features", "train", "validate", "portfolio"]


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Equity Alpha pipeline")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-pull from WRDS (delete parquet cache)",
    )
    args = parser.parse_args()

    _seed_everything()

    # Handle --refresh
    if args.refresh:
        import shutil
        if config.DATA_DIR.exists():
            shutil.rmtree(config.DATA_DIR)
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            log.info("Cleared data cache")

    ctx: dict = {}

    # If resuming from a later stage, try loading cached panel
    panel_path = config.DATA_DIR / "panel.parquet"
    if args.stage not in ("data", "features", "all") and panel_path.exists():
        from features.feature_pipeline import get_feature_cols
        log.info("Loading cached panel from %s ...", panel_path)
        ctx["panel"] = pd.read_parquet(panel_path)
        ctx["feature_cols"] = get_feature_cols()

    if args.stage == "all":
        stages_to_run = STAGE_ORDER
    else:
        stages_to_run = [args.stage]

    t_total = time.time()
    for stage_name in stages_to_run:
        log.info("━" * 60)
        log.info("STAGE: %s", stage_name)
        log.info("━" * 60)
        ctx = STAGES[stage_name](ctx)

    log.info("━" * 60)
    log.info("Pipeline complete in %.1fs", time.time() - t_total)


if __name__ == "__main__":
    main()
