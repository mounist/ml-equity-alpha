"""
Global configuration for the ML Equity Alpha pipeline.
All constants, paths, and hyperparameter defaults live here.
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── WRDS ─────────────────────────────────────────────────────────────────────
WRDS_USERNAME = "mounist"

# ── Date boundaries ──────────────────────────────────────────────────────────
START_DATE = "2000-01-01"
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"
END_DATE = "2024-12-31"

# ── Walk-forward CV ──────────────────────────────────────────────────────────
CV_MIN_TRAIN_YEARS = 5
CV_VAL_WINDOW = 12        # months
CV_REFIT_FREQ = 12        # months (refit each January)
PURGE_GAP = 1             # month gap between train end and val start

# ── Optuna ───────────────────────────────────────────────────────────────────
N_OPTUNA_TRIALS = 15

# ── SHAP ─────────────────────────────────────────────────────────────────────
SHAP_SAMPLE_N = 5000

# ── Portfolio / backtest ─────────────────────────────────────────────────────
TC_ONE_WAY_BPS = 10.0     # one-way transaction cost in basis points
HOLDING_PERIOD_MONTHS = 1

# ── Factor regression ───────────────────────────────────────────────────────
NW_LAGS = 5

# ── Winsorisation ────────────────────────────────────────────────────────────
WINSOR_LIMITS = (0.01, 0.99)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "artifacts" / "data"
RESULTS_DIR = ROOT / "artifacts" / "results"
FIGURES_DIR = ROOT / "artifacts" / "figures"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── LightGBM defaults ───────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=32,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="regression",
    device="gpu",
    n_jobs=-1,
    random_state=SEED,
    verbosity=-1,
)
