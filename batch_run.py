"""
batch_run.py
Batch forecasting runner — models each listed dependent variable independently
from a single data file and consolidates all results into a master output.

Usage
─────
1. Edit the CONFIG block below.
2. python batch_run.py

All listed dependent variables are automatically excluded from the exogenous
candidate pool for every run, so no dependent variable ever appears as a
predictor of another.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from exports import BatchForecastExporter, _ranked_models
from forecaster import ARIMAXForecaster, ETSForecaster, ForecastResult, MLForecaster, MLResult, ThetaForecaster


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these settings before running
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE = 'sample_data.xlsx'
SHEET_NAME = None                       # None → first sheet (ignored for CSV)
TIME_COL = "quarter"                       # column name used as the time index

# Each column listed here is modelled once as the dependent variable.
# All listed columns are excluded from the exogenous pool in every run.
DEPENDENT_VARS: List[str] = ["sales", "revenue", "margin_pct"]

N_FORECAST = 3                          # periods to hold out and forecast

EXPORT_DIR = None                       # None → same directory as DATA_FILE
EXPORT_PREFIX = "batch_forecast"        # file stem; a timestamp is appended automatically

# ─────────────────────────────────────────────────────────────────────────────


def _load_file(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        s  = sheet or xl.sheet_names[0]
        return pd.read_excel(path, sheet_name=s)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx, .xls, or .csv.")


def _run_one(
    y: pd.Series,
    X: pd.DataFrame,
    n_forecast: int,
) -> Tuple[ForecastResult, List[MLResult]]:
    """Run ARIMAX + ML pipeline for a single dependent variable."""
    forecaster = ARIMAXForecaster(verbose=True)
    result     = forecaster.fit_and_forecast(y, X, n_forecast)

    exog_df    = X[forecaster.selected_exog_] if forecaster.selected_exog_ else None
    ml         = MLForecaster(verbose=True)
    ml_results: List[MLResult] = []

    print("\n  Fitting Random Forest...")
    try:
        ml_results.append(ml.fit_and_forecast(y, exog_df, n_forecast, method="rf"))
    except Exception as e:
        print(f"  RandomForest failed: {e}")

    print("\n  Fitting XGBoost...")
    try:
        ml_results.append(ml.fit_and_forecast(y, exog_df, n_forecast, method="xgb"))
    except Exception as e:
        ename = type(e).__name__
        if "Import" in ename or "XGBoost" in ename or "libomp" in str(e):
            print("  XGBoost unavailable — skipping.")
            print("  Fix: pip install xgboost  (macOS: also run brew install libomp)")
        else:
            print(f"  XGBoost failed: {e}")

    print("\n  Fitting Theta...")
    try:
        ml_results.append(ThetaForecaster(verbose=True).fit_and_forecast(y, n_forecast))
    except Exception as e:
        print(f"  Theta failed: {e}")

    print("\n  Fitting ETS...")
    try:
        ml_results.append(ETSForecaster(verbose=True).fit_and_forecast(y, n_forecast))
    except Exception as e:
        print(f"  ETS failed: {e}")

    return result, ml_results


def main(config: Optional[Dict] = None) -> None:
    """
    Run the batch pipeline.

    Parameters
    ──────────
    config : dict, optional
        Override any module-level CONFIG values. Keys match the CONFIG constant
        names (DATA_FILE, TIME_COL, DEPENDENT_VARS, N_FORECAST, EXPORT_DIR,
        EXPORT_PREFIX, SHEET_NAME). Primarily used for testing.
    """
    cfg: Dict = {
        "DATA_FILE":      DATA_FILE,
        "SHEET_NAME":     SHEET_NAME,
        "TIME_COL":       TIME_COL,
        "DEPENDENT_VARS": DEPENDENT_VARS,
        "N_FORECAST":     N_FORECAST,
        "EXPORT_DIR":     EXPORT_DIR,
        "EXPORT_PREFIX":  EXPORT_PREFIX,
    }
    if config:
        cfg.update(config)

    print("=" * 62)
    print("  autoARIMA  —  Batch Forecasting Runner")
    print("=" * 62)

    # ── Validate config ───────────────────────────────────────────────────────
    data_path = Path(cfg["DATA_FILE"])
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found — {data_path.resolve()}")

    dep_vars: List[str] = cfg["DEPENDENT_VARS"]
    if not dep_vars:
        sys.exit("ERROR: DEPENDENT_VARS list is empty — add at least one column name.")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading: {data_path}")
    df = _load_file(str(data_path), cfg["SHEET_NAME"])
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    time_col: str = cfg["TIME_COL"]
    if time_col not in df.columns:
        sys.exit(f"ERROR: TIME_COL '{time_col}' not found. Available: {list(df.columns)}")

    missing = [v for v in dep_vars if v not in df.columns]
    if missing:
        sys.exit(f"ERROR: DEPENDENT_VARS columns not found in file — {missing}")

    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception:
        pass
    df = df.sort_values(time_col).reset_index(drop=True).set_index(time_col)

    # ── Validate n_forecast ───────────────────────────────────────────────────
    n_forecast: int = cfg["N_FORECAST"]
    max_n = max(1, len(df) // 3)
    if n_forecast < 1 or n_forecast > max_n:
        sys.exit(
            f"ERROR: N_FORECAST={n_forecast} out of valid range [1, {max_n}] "
            f"for a dataset with {len(df)} rows."
        )

    # ── Set up exporter ───────────────────────────────────────────────────────
    out_dir = Path(cfg["EXPORT_DIR"]) if cfg["EXPORT_DIR"] else data_path.parent
    exporter = BatchForecastExporter(
        output_dir=str(out_dir),
        prefix=cfg["EXPORT_PREFIX"],
    )

    # ── Iterate over dependent variables ──────────────────────────────────────
    failed: List[str] = []

    for dep_col in dep_vars:
        print("\n" + "=" * 62)
        print(f"  VARIABLE: {dep_col}")
        print("=" * 62)

        y = df[dep_col].astype(float)
        # Exclude ALL listed dep vars so none contaminates another's model
        X = df.drop(columns=dep_vars, errors="ignore").select_dtypes(include="number")

        if X.empty:
            print("  No numeric exogenous columns available — running as pure ARIMA.")

        obs_train = len(df) - n_forecast
        if obs_train < 10:
            print(
                f"  WARNING: only {obs_train} training observations after holding out "
                f"{n_forecast}. Results may be unreliable."
            )

        try:
            result, ml_results = _run_one(y, X, n_forecast)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(dep_col)
            continue

        exporter.add_run(
            dep_col=dep_col,
            time_col=time_col,
            result=result,
            ml_results=ml_results,
        )

        ranked = _ranked_models(result, ml_results)
        print(f"\n  Model Rankings — {dep_col}  (sorted by RMSE, best first)")
        print(f"  {'Rank':<6} {'Model':<22} {'MAE':>9} {'RMSE':>9} {'MAPE':>9}")
        print("  " + "-" * 60)
        for m in ranked:
            star = "★" if m["rank"] == 1 else " "
            print(
                f"  {star} #{m['rank']:<4} {m['label']:<22}"
                f" {m['mae']:>9.4f} {m['rmse']:>9.4f} {m['mape']:>8.2f}%"
            )

    # ── Export ────────────────────────────────────────────────────────────────
    if exporter.has_results:
        print("\n" + "=" * 62)
        print("  Exporting results")
        print("=" * 62)
        exporter.export_excel()
        exporter.export_charts()
    else:
        print("\nNo successful runs — nothing to export.")

    if failed:
        print(f"\n  Variables that failed and were skipped: {failed}")

    print("\nBatch run complete.")


if __name__ == "__main__":
    main()
