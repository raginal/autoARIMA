"""
batch_run.py
Batch forecasting runner — models each listed dependent variable independently
from a single data file and consolidates all results into a master output.

Usage
─────
1. Edit the CONFIG block below.
2. python batch_run.py

All listed dependent variables are automatically excluded from the exogenous
candidate pool for every run, so no dependent variable predicts another.

Each variable is ranked by a rolling-origin **backtest over the settled history**
(MASE). The forecast for the most recent N_FORECAST periods — whose reported
actuals are unreliable due to reporting lags — is the deliverable; those reported
actuals are shown for reference only and never drive model selection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from evaluation import evaluate_variable
from exports import BatchForecastExporter
from forecaster import detect_period


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these settings before running
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE = "sample_data.xlsx"
SHEET_NAME = None                          # None → first sheet (ignored for CSV)
TIME_COL = "quarter"                       # column name used as the time index

# Each column listed here is modelled once as the dependent variable.
# All listed columns are excluded from the exogenous pool in every run.
DEPENDENT_VARS: List[str] = ["sales", "revenue", "margin_pct"]

N_FORECAST = 3                             # most-recent periods to forecast (the unreliable tail)
N_BACKTEST_FOLDS = 3                       # rolling-origin folds used for ranking

EXPORT_DIR = None                          # None → same directory as DATA_FILE
EXPORT_PREFIX = "batch_forecast"           # file stem; a timestamp is appended to the Excel file

# ─────────────────────────────────────────────────────────────────────────────


def _load_file(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        return pd.read_excel(path, sheet_name=(sheet or xl.sheet_names[0]))
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx, .xls, or .csv.")


def main(config: Optional[Dict] = None) -> None:
    cfg: Dict = {
        "DATA_FILE": DATA_FILE, "SHEET_NAME": SHEET_NAME, "TIME_COL": TIME_COL,
        "DEPENDENT_VARS": DEPENDENT_VARS, "N_FORECAST": N_FORECAST,
        "N_BACKTEST_FOLDS": N_BACKTEST_FOLDS,
        "EXPORT_DIR": EXPORT_DIR, "EXPORT_PREFIX": EXPORT_PREFIX,
    }
    if config:
        cfg.update(config)

    print("=" * 62)
    print("  autoARIMA  —  Batch Forecasting Runner")
    print("=" * 62)

    data_path = Path(cfg["DATA_FILE"])
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found — {data_path.resolve()}")

    dep_vars: List[str] = cfg["DEPENDENT_VARS"]
    if not dep_vars:
        sys.exit("ERROR: DEPENDENT_VARS list is empty — add at least one column name.")

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

    period = detect_period(df.index)
    print(f"Seasonal period detected: {period}" + ("" if period > 1 else " (non-seasonal)"))

    n_forecast: int = cfg["N_FORECAST"]
    max_n = max(1, len(df) // 3)
    if n_forecast < 1 or n_forecast > max_n:
        sys.exit(f"ERROR: N_FORECAST={n_forecast} out of valid range [1, {max_n}] for {len(df)} rows.")

    out_dir = Path(cfg["EXPORT_DIR"]) if cfg["EXPORT_DIR"] else data_path.parent
    exporter = BatchForecastExporter(output_dir=str(out_dir), prefix=cfg["EXPORT_PREFIX"])

    failed: List[str] = []
    for dep_col in dep_vars:
        print("\n" + "=" * 62)
        print(f"  VARIABLE: {dep_col}")
        print("=" * 62)

        y = df[dep_col].astype(float)
        X = df.drop(columns=dep_vars, errors="ignore").select_dtypes(include="number")
        if X.empty:
            print("  No numeric exogenous columns available — running univariate models only.")
        if len(df) - n_forecast < 10:
            print(f"  WARNING: only {len(df) - n_forecast} training observations — results may be unreliable.")

        try:
            vr = evaluate_variable(
                y=y, X=X, dep_col=dep_col, time_col=time_col,
                n_forecast=n_forecast, period=period,
                n_folds=cfg["N_BACKTEST_FOLDS"], verbose=True,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(dep_col)
            continue

        exporter.add_run(vr)
        _print_rankings(vr)

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


def _print_rankings(vr) -> None:
    print(f"\n  Model Rankings — {vr.dep_col}  (sorted by backtest MASE, best first)")
    print(f"  {'Rank':<5} {'Model':<16} {'MASE':>7} {'sMAPE':>8} {'RMSE':>10}  Flags")
    print("  " + "-" * 60)
    for m in vr.models:
        star = "★" if m.rank == 1 else " "
        flag = "⚠" if any(f for f in m.flags) else ""
        print(f"  {star}#{m.rank:<3} {m.name:<16} {m.metrics['mase']:>7.3f} "
              f"{m.metrics['smape']:>7.2f}% {m.metrics['rmse']:>10.3f}  {flag}")


if __name__ == "__main__":
    main()
