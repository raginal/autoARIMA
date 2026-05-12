"""
main.py
Entry point — analyst-driven selections, then orchestrates the full pipeline.

Usage:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from exports import ForecastExporter
from forecaster import ARIMAXForecaster, ETSForecaster, MLForecaster, ThetaForecaster, rank_models


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner() -> None:
    print("=" * 62)
    print("  autoARIMA  —  ARIMAX Time-Series Forecaster")
    print("=" * 62)


def _pick_file() -> str:
    """Open a native file-picker dialog; fall back to typed path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[
                ("Excel / CSV", "*.xlsx *.xls *.csv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path
    except Exception:
        return input("Enter path to data file: ").strip().strip('"')


def _load_file(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        if len(xl.sheet_names) > 1:
            print("\nSheets available:")
            for i, s in enumerate(xl.sheet_names, 1):
                print(f"  {i:2}. {s}")
            sheet = _pick_from_list(xl.sheet_names, "Select sheet: ")
        else:
            sheet = xl.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx, .xls, or .csv.")


def _pick_from_list(options: List[str], prompt: str = "Select: ") -> str:
    while True:
        try:
            n = int(input(prompt)) - 1
            if 0 <= n < len(options):
                return options[n]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid — enter a number from the list above.")


def _pick_column(df: pd.DataFrame, prompt: str, exclude: Optional[List[str]] = None) -> str:
    cols = [c for c in df.columns if c not in (exclude or [])]
    print(f"\n{prompt}")
    for i, c in enumerate(cols, 1):
        dtype = str(df[c].dtype)
        sample = df[c].dropna().iloc[0] if not df[c].dropna().empty else "—"
        print(f"  {i:2}. {c:<30} [{dtype}]  e.g. {sample}")
    return _pick_from_list(cols, "Enter number: ")


def _pick_n(max_n: int) -> int:
    while True:
        raw = input(f"\nNumber of periods to forecast/predict (1–{max_n}): ").strip()
        try:
            n = int(raw)
            if 1 <= n <= max_n:
                return n
        except ValueError:
            pass
        print(f"  Enter an integer between 1 and {max_n}.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner()

    # ── Step 1: load data ─────────────────────────────────────────────────────
    print("\nStep 1: Select data file")
    filepath = _pick_file()
    if not filepath:
        sys.exit("No file selected — exiting.")

    print(f"  Loading: {filepath}")
    df = _load_file(filepath)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns\n")
    if df.empty:
        sys.exit("File is empty — exiting.")

    # ── Step 2: time variable ─────────────────────────────────────────────────
    time_col = _pick_column(df, "Step 2: Select the TIME variable")
    # Try to parse as datetime; keep as-is if it fails (e.g., integer years)
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception:
        pass
    df = df.sort_values(time_col).reset_index(drop=True)
    df = df.set_index(time_col)

    # ── Step 3: dependent variable ────────────────────────────────────────────
    dep_col = _pick_column(df, "Step 3: Select the DEPENDENT (target) variable")

    # ── Step 4: number of periods to forecast ─────────────────────────────────
    max_n = max(1, len(df) // 3)  # cap at 1/3 of data to ensure enough training
    n_forecast = _pick_n(max_n)

    if len(df) - n_forecast < 10:
        print(
            f"\n  WARNING: only {len(df) - n_forecast} training observations after"
            f" holding out {n_forecast}. Results may be unreliable."
        )

    # ── ARIMAX pipeline ───────────────────────────────────────────────────────
    y = df[dep_col].astype(float)
    X = df.drop(columns=[dep_col]).select_dtypes(include="number")

    if X.empty:
        print("\n  No numeric exogenous columns found — running as pure ARIMA.")

    print("\n" + "=" * 62)
    print("  Running ARIMAX pipeline")
    print("=" * 62)

    forecaster = ARIMAXForecaster(verbose=True)
    result     = forecaster.fit_and_forecast(y, X, n_forecast)

    # ── ML comparison (always runs; XGBoost skipped gracefully if not installed) ──
    exog_df = X[forecaster.selected_exog_] if forecaster.selected_exog_ else None
    ml      = MLForecaster(verbose=True)
    ml_results = []

    print("\n" + "=" * 62)
    print("  Running ML models")
    print("=" * 62)

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

    # ── Rank all models by RMSE ───────────────────────────────────────────────
    ranked = rank_models(result, ml_results)

    # ── Export ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Exporting results")
    print("=" * 62)

    output_dir = Path(filepath).parent
    exporter   = ForecastExporter(output_dir=str(output_dir))
    exporter.export_excel(result, ml_results or None, dep_col, time_col)
    exporter.export_chart(result, ml_results or None, dep_col, time_col)

    # ── Ranked summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Model Rankings  (sorted by RMSE, best first)")
    print("=" * 62)
    print(f"  {'Rank':<6} {'Model':<22} {'MAE':>9} {'RMSE':>9} {'MAPE':>9}")
    print("  " + "-" * 60)
    for m in ranked:
        star = "★" if m["rank"] == 1 else " "
        print(
            f"  {star} #{m['rank']:<4} {m['label']:<22}"
            f" {m['mae']:>9.4f} {m['rmse']:>9.4f} {m['mape']:>8.2f}%"
        )
    print(f"\n  Best fit: ★ {ranked[0]['label']}")
    print(f"  Dependent variable : {dep_col}")
    print(f"  Transform applied  : {result.dep_transform.method}")
    print(f"  ARIMA order        : {result.order}")
    print(f"  Exogenous used     : {result.selected_exog or 'none'}")
    print(f"  Forecast periods   : {n_forecast}")
    print("\nDone.")


if __name__ == "__main__":
    main()
