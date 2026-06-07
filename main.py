"""
main.py
Entry point — analyst-driven selections, then runs the full pipeline for one
dependent variable.

Usage:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from evaluation import evaluate_variable
from exports import ForecastExporter
from forecaster import detect_period


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner() -> None:
    print("=" * 62)
    print("  autoARIMA  —  Multi-Model Time-Series Forecaster")
    print("=" * 62)


def _pick_file() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("Excel / CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")],
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
        raw = input(f"\nNumber of periods to forecast (1–{max_n}): ").strip()
        try:
            n = int(raw)
            if 1 <= n <= max_n:
                return n
        except ValueError:
            pass
        print(f"  Enter an integer between 1 and {max_n}.")


def _pick_folds(max_folds: int, default: int) -> int:
    """Number of rolling-origin backtest folds used to rank models. Press Enter
    for the default. Folds control ranking robustness, not the forecast length."""
    while True:
        raw = input(
            f"\nNumber of backtest folds for ranking (1–{max_folds}) [Enter for {default}]: "
        ).strip()
        if raw == "":
            return default
        try:
            n = int(raw)
            if 1 <= n <= max_folds:
                return n
        except ValueError:
            pass
        print(f"  Enter an integer between 1 and {max_folds} (or press Enter for {default}).")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner()

    print("\nStep 1: Select data file")
    filepath = _pick_file()
    if not filepath:
        sys.exit("No file selected — exiting.")

    print(f"  Loading: {filepath}")
    df = _load_file(filepath)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns\n")
    if df.empty:
        sys.exit("File is empty — exiting.")

    time_col = _pick_column(df, "Step 2: Select the TIME variable")
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception:
        pass
    df = df.sort_values(time_col).reset_index(drop=True).set_index(time_col)

    dep_col = _pick_column(df, "Step 3: Select the DEPENDENT (target) variable")

    max_n = max(1, len(df) // 3)
    n_forecast = _pick_n(max_n)
    if len(df) - n_forecast < 10:
        print(f"\n  WARNING: only {len(df) - n_forecast} training observations after holding out "
              f"{n_forecast}. Results may be unreliable.")

    period = detect_period(df.index)
    print(f"\n  Seasonal period detected: {period}" + ("" if period > 1 else " (non-seasonal)"))

    # How many rolling-origin backtest folds can the settled history support?
    n_settled = len(df) - n_forecast
    min_train = max(10, 2 * period, n_forecast + 2)
    max_folds = max(1, n_settled - n_forecast - min_train + 1)
    if max_folds <= 1:
        n_folds = 1
        print("  Backtest folds: 1 (limited by available history)")
    else:
        n_folds = _pick_folds(max_folds, default=min(3, max_folds))

    y = df[dep_col].astype(float)
    X = df.drop(columns=[dep_col]).select_dtypes(include="number")
    if X.empty:
        print("  No numeric exogenous columns found — univariate models only.")

    print("\n" + "=" * 62)
    print("  Running models (ranked by backtest MASE on settled history)")
    print("=" * 62)

    vr = evaluate_variable(
        y=y, X=X, dep_col=dep_col, time_col=time_col,
        n_forecast=n_forecast, period=period, n_folds=n_folds, verbose=True,
    )

    print("\n" + "=" * 62)
    print("  Exporting results")
    print("=" * 62)
    exporter = ForecastExporter(output_dir=str(Path(filepath).parent))
    exporter.export_excel(vr)
    exporter.export_chart(vr)

    print("\n" + "=" * 62)
    print("  Model Rankings  (sorted by backtest MASE, best first)")
    print("=" * 62)
    print(f"  {'Rank':<5} {'Model':<16} {'MASE':>7} {'sMAPE':>8} {'RMSE':>10}  Flags")
    print("  " + "-" * 60)
    for m in vr.models:
        star = "★" if m.rank == 1 else " "
        flag = "⚠" if any(f for f in m.flags) else ""
        print(f"  {star}#{m.rank:<3} {m.name:<16} {m.metrics['mase']:>7.3f} "
              f"{m.metrics['smape']:>7.2f}% {m.metrics['rmse']:>10.3f}  {flag}")

    best = vr.models[0]
    print(f"\n  Best fit: ★ {best.name}  (MASE={best.metrics['mase']:.3f})")
    print(f"  Dependent variable : {dep_col}")
    print(f"  Selected exogenous : {vr.selected_exog or 'none'}")
    print(f"  Forecast periods   : {n_forecast}  (reported actuals shown but treated as UNRELIABLE)")
    if any(f for f in best.flags):
        print("  ⚠ Some forecast steps were flagged as large/implausible — see the chart & Excel flags.")
    print("\nDone.")


if __name__ == "__main__":
    main()
