"""
exports.py
Handles all output: Excel workbooks and PNG line charts.

Public classes
──────────────
ForecastExporter      — single dependent-variable run (used by main.py)
BatchForecastExporter — multi-variable batch run (used by batch_run.py)

Module-level helpers
────────────────────
_ranked_models()      — build a ranked list (by RMSE) from ARIMAX + ML results
_draw_forecast_chart()— populate a matplotlib Axes with the standard forecast chart
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive; safe for scripts without a display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecaster import ForecastResult, MLResult


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ranked_models(
    result: ForecastResult,
    ml_results: Optional[List[MLResult]],
) -> List[Dict]:
    """
    Build a flat list of all models sorted by RMSE ascending.

    Each dict: label, forecast, ci_lower, ci_upper, mae, rmse, mape,
               is_arimax, aic, order, rank.
    """
    rows: List[Dict] = [
        {
            "label":     f"ARIMAX{result.order}",
            "forecast":  result.forecast,
            "ci_lower":  result.ci_lower,
            "ci_upper":  result.ci_upper,
            "mae":       result.mae,
            "rmse":      result.rmse,
            "mape":      result.mape,
            "is_arimax": True,
            "aic":       result.aic,
            "order":     str(result.order),
        }
    ]
    for r in (ml_results or []):
        rows.append(
            {
                "label":     r.method,
                "forecast":  r.forecast,
                "ci_lower":  None,
                "ci_upper":  None,
                "mae":       r.mae,
                "rmse":      r.rmse,
                "mape":      r.mape,
                "is_arimax": False,
                "aic":       "—",
                "order":     "—",
            }
        )
    rows.sort(key=lambda x: x["rmse"])
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def _draw_forecast_chart(
    ax,
    result: ForecastResult,
    ml_results: Optional[List[MLResult]],
    dep_name: str,
    time_name: str,
) -> None:
    """Populate a matplotlib Axes with the standard ranked forecast chart."""
    ranked = _ranked_models(result, ml_results)
    use_dates = pd.api.types.is_datetime64_any_dtype(result.forecast.index)

    train = result.train_actuals
    ax.plot(
        train.index, train.values,
        color="#2c7bb6", linewidth=1.5, label="Training data",
    )

    act = result.actuals
    ax.plot(
        act.index, act.values,
        "o-", color="#1a9641", linewidth=1.5, markersize=5,
        label=f"Actual (held-out, n={result.n_forecast})",
    )

    palette = ["#d7191c", "#f4a261", "#7b2d8b", "#00bcd4"]
    for m, color in zip(ranked, palette):
        is_best   = m["rank"] == 1
        star      = "★ " if is_best else ""
        linestyle = "-"  if is_best else "--"
        marker    = "s"  if is_best else "^"
        lw        = 2.0  if is_best else 1.4
        label     = (
            f"{star}{m['label']}  "
            f"(MAE={m['mae']:.3f} | RMSE={m['rmse']:.3f} | MAPE={m['mape']:.1f}%)"
        )
        ax.plot(
            m["forecast"].index, m["forecast"].values,
            marker + linestyle, color=color,
            linewidth=lw, markersize=5 if is_best else 4,
            label=label, zorder=4 if is_best else 3,
        )

    ax.fill_between(
        result.forecast.index,
        result.ci_lower.values,
        result.ci_upper.values,
        alpha=0.12, color="#d7191c", label="ARIMAX 95% CI",
    )

    if len(train) > 0:
        ax.axvline(train.index[-1], color="gray", linestyle=":", linewidth=1, alpha=0.6)

    if use_dates:
        ax.xaxis.set_major_formatter(
            mdates.AutoDateFormatter(mdates.AutoDateLocator())
        )

    best = ranked[0]
    ax.set_title(
        f"Forecast — {dep_name}  |  Best fit: ★ {best['label']}"
        f"  (RMSE={best['rmse']:.4f})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel(time_name, fontsize=11)
    ax.set_ylabel(dep_name,  fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25)


def _autofit_columns(ws) -> None:
    for col_cells in ws.columns:
        max_len = max(
            (len(str(c.value)) for c in col_cells if c.value is not None),
            default=10,
        )
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)


def _transform_detail(t) -> str:
    """Human-readable detail string for a TransformInfo."""
    if t.method == "boxcox":
        return f"λ={t.lambda_:.4f}, shift={t.shift:.4f}"
    if t.method in ("log", "sqrt", "pct_change") and t.shift != 0:
        return f"shift={t.shift:.4f}"
    if t.method == "diff":
        return "—"
    return "—"


def _build_variables_df(result: ForecastResult, dep_name: str) -> pd.DataFrame:
    """
    Build the variable-metadata table for a single run:
    one row for the dependent variable, one row per selected exogenous variable.
    """
    t = result.dep_transform
    rows = [
        {
            "Variable":  dep_name,
            "Role":      "Dependent",
            "Transform": t.method,
            "Detail":    _transform_detail(t),
        }
    ]
    for v in result.selected_exog:
        xt = result.exog_transforms.get(v)
        rows.append(
            {
                "Variable":  v,
                "Role":      "Exogenous (selected)",
                "Transform": xt.method if xt else "—",
                "Detail":    _transform_detail(xt) if xt else "—",
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Single-run exporter  (used by main.py)
# ─────────────────────────────────────────────────────────────────────────────

class ForecastExporter:
    """
    Writes results to:
      • "Estimates YYYY-MM-DD HH-MM-SS.xlsx"  — Forecasts / Metrics / Variables sheets
      • "Estimates YYYY-MM-DD HH-MM-SS.png"   — line chart at DPI=300

    All models are sorted by RMSE (best first) throughout every output.
    The best-fit model is highlighted with a "★" prefix.
    Both files are saved to output_dir (defaults to the source data directory).
    """

    def __init__(self, output_dir: str = ".") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp     = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.stem = f"Estimates {stamp}"

    @property
    def excel_path(self) -> Path:
        return self.output_dir / f"{self.stem}.xlsx"

    @property
    def chart_path(self) -> Path:
        return self.output_dir / f"{self.stem}.png"

    def export_excel(
        self,
        result: ForecastResult,
        ml_results: Optional[List[MLResult]] = None,
        dep_name: str = "y",
        time_name: str = "time",
    ) -> Path:
        ranked = _ranked_models(result, ml_results)

        with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:

            # Sheet 1: Forecasts
            fc_data: Dict = {
                time_name:              ranked[0]["forecast"].index,
                f"{dep_name} (Actual)": result.actuals.values,
            }
            for m in ranked:
                star = "★ " if m["rank"] == 1 else ""
                fc_data[f"{star}{m['label']} Forecast"] = m["forecast"].values
            fc_data["ARIMAX CI Lower 95%"] = result.ci_lower.values
            fc_data["ARIMAX CI Upper 95%"] = result.ci_upper.values
            pd.DataFrame(fc_data).to_excel(writer, sheet_name="Forecasts", index=False)
            _autofit_columns(writer.sheets["Forecasts"])

            # Sheet 2: Metrics
            metric_rows = []
            for m in ranked:
                star = "★ " if m["rank"] == 1 else ""
                metric_rows.append(
                    {
                        "Rank":     m["rank"],
                        "Model":    f"{star}{m['label']}",
                        "Order":    m["order"],
                        "AIC":      round(m["aic"], 4) if isinstance(m["aic"], float) else m["aic"],
                        "MAE":      round(m["mae"],  4),
                        "RMSE":     round(m["rmse"], 4),
                        "MAPE (%)": round(m["mape"], 2),
                    }
                )
            pd.DataFrame(metric_rows).to_excel(writer, sheet_name="Metrics", index=False)
            _autofit_columns(writer.sheets["Metrics"])

            # Sheet 3: Variables
            _build_variables_df(result, dep_name).to_excel(
                writer, sheet_name="Variables", index=False
            )
            _autofit_columns(writer.sheets["Variables"])

        print(f"  Excel saved  →  {self.excel_path}")
        return self.excel_path

    def export_chart(
        self,
        result: ForecastResult,
        ml_results: Optional[List[MLResult]] = None,
        dep_name: str = "y",
        time_name: str = "time",
    ) -> Path:
        use_dates = pd.api.types.is_datetime64_any_dtype(result.forecast.index)
        fig, ax = plt.subplots(figsize=(14, 6))
        _draw_forecast_chart(ax, result, ml_results, dep_name, time_name)
        if use_dates:
            fig.autofmt_xdate()
        plt.tight_layout()
        fig.savefig(self.chart_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved  →  {self.chart_path}")
        return self.chart_path


# ─────────────────────────────────────────────────────────────────────────────
# Batch exporter  (used by batch_run.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _BatchRun:
    dep_col:    str
    time_col:   str
    result:     ForecastResult
    ml_results: List[MLResult]


class BatchForecastExporter:
    """
    Accumulates results across multiple dependent-variable runs, then writes:

    Excel  →  {prefix}_{timestamp}.xlsx
        • "Master Forecasts" — all variables, one row per forecast period,
          with a leading "variable" column
        • "Master Metrics"   — all models for all variables, ranked by RMSE,
          with a leading "variable" column
        • "FC — {var}"       — per-variable Forecasts sheet (same layout as
          the single-run exporter, max 31 chars per sheet name)

    Charts →  {prefix}_{var}_{timestamp}.png   (one per dependent variable)
    """

    _ML_LABELS = ("RandomForest", "XGBoost")   # canonical ML model name prefixes

    def __init__(self, output_dir: str = ".", prefix: str = "batch_forecast") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp      = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.stem  = f"{prefix}_{stamp}"
        self._runs: List[_BatchRun] = []

    @property
    def has_results(self) -> bool:
        return len(self._runs) > 0

    @property
    def excel_path(self) -> Path:
        return self.output_dir / f"{self.stem}.xlsx"

    def chart_path(self, dep_col: str) -> Path:
        safe = dep_col.replace("/", "-").replace("\\", "-")
        return self.output_dir / f"{self.stem}_{safe}.png"

    def add_run(
        self,
        dep_col: str,
        time_col: str,
        result: ForecastResult,
        ml_results: List[MLResult],
    ) -> None:
        self._runs.append(
            _BatchRun(
                dep_col=dep_col,
                time_col=time_col,
                result=result,
                ml_results=ml_results,
            )
        )

    # ── Excel ─────────────────────────────────────────────────────────────────

    def export_excel(self) -> Path:
        master_fc_rows:  List[Dict] = []
        master_met_rows: List[Dict] = []

        with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:

            for run in self._runs:
                result     = run.result
                ml_results = run.ml_results
                ranked     = _ranked_models(result, ml_results)
                dep_col    = run.dep_col
                time_col   = run.time_col

                # ── Build ML forecast lookup {label_prefix: Series} ───────────
                ml_fc: Dict[str, Optional[np.ndarray]] = {
                    lbl: None for lbl in self._ML_LABELS
                }
                for r in ml_results:
                    for lbl in self._ML_LABELS:
                        if r.method.startswith(lbl):
                            ml_fc[lbl] = r.forecast.values

                best_label = ranked[0]["label"]

                # ── Per-period rows for master forecast table ─────────────────
                for i, idx_val in enumerate(result.forecast.index):
                    row: Dict = {
                        "variable":         dep_col,
                        time_col:           idx_val,
                        "actual":           float(result.actuals.iloc[i]),
                        "best_model":       best_label,
                        "ARIMAX_forecast":  float(result.forecast.iloc[i]),
                        "ARIMAX_CI_lower":  float(result.ci_lower.iloc[i]),
                        "ARIMAX_CI_upper":  float(result.ci_upper.iloc[i]),
                    }
                    for lbl in self._ML_LABELS:
                        col_name = f"{lbl}_forecast"
                        row[col_name] = (
                            float(ml_fc[lbl][i]) if ml_fc[lbl] is not None else np.nan
                        )
                    master_fc_rows.append(row)

                # ── Metric rows for master metrics table ──────────────────────
                for m in ranked:
                    star = "★ " if m["rank"] == 1 else ""
                    master_met_rows.append(
                        {
                            "variable":  dep_col,
                            "rank":      m["rank"],
                            "model":     f"{star}{m['label']}",
                            "order":     m["order"],
                            "AIC":       round(m["aic"], 4) if isinstance(m["aic"], float) else m["aic"],
                            "MAE":       round(m["mae"],  4),
                            "RMSE":      round(m["rmse"], 4),
                            "MAPE (%)":  round(m["mape"], 2),
                        }
                    )

                # ── Per-variable Forecasts sheet ──────────────────────────────
                fc_data: Dict = {
                    time_col:              result.forecast.index,
                    f"{dep_col} (Actual)": result.actuals.values,
                }
                for m in ranked:
                    star = "★ " if m["rank"] == 1 else ""
                    fc_data[f"{star}{m['label']} Forecast"] = m["forecast"].values
                fc_data["ARIMAX CI Lower 95%"] = result.ci_lower.values
                fc_data["ARIMAX CI Upper 95%"] = result.ci_upper.values

                df_fc = pd.DataFrame(fc_data)
                df_vars = _build_variables_df(result, dep_col)

                # Excel sheet names are capped at 31 characters
                sheet_name = f"FC — {dep_col}"[:31]
                # Write forecast table, then variables table two rows below
                df_fc.to_excel(writer, sheet_name=sheet_name, index=False)
                start_row = len(df_fc) + 3  # header + data rows + 2-row gap
                df_vars.to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=start_row
                )
                _autofit_columns(writer.sheets[sheet_name])

            # ── Master Forecasts sheet (written first via sheet ordering) ─────
            if master_fc_rows:
                df_master = pd.DataFrame(master_fc_rows)
                df_master.to_excel(
                    writer, sheet_name="Master Forecasts", index=False
                )
                _autofit_columns(writer.sheets["Master Forecasts"])

            # ── Master Metrics sheet ──────────────────────────────────────────
            if master_met_rows:
                df_met = pd.DataFrame(master_met_rows)
                df_met.to_excel(
                    writer, sheet_name="Master Metrics", index=False
                )
                _autofit_columns(writer.sheets["Master Metrics"])

        # openpyxl writes sheets in insertion order; reorder so master sheets
        # appear first (indices 0 and 1).
        try:
            import openpyxl
            wb = openpyxl.load_workbook(self.excel_path)
            names = wb.sheetnames
            desired_first = [n for n in ("Master Forecasts", "Master Metrics") if n in names]
            rest = [n for n in names if n not in desired_first]
            wb._sheets = [wb[n] for n in desired_first + rest]
            wb.save(self.excel_path)
        except Exception:
            pass  # sheet ordering is cosmetic; don't fail the export

        print(f"  Excel saved  →  {self.excel_path}")
        return self.excel_path

    # ── Charts ────────────────────────────────────────────────────────────────

    def export_charts(self) -> List[Path]:
        paths: List[Path] = []
        for run in self._runs:
            use_dates = pd.api.types.is_datetime64_any_dtype(
                run.result.forecast.index
            )
            fig, ax = plt.subplots(figsize=(14, 6))
            _draw_forecast_chart(
                ax, run.result, run.ml_results, run.dep_col, run.time_col
            )
            if use_dates:
                fig.autofmt_xdate()
            plt.tight_layout()
            p = self.chart_path(run.dep_col)
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Chart saved  →  {p}")
            paths.append(p)
        return paths
