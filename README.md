# autoARIMA

Local, multi-model time-series forecaster. Runs a roster of models
(SARIMAX with exogenous regressors, Random Forest, XGBoost, ElasticNet, Theta,
ETS, a seasonal-naive baseline, and an ensemble), ranks them by an **honest
rolling-origin backtest**, and exports forecasts, metrics, diagnostics and charts.

Built for *difficult* series: short histories, possible non-linearity/non-normality,
late-starting exogenous variables, and — importantly — the case where the **most
recent reported values are unreliable** (e.g. reporting lags). Those recent values
are forecast as the deliverable but are **never used to choose the model**.

---

## Quick start

```bash
pip install -r requirements.txt
```

macOS also needs OpenMP for XGBoost (optional — the pipeline skips XGBoost gracefully if absent):

```bash
brew install libomp
```

**Interactive (one variable at a time):**
```bash
python3 main.py
```

**Batch (multiple dependent variables, unattended):**
```bash
# 1. Edit the CONFIG block at the top of batch_run.py
# 2. Run:
python3 batch_run.py
```

---

## How model selection works (read this)

The system separates two jobs that are usually conflated:

1. **Selection / ranking** — every model is scored by a **rolling-origin backtest
   over the *settled* history** (the periods you trust). Folds train on an expanding
   window and forecast `N_FORECAST` steps ahead; errors are pooled and summarised by
   **MASE** (Mean Absolute Scaled Error). MASE is scale-free, so it is comparable
   across dependent variables, and `MASE ≥ 1` means a model is **no better than the
   seasonal-naive baseline** (surfaced in the output).

2. **Deliverable forecast** — each model refits on all settled data and forecasts the
   most recent `N_FORECAST` periods. Their reported "actuals" are shown for reference
   but flagged **UNRELIABLE** and excluded from ranking.

This is the fix for the common failure mode where a model looks "best" only because it
reproduces under-reported recent values.

---

## Workflow

### Interactive mode (`main.py`)

| Step | Who | What |
|---|---|---|
| 1 | Analyst | Select data file (`.xlsx`, `.xls`, `.csv`) |
| 2 | Analyst | Select time variable |
| 3 | Analyst | Select dependent (target) variable |
| 4 | Analyst | Select N — periods to forecast (the recent, possibly-unreliable tail) |
| 5 | Analyst | Select number of backtest folds for ranking (press Enter for the default, 3) |
| 6 | Auto | Detect seasonal period; select exogenous variables (once, on settled training) |
| 7 | Auto | For each model: deliverable forecast + rolling-origin backtest |
| 8 | Auto | Build the inverse-error ensemble; rank by backtest MASE; flag implausible moves |
| 9 | Auto | Export Excel workbook + PNG chart (best model first throughout) |

The forecast horizon (N) and the fold count are never hardcoded: the horizon is the `N`
you enter here (or `N_FORECAST` in batch mode) and the fold count is this prompt (or
`N_BACKTEST_FOLDS` in batch mode).

### Batch mode (`batch_run.py`)

Edit the CONFIG block, then run unattended.

| Setting | Description |
|---|---|
| `DATA_FILE` | Path to your `.xlsx`, `.xls`, or `.csv` file |
| `SHEET_NAME` | Excel sheet name (or `None` for the first sheet) |
| `TIME_COL` | Column name used as the time index |
| `DEPENDENT_VARS` | List of columns to model — each is modelled in turn; all are excluded from the exogenous pool for every run |
| `N_FORECAST` | Number of most-recent periods to forecast (the unreliable tail) |
| `N_BACKTEST_FOLDS` | Rolling-origin folds used for ranking (default 3) |
| `EXPORT_DIR` | Output directory (`None` → same folder as data file) |
| `EXPORT_PREFIX` | Stem for output filenames |

---

## Models

The roster lives in **one place** — `build_models()` in `forecaster.py`. Add or remove
a single line there to change which models run; the orchestrators, backtester, exports
and charts pick up the change automatically. Every model implements the same
`fit_predict(y_train, X_train, X_future, h)` interface.

| Model | Exog? | Notes |
|---|---|---|
| **SeasonalNaive** | no | Reference baseline and the MASE denominator. |
| **ARIMAX (SARIMAX)** | yes | Auto `(p,d,q)(P,D,Q)` via AICc; seasonal when a period is detected; exogenous regressors via the correct `X=` keyword; exog kept in levels. |
| **RandomForest** | yes | Lag + exog features, recursive multi-step. |
| **XGBoost** | yes | As above (optional dependency). |
| **ElasticNet** | yes | Standardised lags + exog + time index; linear, so it can extrapolate a trend and is robust to many exog. |
| **Theta** | no | `ThetaModel`, deseasonalised when a period is present. |
| **ETS** | no | Holt-Winters; **damped** trend included and AICc-selected so the trend cannot run away. |
| **Ensemble** | — | Inverse-MASE weighted combination of the strongest models. |

---

## Outputs

Charts have **stable filenames (no timestamp)**, so a re-run overwrites the previous
chart of the same name. Excel files keep a timestamp so history is preserved.

### Interactive mode (`main.py`)

- **`Estimates {dep} YYYY-MM-DD HH-MM-SS.xlsx`**

  | Sheet | Contents |
  |---|---|
  | **Forecasts** | Period, reported actual (UNRELIABLE), every model's forecast (★ = best, ⚠ = flagged), best model's 95% CI, and a per-period flag column |
  | **Metrics** | Rank, MASE, sMAPE, RMSE, MAE, MAPE, ARIMA order, AIC, "Beats naive", "Flagged" — sorted by MASE |
  | **Diagnostics** | Seasonal period, selected exogenous, ranking basis, and ARIMAX residual diagnostics (Ljung-Box, Shapiro-Wilk, ARCH) |

- **`Estimates {dep}.png`** — line chart (300 DPI): settled history, reported-but-unreliable tail (dashed grey), every model's forecast (best solid + ★), and the best model's 95% CI.

### Batch mode (`batch_run.py`)

- **`{EXPORT_PREFIX}_{timestamp}.xlsx`**

  | Sheet | Contents |
  |---|---|
  | **Master Forecasts** | All variables; one row per forecast period; `variable`, `reported_actual_UNRELIABLE`, `best_model`, `best_forecast`, `flag`, and a column per model |
  | **Master Metrics** | All models for all variables; leading `variable` column; ranked by MASE within each variable |
  | **FC — {var}** | Per-variable forecast table (top) + diagnostics table (below) |

- **`{EXPORT_PREFIX}_{var}.png`** — one chart per dependent variable (stable name).

---

## Statistical assumptions & robustness

The dependent variable is **not** pre-differenced — SARIMAX selects its own
integration orders `(d, D)`. The checker only applies a **variance-stabilizing**
transform when warranted.

| Concern | Test / handling |
|---|---|
| Variance stabilization (dependent var) | Box-Cox MLE λ → none / log / Box-Cox (inverse is domain-guarded, never `NaN`) |
| Exogenous relevance | **Spearman** rank correlation (catches monotonic non-linearity; invariant to a monotone transform of y) |
| Too many regressors | Cap at ≈ n/10 strongest by \|correlation\| |
| Multicollinearity | VIF > 10 pruned **with an intercept** |
| Late-starting exog | Dropped if coverage < 80% or missing over the forecast horizon; otherwise the shared leading-NaN region is trimmed |
| Non-linearity / non-normality | Tree, ElasticNet and ensemble models capture non-linear effects; ranking is distribution-free (MASE on out-of-sample folds) |

### Residual diagnostics (post-fit, reported in the Excel **Diagnostics** sheet)

| Check | Test | Correctness note |
|---|---|---|
| Autocorrelation | Ljung-Box | uses the `model_df = p+q+P+Q` correction; transient residuals trimmed |
| Normality | Shapiro-Wilk | reliable at small sample sizes |
| Heteroskedasticity | Engle's ARCH | flags non-constant residual variance (CI validity) |

### Forecast guardrails

Models are damped/robust so trends cannot run away, and each forecast step is **flagged**
(⚠) — never silently altered — when it jumps more than 3× the largest historical one-step
move or leaves the historical range by more than half its span.

---

## File structure

```
autoARIMA/
├── main.py          # interactive entry point + analyst UI
├── batch_run.py     # batch runner — CONFIG block, unattended multi-variable run
├── forecaster.py    # model interface (Forecast/BaseForecaster), AssumptionChecker,
│                    # VariableSelector, all model classes, detect_period(), build_models()
├── evaluation.py    # rolling_backtest, metrics (MASE/sMAPE/…), build_ensemble,
│                    # flag_implausible, rank_models, evaluate_variable, result structures
├── exports.py       # ForecastExporter (single-run), BatchForecastExporter (batch)
├── sample_data.xlsx # 52-quarter example dataset (3 dep vars, 4 exog)
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## Notes & limitations

- **All processing is local** — no data leaves your machine.
- Exogenous values for the forecast horizon are read from the dataset, so those values
  must be present in the file (they typically are, even when the target is lagged).
- `auto_arima` searches up to ARIMA(5,2,5)(2,1,2). Adjust `max_*` in `forecaster.py`
  if needed (at the cost of speed). The selected order is searched once per variable and
  refit across backtest folds to keep large batches fast.
- XGBoost requires `pip install xgboost` (+ `brew install libomp` on macOS). If
  unavailable it is skipped and the remaining models still run.

---

## Changelog

### 2026-06-07 — v2.0
- **Critical fix:** ARIMAX now passes exogenous regressors via the correct `X=` keyword
  (pmdarima ≥ 2.0). Previously `exogenous=` was silently dropped, so ARIMAX ran as a
  pure ARIMA — producing flat, mean-reverting forecasts unresponsive to exog changes.
- **Backtest-based selection:** models are ranked by a rolling-origin backtest over the
  settled history (MASE), not by error against the recent, possibly-unreliable actuals.
- **Common model interface** (`fit_predict`) + a single `build_models()` registry — add or
  remove a model in one line. New `evaluation.py` centralizes backtesting, metrics,
  ensembling and guardrails.
- **New models:** seasonal-naive baseline, ElasticNet lag model, and an inverse-error
  **ensemble**. ETS now uses a **damped** trend (AICc-selected); Theta deseasonalizes.
- **Exogenous selection rebuilt:** Spearman correlation, regressor cap (~n/10), VIF with an
  intercept, and explicit late-starting-exog handling (coverage + horizon checks + trim).
- **Assumption checks corrected:** dependent variable no longer pre-differenced (SARIMAX
  picks `d`/`D`); Box-Cox inverse domain-guarded; Ljung-Box uses the `model_df` correction;
  Shapiro-Wilk replaces Jarque-Bera; ARCH heteroskedasticity test added; all residual
  diagnostics are now exported, not just printed.
- **Guardrails:** implausible forecast jumps are flagged (⚠) in console, Excel and charts.
- **Stable chart filenames** (no timestamp) so names stay constant across runs.

### 2026-05-12 — v1.5
- ETS model added (`ETSForecaster`).

### 2026-05-12 — v1.4
- Theta model added; pre-model normality requirement removed; `t_index` added to ML features.

### 2026-04-27 — v1.3
- `batch_run.py` added; per-variable sheets; sample dataset.

### 2026-04-26 — v1.0–v1.1
- Initial three-file architecture, analyst UI, ARIMAX + RF + XGBoost, Excel/PNG export.
