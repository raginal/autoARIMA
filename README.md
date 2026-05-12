# autoARIMA

Local ARIMAX time-series forecaster with automatic variable selection,
assumption checking, and optional ML comparison.

---

## Quick start

```bash
pip install -r requirements.txt
pip install xgboost          # required for XGBoost comparison
```

macOS also needs OpenMP for XGBoost:

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

Tested on Python 3.13.5.

---

## Workflow

### Interactive mode  (`main.py`)

The analyst is prompted for four choices; everything else is automatic.

| Step | Who | What |
|---|---|---|
| 1 | Analyst | Select data file (`.xlsx`, `.xls`, `.csv`) |
| 2 | Analyst | Select time variable |
| 3 | Analyst | Select dependent (target) variable |
| 4 | Analyst | Select N — number of periods to forecast |
| 5 | Auto | Check + transform dependent variable |
| 6 | Auto | Check + transform each exogenous candidate |
| 7 | Auto | Select best exogenous via correlation & VIF |
| 8 | Auto | Fit ARIMAX (auto p, d, q via AIC) on data[:-N] |
| 9 | Auto | Forecast last N periods; compare to actuals |
| 10 | Auto | Check model residuals |
| 11 | Auto | Fit Random Forest, XGBoost, Theta, and ETS; rank all five models by RMSE |
| 12 | Auto | Export Excel workbook + PNG chart (best model first throughout) |

### Batch mode  (`batch_run.py`)

Edit the CONFIG block at the top of `batch_run.py`, then run it unattended.

| Setting | Description |
|---|---|
| `DATA_FILE` | Path to your `.xlsx`, `.xls`, or `.csv` file |
| `SHEET_NAME` | Excel sheet name (or `None` for the first sheet) |
| `TIME_COL` | Column name used as the time index |
| `DEPENDENT_VARS` | List of columns to model — each is the dependent variable for one run; all are excluded from the exogenous candidate pool for every run |
| `N_FORECAST` | Periods to hold out and forecast (shared across all variables) |
| `EXPORT_DIR` | Output directory (`None` → same folder as data file) |
| `EXPORT_PREFIX` | Stem for output filenames (a timestamp is always appended) |

Each variable in `DEPENDENT_VARS` is modelled in turn with the full ARIMAX + Random Forest + XGBoost + Theta + ETS pipeline. Results are consolidated into a single Excel workbook and one PNG chart per variable.

---

## Outputs

### Interactive mode (`main.py`)

Both files are saved to the same directory as the source data file.

#### `Estimates YYYY-MM-DD HH-MM-SS.xlsx`

| Sheet | Contents |
|---|---|
| **Forecasts** | Period, actual, then all model forecasts sorted best-first (★ = best fit) |
| **Metrics** | Rank, MAE, RMSE, MAPE, AIC — sorted by RMSE ascending |
| **Variables** | Dependent + exogenous variables with transforms applied |

#### `Estimates YYYY-MM-DD HH-MM-SS.png`

Line chart (300 DPI) showing training series, held-out actuals, all model forecasts (best solid + ★, others dashed), and ARIMAX 95% CI.

---

### Batch mode (`batch_run.py`)

#### `{EXPORT_PREFIX}_{timestamp}.xlsx`

| Sheet | Contents |
|---|---|
| **Master Forecasts** | All variables combined; one row per forecast period; leading `variable` column identifies the dependent variable; columns: `actual`, `best_model`, `ARIMAX_forecast`, `ARIMAX_CI_lower`, `ARIMAX_CI_upper`, `RandomForest_forecast`, `XGBoost_forecast`, `Theta_forecast`, `ETS_forecast` |
| **Master Metrics** | All models for all variables; leading `variable` column; ranked by RMSE within each variable |
| **FC — {var}** | Per-variable sheet: forecast table (top) + variables table (below, separated by two blank rows) listing the dependent variable and each selected exogenous variable with their transform method and detail |

#### `{EXPORT_PREFIX}_{var}_{timestamp}.png`

One chart per dependent variable — same layout as interactive mode.

---

## Statistical assumptions enforced

### Dependent & exogenous variables

| Assumption | Test | Remediation |
|---|---|---|
| Stationarity | ADF + KPSS | log → Box-Cox → sqrt → first-difference → pct-change; discard if unsolvable |
| Linearity of y ~ x | Harvey-Collier (on first-differenced series) | warning only; ML models capture non-linear effects |
| No multicollinearity | VIF < 10 | iteratively remove highest-VIF variable |
| Correlation with target | \|r\| ≥ 0.10 | discard exogenous candidate |

### Model residuals (warnings, not hard stops)

| Check | Test |
|---|---|
| Residual normality | Jarque-Bera |
| No residual autocorrelation | Ljung-Box |

---

## File structure

```
autoARIMA/
├── main.py          # interactive entry point + analyst UI
├── batch_run.py     # batch runner — CONFIG block, unattended multi-variable run
├── forecaster.py    # AssumptionChecker, VariableSelector,
│                    # ARIMAXForecaster, MLForecaster, ThetaForecaster, ETSForecaster
├── exports.py       # ForecastExporter (single-run), BatchForecastExporter (batch)
├── sample_data.xlsx # 52-quarter example dataset (3 dep vars, 4 exog)
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## Notes & limitations

- **All processing is local** — no data leaves your machine.
- Exogenous variables for the forecast period are taken from the dataset
  (the held-out actuals). This is consistent with nowcasting / backtesting
  but means you need those values in the file.
- `auto_arima` tests up to ARIMA(5,2,5). Increase `max_p`/`max_q` in
  `forecaster.py` if your series requires higher orders (at the cost of speed).
- If ARIMAX fitting fails with exogenous variables, the model automatically
  falls back to pure ARIMA.
- XGBoost requires `pip install xgboost`. macOS also needs `brew install libomp`. If unavailable, the pipeline runs with ARIMAX + RandomForest only.

---

## Changelog

### 2026-05-12 — v1.5
- **ETS model added** — `ETSForecaster` in `forecaster.py`; uses Holt-Winters `ExponentialSmoothing`; auto-selects seasonal period (4 for quarterly, 12 for monthly) from a DatetimeIndex with enough observations; falls back to trend-only then simple exponential smoothing; runs in both interactive and batch modes; appears in all exports and rankings

### 2026-05-12 — v1.4
- **Theta model added** — `ThetaForecaster` in `forecaster.py`; runs alongside RF and XGBoost in both interactive and batch modes; univariate (no exog required); appears in all exports and rankings
- **Pre-model normality requirement removed** — `AssumptionChecker._passes` now checks stationarity only; `is_normal` is retained for reference but not applied to input transforms. ARIMAX does not require normally distributed inputs — only residuals matter. This prevents unnecessary over-differencing that caused flat forecasts on many series
- **XGBoost/RF improvement** — `MLForecaster._build_feature_matrix` now includes a `t_index` (monotonic integer position) feature so tree models can learn long-term trends that short lag windows cannot capture alone

### 2026-04-27 — v1.3
- **`batch_run.py`** — new batch runner: edit a CONFIG block, run unattended across any number of dependent variables
- All listed dependent variables automatically excluded from each other's exogenous candidate pool (no cross-contamination)
- **`exports.py`** — added `BatchForecastExporter`: Master Forecasts + Master Metrics sheets (with leading `variable` column) plus per-variable Forecasts sheets and one PNG per variable
- Stationarity transform cascade extended: first-difference and percentage-growth-rate transforms added after log / Box-Cox / sqrt; each recovers to original level scale before evaluation
- Added `sample_data.xlsx` — 52-quarter example dataset (3 dependent variables, 4 exogenous candidates) for immediate use with `batch_run.py`
- Per-variable sheets in batch output now include a variables table (below the forecast table) showing each selected exogenous variable with its transform method and detail; single-run Variables sheet upgraded to the same format

### 2026-04-26 — v1.1
- All three models (ARIMAX, RandomForest, XGBoost) always run; ranked by RMSE
- Best-fit model (★) shown first in both the Forecasts sheet columns and Metrics rows
- Added Harvey-Collier linearity test per exogenous variable (first-differenced to avoid false positives on autocorrelated series)
- Robustified KPSS nlags to avoid failures on very short series (Python 3.13.5 compatible)
- XGBoost errors (including missing libomp on macOS) handled gracefully with install instructions
- Removed analyst prompt for ML comparison — all models always run

### 2026-04-26 — v1.0
- Three-file architecture: `main.py`, `forecaster.py`, `exports.py`
- Analyst-driven UI: file, time column, dependent column, N selection
- Automatic stationarity + normality testing with log / Box-Cox / sqrt transforms
- Exogenous variable selection: Pearson correlation filter + iterative VIF pruning
- `pmdarima.auto_arima` for ARIMA order selection
- Residual diagnostics: Jarque-Bera + Ljung-Box
- Accuracy reporting: MAE, RMSE, MAPE
- Excel export with Forecasts, Metrics, Variables sheets
- PNG chart (300 DPI) with confidence interval
