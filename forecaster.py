"""
forecaster.py
Core forecasting logic.

Public surface
──────────────
- Forecast            : uniform model output (numpy point forecast + optional CI + meta dict)
- BaseForecaster      : common ``fit_predict(y_train, X_train, X_future, h)`` interface
- AssumptionChecker   : dependent-variable variance-stabilizing transforms + residual diagnostics
- VariableSelector    : Spearman correlation + VIF exogenous selection (late-start aware)
- Model classes       : SeasonalNaive, ARIMAXForecaster, RandomForestForecaster,
                        XGBoostForecaster, ThetaForecaster, ETSForecaster, LinearLagForecaster
- detect_period()     : seasonal period from a DatetimeIndex (quarterly→4, monthly→12, weekly→52)
- build_models()      : registry factory → list of model instances.
                        Add / remove ONE line here to change the model roster.

Design
──────
Every model exposes the same ``fit_predict`` so a single rolling-origin backtester
(see ``evaluation.py``) can evaluate any of them, and the same call produces the final
deliverable forecast. Models return plain numpy arrays; the orchestrator attaches the
pandas index. No model holds the held-out split — that lives in the orchestrator.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import coint


@contextmanager
def _silence():
    """Suppress noisy convergence / runtime warnings only around a fit call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal period detection (shared by several models)
# ─────────────────────────────────────────────────────────────────────────────

def detect_period(index) -> int:
    """
    Infer the seasonal period from a DatetimeIndex. Returns 1 (non-seasonal) when
    the index is not datetime or no regular frequency can be inferred.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return 1
    freq = getattr(index, "freq", None)
    if freq is None:
        try:
            freq = pd.infer_freq(index)
        except Exception:
            freq = None
    if freq is None:
        return 1
    fs = str(freq).upper()
    if "Q" in fs:
        return 4
    if fs.startswith("M") or "ME" in fs or "MS" in fs or "BM" in fs:
        return 12
    if "W" in fs:
        return 52
    if fs.startswith("D"):
        return 7
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformInfo:
    """Variance-stabilizing transform applied to the dependent variable only."""
    method: str = "none"           # 'none' | 'log' | 'boxcox'
    lambda_: Optional[float] = None
    shift: float = 0.0             # added before transform to make the series positive


@dataclass
class Forecast:
    """Uniform output of every model's ``fit_predict``."""
    point:    np.ndarray
    ci_lower: Optional[np.ndarray] = None
    ci_upper: Optional[np.ndarray] = None
    meta:     Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Assumption checking and dependent-variable transformation
# ─────────────────────────────────────────────────────────────────────────────

class AssumptionChecker:
    """
    Handles the dependent variable's *variance-stabilizing* transform and the
    post-fit residual diagnostics.

    Differencing is left to SARIMAX, which selects its own (d, D) integration orders;
    this class only applies log / Box-Cox to stabilize variance when it helps.
    """

    MIN_OBS = 10

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    # ── dependent-variable transform ──────────────────────────────────────────

    def choose_transform(self, series: pd.Series, name: str = "y") -> TransformInfo:
        """
        Pick a variance-stabilizing transform via the Box-Cox MLE λ:
          λ ≈ 1  → no transform   |  λ ≈ 0 → log   |  otherwise → Box-Cox(λ).
        Falls back to 'none' on any failure. Never differences.
        """
        s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < self.MIN_OBS:
            return TransformInfo()

        shift = 0.0
        if (s <= 0).any():
            shift = float(-s.min() + 1.0)
        sp = s + shift
        try:
            with _silence():
                lam = float(stats.boxcox_normmax(sp.values, method="mle"))
        except Exception:
            return TransformInfo()

        lam = float(np.clip(lam, -1.0, 2.0))
        if abs(lam - 1.0) < 0.30:
            self._log(f"{name}: no transform (Box-Cox λ≈{lam:.2f})")
            return TransformInfo(method="none")
        if abs(lam) < 0.30:
            self._log(f"{name}: log transform (Box-Cox λ≈{lam:.2f}, shift={shift:.4f})")
            return TransformInfo(method="log", shift=shift)
        self._log(f"{name}: Box-Cox transform (λ={lam:.4f}, shift={shift:.4f})")
        return TransformInfo(method="boxcox", lambda_=lam, shift=shift)

    def apply(self, series: pd.Series, info: TransformInfo) -> pd.Series:
        s = pd.Series(series).astype(float)
        if info.method == "log":
            return np.log(s + info.shift)
        if info.method == "boxcox":
            with _silence():
                vals = stats.boxcox(s.clip(lower=-info.shift + 1e-9) + info.shift, lmbda=info.lambda_)
            return pd.Series(vals, index=s.index)
        return s

    def inverse(self, values: np.ndarray, info: TransformInfo) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if info.method == "log":
            return np.exp(np.clip(v, None, 700)) - info.shift
        if info.method == "boxcox":
            lam = info.lambda_
            if lam == 0:
                out = np.exp(np.clip(v, None, 700))
            else:
                # guard the domain: 1 + λv must be > 0, else inv_boxcox returns NaN
                base = np.clip(1.0 + lam * v, 1e-9, None)
                out = np.power(base, 1.0 / lam)
            return out - info.shift
        return v

    # ── residual diagnostics (post-fit) ─────────────────────────────────────────

    def residual_diagnostics(self, model, period: int = 1) -> Dict[str, object]:
        """
        Statistically correct residual checks on a fitted pmdarima ARIMA model:
          • Ljung-Box with the model_df (AR+MA params) correction,
          • Shapiro-Wilk normality (reliable at small n, unlike Jarque-Bera),
          • Engle's ARCH test for heteroskedasticity.
        Initialization-transient residuals are trimmed and residuals standardized.
        Returns a dict of p-values + human-readable notes (never raises).
        """
        out: Dict[str, object] = {
            "ljung_box_p": np.nan, "normality_p": np.nan, "arch_p": np.nan, "notes": ""
        }
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

            resid = np.asarray(model.resid(), dtype=float)
            order = getattr(model, "order", (0, 0, 0))
            sorder = getattr(model, "seasonal_order", (0, 0, 0, 0))
            p, d, q = order
            P, D, Q, m = (sorder + (0, 0, 0, 0))[:4]

            trim = int(d + D * max(m, 1) + p + P * max(m, 1))
            trim = max(0, min(trim, len(resid) // 4))
            r = resid[trim:]
            r = r[np.isfinite(r)]
            if len(r) < 8:
                out["notes"] = "too few residuals for diagnostics"
                return out
            sd = np.std(r)
            r = (r - np.mean(r)) / sd if sd > 0 else r

            model_df = int(p + q + P + Q)
            lags = min(2 * period if period > 1 else 10, max(model_df + 2, len(r) // 5))
            with _silence():
                lb = acorr_ljungbox(r, lags=[lags], model_df=model_df, return_df=True)
                out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
                out["normality_p"] = float(stats.shapiro(r)[1]) if len(r) <= 5000 else np.nan
                try:
                    out["arch_p"] = float(het_arch(r, nlags=min(lags, len(r) // 5))[1])
                except Exception:
                    out["arch_p"] = np.nan

            notes = []
            if out["ljung_box_p"] < 0.05:
                notes.append("residual autocorrelation")
            if out["normality_p"] < 0.05:
                notes.append("non-normal residuals")
            if out["arch_p"] < 0.05:
                notes.append("heteroskedasticity (ARCH)")
            out["notes"] = "; ".join(notes) if notes else "OK"
        except Exception as e:
            out["notes"] = f"diagnostics skipped ({e})"
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Exogenous variable selection
# ─────────────────────────────────────────────────────────────────────────────

class VariableSelector:
    """
    Selects exogenous regressors via:
      1. Coverage + future-availability filter (handles late-starting series),
      2. **Stationarity-aware** Spearman relevance: y and each exog are differenced
         to a common stationary order before correlating, so two unrelated integrated
         series do not correlate spuriously (the classic spurious-regression trap).
         A non-stationary pair is also kept if it is genuinely **cointegrated** — the
         one case where a levels relationship is valid (Engle-Granger test).
      3. cap on the number of regressors (~n/10) to prevent over-parameterization,
      4. iterative VIF pruning (with an intercept) to remove collinear regressors.

    After ``select`` runs, ``self.report_`` holds the per-exog integration order and the
    basis on which each was kept/dropped (for the exported diagnostics).
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.report_: Dict[str, object] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    @staticmethod
    def _ndiffs(s: pd.Series, max_d: int = 2) -> int:
        """Integration order via the KPSS unit-root test (same test auto_arima uses
        for d). Returns 0 for a stationary series."""
        s = pd.Series(s).dropna()
        if len(s) < 12:
            return 0
        try:
            from pmdarima.arima import ndiffs
            return int(ndiffs(s.values, test="kpss", max_d=max_d))
        except Exception:
            return 0

    def select(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        X_future: Optional[pd.DataFrame] = None,
        corr_threshold: float = 0.30,
        max_vif: float = 10.0,
        max_exog: Optional[int] = None,
        min_coverage: float = 0.80,
        coint_alpha: float = 0.01,   # strict: a levels relationship is rescued only on strong cointegration evidence
    ) -> List[str]:
        self.report_ = {}
        if X is None or X.empty:
            return []

        n = len(y)
        if max_exog is None:
            max_exog = max(1, n // 10)

        dy = self._ndiffs(y)
        self.report_["_y_order"] = dy
        self._log(f"dependent variable integration order: I({dy})")

        # ── Step 1+2: coverage, future-availability, stationarity-aware relevance ─
        scored: List[Tuple[str, float]] = []
        for col in X.columns:
            try:
                if X_future is not None and col in X_future.columns and X_future[col].isna().any():
                    self._log(f"{col}: dropped — exog missing over the forecast horizon")
                    continue
                cov = float(X[col].notna().mean())
                if cov < min_coverage:
                    self._log(f"{col}: dropped — only {cov:.0%} coverage (< {min_coverage:.0%})")
                    continue
                pair = pd.concat([y, X[col]], axis=1).dropna()
                if len(pair) < 6:
                    continue

                dx = self._ndiffs(X[col])
                d_common = max(dy, dx)
                ys, xs = pair.iloc[:, 0], pair.iloc[:, 1]
                if d_common > 0:
                    diffed = pd.concat([ys.diff(d_common), xs.diff(d_common)], axis=1).dropna()
                else:
                    diffed = pd.concat([ys, xs], axis=1)
                if len(diffed) < 5:
                    continue
                r = abs(float(stats.spearmanr(diffed.iloc[:, 0], diffed.iloc[:, 1]).statistic))
                r = r if np.isfinite(r) else 0.0

                keep = r >= corr_threshold
                basis = ("differences" if d_common > 0 else "levels") if keep else "dropped"

                # cointegration rescue: a genuine long-run levels relationship
                coint_p = np.nan
                if dy >= 1 and dx >= 1:
                    try:
                        with _silence():
                            coint_p = float(coint(pair.iloc[:, 0], pair.iloc[:, 1])[1])
                    except Exception:
                        coint_p = np.nan
                    if not keep and np.isfinite(coint_p) and coint_p < coint_alpha:
                        keep, basis = True, "cointegration"

                self.report_[col] = {
                    "order": dx, "spearman": round(r, 3),
                    "coint_p": (round(coint_p, 3) if np.isfinite(coint_p) else None),
                    "basis": basis,
                }
                msg = f"{col}: I({dx}), |Spearman({'Δ' if d_common > 0 else 'lvl'})|={r:.3f}"
                if np.isfinite(coint_p):
                    msg += f", coint p={coint_p:.3f}"
                self._log(msg + (f" → kept ({basis})" if keep else " → dropped (spurious/weak)"))
                if keep:
                    scored.append((col, max(r, corr_threshold) if basis == "cointegration" else r))
            except Exception:
                continue

        if not scored:
            self._log("No exogenous variable passes the stationarity-aware relevance test — pure ARIMA")
            return []

        # ── Step 3: cap to the strongest max_exog by relevance ─────────────────
        scored.sort(key=lambda t: t[1], reverse=True)
        if len(scored) > max_exog:
            self._log(f"Capping exogenous count {len(scored)} → {max_exog} (sample size guard)")
        selected = [c for c, _ in scored[:max_exog]]

        # ── Step 4: VIF pruning with an intercept ──────────────────────────────
        while len(selected) > 1:
            sub = X[selected].dropna()
            if len(sub) <= len(selected) + 1:
                break
            try:
                with _silence():
                    design = add_constant(sub.values, has_constant="add")
                    # column 0 is the intercept; skip it when reading VIFs
                    vifs = [variance_inflation_factor(design, i + 1) for i in range(len(selected))]
            except Exception:
                break
            mx = max(vifs)
            if mx <= max_vif:
                break
            drop_idx = vifs.index(mx)
            self._log(f"{selected[drop_idx]}: VIF={mx:.2f} > {max_vif} → dropped (collinear)")
            selected.pop(drop_idx)

        self._log(f"Final exogenous selection: {selected or 'none'}")
        return selected


# ─────────────────────────────────────────────────────────────────────────────
# Common model interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseForecaster:
    """All models implement ``fit_predict`` and declare whether they use exog."""
    name: str = "base"
    uses_exog: bool = False

    def fit_predict(
        self,
        y_train: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_future: Optional[pd.DataFrame],
        h: int,
    ) -> Forecast:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal-naive baseline
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalNaive(BaseForecaster):
    """ŷ_{T+i} = y_{T+i-m}. With m=1 this is the plain random-walk naive forecast."""
    uses_exog = False

    def __init__(self, period: int = 1) -> None:
        self.period = max(1, int(period))
        self.name = "SeasonalNaive" if self.period > 1 else "Naive"

    def fit_predict(self, y_train, X_train, X_future, h) -> Forecast:
        vals = pd.Series(y_train).astype(float).ffill().bfill().values
        m = self.period if len(vals) >= self.period else 1
        last = vals[-m:]
        point = np.array([last[i % m] for i in range(h)], dtype=float)
        return Forecast(point=point, meta={"period": m})


# ─────────────────────────────────────────────────────────────────────────────
# ARIMAX / SARIMAX
# ─────────────────────────────────────────────────────────────────────────────

class ARIMAXForecaster(BaseForecaster):
    """
    Auto-(S)ARIMAX with exogenous regressors.

    Key behaviours:
      • exogenous regressors passed via the ``X=`` keyword (pmdarima ≥ 2.0),
      • each exog is passed through a monotone Yeo-Johnson transform whose λ is chosen
        on the TRAINING slice to best linearize its relationship with y — this honours
        SARIMAX's linear-in-exog assumption WITHOUT dropping a strong non-linear driver
        (one column per exog, so no regressor blow-up). λ=1 (identity) is kept unless a
        transform meaningfully improves linearity,
      • seasonal SARIMAX enabled when a seasonal period is detected,
      • the (p,d,q)(P,D,Q) order is searched once and cached; later calls (backtest
        folds) refit that fixed order, which keeps a 10-variable run fast,
      • late-starting exog are aligned by trimming the shared leading-NaN region.
    """
    uses_exog = True

    # candidate Yeo-Johnson powers; 1.0 is the identity (no transform)
    _YJ_LAMBDAS = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    _YJ_GAIN = 0.03   # adopt a non-identity λ only if it improves |corr| by at least this

    def __init__(self, period: int = 1, reuse_order: bool = True, verbose: bool = True) -> None:
        self.period = max(1, int(period))
        self.reuse_order = reuse_order
        self.verbose = verbose
        self.name = "ARIMAX"
        self._checker = AssumptionChecker(verbose)
        self._order: Optional[Tuple] = None
        self._seasonal_order: Optional[Tuple] = None

    def reset_order(self) -> None:
        self._order = None
        self._seasonal_order = None

    @staticmethod
    def _align(y: pd.Series, X: Optional[pd.DataFrame]):
        """Trim the common leading-NaN region; interpolate internal gaps in exog."""
        y = pd.Series(y).astype(float)
        if X is None or X.empty:
            return y.ffill().bfill(), None
        X = X.astype(float)
        joined = pd.concat([y.rename("__y__"), X], axis=1)
        first_valid = joined.apply(lambda c: c.first_valid_index())
        start = max([i for i in first_valid if i is not None], default=joined.index[0])
        joined = joined.loc[start:]
        y_a = joined["__y__"].interpolate().ffill().bfill()
        X_a = joined.drop(columns="__y__").interpolate().ffill().bfill()
        return y_a, X_a

    @staticmethod
    def _yeojohnson(x: np.ndarray, lam: float) -> np.ndarray:
        """Yeo-Johnson power transform (monotone increasing for every λ; defined for
        negative, zero and positive values)."""
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        pos = x >= 0
        neg = ~pos
        if lam != 0.0:
            out[pos] = ((x[pos] + 1.0) ** lam - 1.0) / lam
        else:
            out[pos] = np.log1p(x[pos])
        if (2.0 - lam) != 0.0:
            out[neg] = -(((-x[neg] + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam))
        else:
            out[neg] = -np.log1p(-x[neg])
        return out

    def _linearize_exog(self, Xtr_df, Xfu_df, y_t):
        """
        For each exogenous column, pick the Yeo-Johnson λ (on the TRAINING slice only)
        that maximizes the linear correlation between the transformed exog and the
        transformed y — i.e. the monotone transform that best linearizes the
        relationship so SARIMAX's linear assumption holds. The same λ and the training
        mean/std are applied to the forecast-horizon exog. Returns standardized arrays
        (train, future) and the chosen λ per column.
        """
        y = np.asarray(y_t, dtype=float)
        train_cols, fut_cols, lambdas = [], [], {}
        for col in Xtr_df.columns:
            xtr = Xtr_df[col].values.astype(float)
            best_lam, best_abs, base_abs = 1.0, -1.0, None
            for lam in self._YJ_LAMBDAS:
                try:
                    t = self._yeojohnson(xtr, lam)
                    if not np.all(np.isfinite(t)) or np.std(t) == 0:
                        continue
                    r = abs(np.corrcoef(t, y)[0, 1])
                except Exception:
                    continue
                if not np.isfinite(r):
                    continue
                if lam == 1.0:
                    base_abs = r
                if r > best_abs:
                    best_abs, best_lam = r, lam
            # keep identity unless a transform meaningfully improves linearity
            if base_abs is not None and best_lam != 1.0 and (best_abs - base_abs) < self._YJ_GAIN:
                best_lam = 1.0

            t_tr = self._yeojohnson(xtr, best_lam)
            mu, sd = float(np.mean(t_tr)), float(np.std(t_tr))
            sd = sd if sd > 0 else 1.0
            train_cols.append((t_tr - mu) / sd)
            if Xfu_df is not None:
                t_fu = self._yeojohnson(Xfu_df[col].values.astype(float), best_lam)
                fut_cols.append((t_fu - mu) / sd)
            lambdas[col] = round(float(best_lam), 2)

        exog_tr = np.column_stack(train_cols) if train_cols else None
        exog_fu = np.column_stack(fut_cols) if fut_cols else None
        return exog_tr, exog_fu, lambdas

    def fit_predict(self, y_train, X_train, X_future, h) -> Forecast:
        import pmdarima as pm

        exog_cols = list(X_train.columns) if (X_train is not None and not X_train.empty) else []
        y_aligned, X_aligned = self._align(y_train, X_train if exog_cols else None)

        info = self._checker.choose_transform(y_aligned, self.name)
        y_t = self._checker.apply(y_aligned, info).values

        exog_lambdas: Dict[str, float] = {}
        if exog_cols and X_future is not None and not X_future.empty:
            exog_tr, exog_fu, exog_lambdas = self._linearize_exog(
                X_aligned[exog_cols], X_future[exog_cols].astype(float), y_t
            )
        else:
            # no usable forecast-horizon exog → fit without regressors
            exog_tr = exog_fu = None
            exog_cols = []

        seasonal = self.period > 1 and len(y_t) >= 2 * self.period
        model = None
        try:
            if self.reuse_order and self._order is not None:
                model = pm.ARIMA(
                    order=self._order,
                    seasonal_order=self._seasonal_order or (0, 0, 0, 0),
                    suppress_warnings=True,
                )
                with _silence():
                    model.fit(y_t, X=exog_tr)
            else:
                with _silence():
                    model = pm.auto_arima(
                        y_t, X=exog_tr,
                        seasonal=seasonal, m=self.period if seasonal else 1,
                        stepwise=True, information_criterion="aicc",
                        max_p=5, max_q=5, max_d=2, max_P=2, max_Q=2, max_D=1,
                        error_action="ignore", suppress_warnings=True,
                    )
                self._order = model.order
                self._seasonal_order = getattr(model, "seasonal_order", (0, 0, 0, 0))
        except Exception:
            # last-resort fallback: pure ARIMA, no exog
            with _silence():
                model = pm.auto_arima(
                    y_t, seasonal=False, stepwise=True,
                    max_p=5, max_q=5, max_d=2,
                    error_action="ignore", suppress_warnings=True,
                )
            self._order = model.order
            self._seasonal_order = (0, 0, 0, 0)
            exog_fu = None
            exog_cols = []
            exog_lambdas = {}

        with _silence():
            fc_t, ci_t = model.predict(n_periods=h, X=exog_fu, return_conf_int=True)

        point = self._checker.inverse(np.asarray(fc_t), info)
        ci_lo = self._checker.inverse(np.asarray(ci_t)[:, 0], info)
        ci_hi = self._checker.inverse(np.asarray(ci_t)[:, 1], info)

        diagnostics = self._checker.residual_diagnostics(model, self.period)
        meta = {
            "order": tuple(model.order),
            "seasonal_order": tuple(getattr(model, "seasonal_order", (0, 0, 0, 0))),
            "aic": float(model.aic()) if hasattr(model, "aic") else np.nan,
            "selected_exog": list(exog_cols),
            "exog_lambdas": exog_lambdas,
            "transform": info.method,
            "diagnostics": diagnostics,
        }
        return Forecast(point=point, ci_lower=ci_lo, ci_upper=ci_hi, meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
# Supervised lag models (Random Forest / XGBoost / ElasticNet)
# ─────────────────────────────────────────────────────────────────────────────

class _SupervisedLagForecaster(BaseForecaster):
    """
    Shared machinery for regression-on-lags models. Features for predicting the target
    at time t are its own lags (t-1…t-L), the contemporaneous selected exog, and —
    only for models that can extrapolate (linear) — a time index. Multi-step forecasts
    are recursive.

    Models that set ``difference = True`` are first made stationary: the target is
    differenced to its integration order (and the exog by the same order), the model is
    fit on the differences, and the forecast is integrated back to levels. This keeps a
    linear regression from running on integrated series (a spurious-regression trap) and
    lets it follow a trend through the drift term instead of mean-reverting. Tree models
    leave ``difference = False`` (they handle levels directly).
    """
    uses_exog = True
    add_time_index: bool = False
    difference: bool = False

    def __init__(self, n_lags_max: int = 5, verbose: bool = True) -> None:
        self.n_lags_max = n_lags_max
        self.verbose = verbose

    def _make_estimator(self):
        raise NotImplementedError

    @staticmethod
    def _integrate(diff_forecast: np.ndarray, y_train: np.ndarray, d: int) -> np.ndarray:
        """Invert d-th differencing: rebuild levels from forecast d-th differences,
        seeding each integration with the last value of the corresponding lower-order
        difference of the training series."""
        cur = np.asarray(diff_forecast, dtype=float)
        for k in range(d - 1, -1, -1):
            seed = float(np.diff(y_train, n=k)[-1])
            cur = seed + np.cumsum(cur)
        return cur

    def fit_predict(self, y_train, X_train, X_future, h) -> Forecast:
        y = pd.Series(y_train).astype(float).reset_index(drop=True)
        n = len(y)
        y_arr = y.values

        exog_cols = list(X_train.columns) if (X_train is not None and not X_train.empty) else []
        have_future = bool(exog_cols) and X_future is not None and not X_future.empty
        if exog_cols and not have_future:
            exog_cols = []     # cannot supply exog over the horizon → drop it
        Xtr = X_train.reset_index(drop=True).astype(float) if exog_cols else None
        Xfu = X_future.reset_index(drop=True).astype(float) if exog_cols else None

        # integration order — only the linear model differences; trees stay on levels
        d = 0
        if self.difference and n > 14:
            try:
                from pmdarima.arima import ndiffs
                d = int(ndiffs(y_arr, test="kpss", max_d=2))
            except Exception:
                d = 0
        use_time = self.add_time_index and d == 0   # differenced model: intercept carries the drift

        z = np.diff(y_arr, n=d) if d > 0 else y_arr.copy()
        m = len(z)

        # exog aligned to z (and over the horizon), differenced by the same order
        exog_z: Dict[str, np.ndarray] = {}
        exog_fu: Dict[str, np.ndarray] = {}
        for c in exog_cols:
            full = np.concatenate([Xtr[c].values, Xfu[c].values])
            col_d = np.diff(full, n=d) if d > 0 else full
            exog_z[c] = col_d[:m]
            exog_fu[c] = col_d[-h:]

        n_lags = max(1, min(self.n_lags_max, m - h - 2))
        rows, targets = [], []
        for t in range(n_lags, m):
            feat = [z[t - k] for k in range(1, n_lags + 1)]
            feat += [exog_z[c][t] for c in exog_cols]
            if use_time:
                feat.append(t)
            rows.append(feat)
            targets.append(z[t])

        feat_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        keep = feat_df.notna().all(axis=1).values
        feat_df = feat_df[keep]
        target_arr = np.asarray(targets)[keep]
        if len(feat_df) < 3:
            raise ValueError("not enough rows after lag construction")

        est = self._make_estimator()
        with _silence():
            est.fit(feat_df.values, target_arr)

        history = list(z)
        zhat = []
        for i in range(h):
            feat = [history[-k] for k in range(1, n_lags + 1)]
            feat += [float(exog_fu[c][i]) for c in exog_cols]
            if use_time:
                feat.append(m + i)
            p = float(est.predict(np.array(feat).reshape(1, -1))[0])
            zhat.append(p)
            history.append(p)

        point = self._integrate(np.array(zhat), y_arr, d) if d > 0 else np.array(zhat, dtype=float)
        return Forecast(point=point,
                        meta={"selected_exog": exog_cols, "n_lags": n_lags, "difference_order": d})


class RandomForestForecaster(_SupervisedLagForecaster):
    name = "RandomForest"
    add_time_index = False   # trees cannot extrapolate a time index — omit it

    def _make_estimator(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)


class XGBoostForecaster(_SupervisedLagForecaster):
    name = "XGBoost"
    add_time_index = False

    def _make_estimator(self):
        from xgboost import XGBRegressor   # raises ImportError if unavailable
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.9, random_state=42, verbosity=0)


class LinearLagForecaster(_SupervisedLagForecaster):
    """
    ElasticNet on standardized lags + exog. Linear → can follow a trend (unlike trees);
    the L1/L2 penalty keeps it stable when lags/exog are collinear. The target is
    differenced to stationarity before fitting (``difference = True``) so the regression
    is not run on integrated series, and the elastic-net penalty is selected by
    time-series cross-validation (no shuffling, train always precedes validation).
    """
    name = "ElasticNet"
    add_time_index = True   # used only when the series is already stationary (d == 0)
    difference = True

    def _make_estimator(self):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                         cv=TimeSeriesSplit(n_splits=3), max_iter=5000, random_state=42),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Theta
# ─────────────────────────────────────────────────────────────────────────────

class ThetaForecaster(BaseForecaster):
    """statsmodels ThetaModel; deseasonalizes when a seasonal period is present."""
    uses_exog = False
    name = "Theta"

    def __init__(self, period: int = 1, verbose: bool = True) -> None:
        self.period = max(1, int(period))
        self.verbose = verbose

    def fit_predict(self, y_train, X_train, X_future, h) -> Forecast:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        vals = pd.Series(pd.Series(y_train).astype(float).ffill().bfill().values)
        deseason = self.period > 1 and len(vals) >= 2 * self.period
        with _silence():
            tm = ThetaModel(
                vals,
                period=self.period if deseason else None,
                deseasonalize=deseason,
            )
            fit = tm.fit(disp=False)
            preds = np.asarray(fit.forecast(h), dtype=float)
        return Forecast(point=preds, meta={"deseasonalized": deseason})


# ─────────────────────────────────────────────────────────────────────────────
# ETS (Exponential Smoothing)
# ─────────────────────────────────────────────────────────────────────────────

class ETSForecaster(BaseForecaster):
    """
    Holt-Winters / ETS. Searches a small grid of {trend, damped, seasonal} configs
    and keeps the lowest-AICc fit. Damping is included so the trend cannot run away
    into implausible jumps. Multiplicative options are tried only on positive data.
    """
    uses_exog = False
    name = "ETS"

    def __init__(self, period: int = 1, verbose: bool = True) -> None:
        self.period = max(1, int(period))
        self.verbose = verbose

    def fit_predict(self, y_train, X_train, X_future, h) -> Forecast:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        vals = pd.Series(y_train).astype(float).ffill().bfill().values
        positive = bool(np.all(vals > 0))
        use_season = self.period > 1 and len(vals) >= 2 * self.period

        configs: List[Dict] = []
        if use_season:
            configs.append({"trend": "add", "damped_trend": True,  "seasonal": "add"})
            configs.append({"trend": "add", "damped_trend": False, "seasonal": "add"})
            if positive:
                configs.append({"trend": "add", "damped_trend": True, "seasonal": "mul"})
        configs.append({"trend": "add", "damped_trend": True,  "seasonal": None})
        configs.append({"trend": "add", "damped_trend": False, "seasonal": None})
        configs.append({"trend": None,  "damped_trend": False, "seasonal": None})

        best_fit, best_cfg, best_aicc = None, None, np.inf
        for cfg in configs:
            try:
                with _silence():
                    fit = ExponentialSmoothing(
                        vals,
                        trend=cfg["trend"],
                        damped_trend=cfg["damped_trend"],
                        seasonal=cfg["seasonal"],
                        seasonal_periods=self.period if cfg["seasonal"] else None,
                        initialization_method="estimated",
                    ).fit(optimized=True)
                aicc = getattr(fit, "aicc", np.inf)
                if np.isfinite(aicc) and aicc < best_aicc:
                    best_fit, best_cfg, best_aicc = fit, cfg, aicc
            except Exception:
                continue

        if best_fit is None:
            raise ValueError("ETS fitting failed under all configurations")

        with _silence():
            preds = np.asarray(best_fit.forecast(h), dtype=float)
        return Forecast(point=preds, meta={"config": best_cfg, "aicc": float(best_aicc)})


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def build_models(period: int = 1, verbose: bool = True) -> List[BaseForecaster]:
    """
    The model roster. Add or remove a single line here to change which models run;
    the orchestrators and the backtester pick them up automatically. The ensemble
    is assembled separately in ``evaluation.py`` from these models' backtests.
    """
    return [
        SeasonalNaive(period=period),
        ARIMAXForecaster(period=period, reuse_order=True, verbose=verbose),
        RandomForestForecaster(verbose=verbose),
        XGBoostForecaster(verbose=verbose),
        ThetaForecaster(period=period, verbose=verbose),
        ETSForecaster(period=period, verbose=verbose),
        LinearLagForecaster(verbose=verbose),
    ]
