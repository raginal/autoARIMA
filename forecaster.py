"""
forecaster.py
Core forecasting logic: assumption checking, variable selection,
ARIMAX modelling, and ML comparison (Random Forest + XGBoost).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pmdarima as pm
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformInfo:
    method: str = "none"          # 'none' | 'log' | 'boxcox' | 'sqrt' | 'diff' | 'pct_change'
    lambda_: Optional[float] = None
    shift: float = 0.0
    anchor: float = 0.0           # last actual level (original scale) before forecast; used by diff / pct_change inverse transforms


@dataclass
class ForecastResult:
    forecast:      pd.Series = field(default_factory=pd.Series)
    ci_lower:      pd.Series = field(default_factory=pd.Series)
    ci_upper:      pd.Series = field(default_factory=pd.Series)
    actuals:       pd.Series = field(default_factory=pd.Series)
    train_actuals: pd.Series = field(default_factory=pd.Series)
    order:         Tuple     = (0, 0, 0)
    aic:           float     = float("nan")
    mae:           float     = float("nan")
    rmse:          float     = float("nan")
    mape:          float     = float("nan")
    selected_exog:   List[str]              = field(default_factory=list)
    exog_transforms: Dict[str, "TransformInfo"] = field(default_factory=dict)
    dep_transform:   TransformInfo          = field(default_factory=TransformInfo)
    n_forecast:      int                    = 0
    model:           object                 = None


@dataclass
class MLResult:
    method:   str        = ""
    forecast: pd.Series  = field(default_factory=pd.Series)
    actuals:  pd.Series  = field(default_factory=pd.Series)
    mae:      float      = float("nan")
    rmse:     float      = float("nan")
    mape:     float      = float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Assumption checking and variable transformation
# ─────────────────────────────────────────────────────────────────────────────

class AssumptionChecker:
    """
    Tests stationarity (ADF + KPSS) and normality (Jarque-Bera).
    Applies log, Box-Cox, or sqrt transforms to satisfy assumptions.
    Raises ValueError if a variable cannot be made stationary.
    """

    MIN_OBS = 10   # minimum observations needed for reliable tests

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    # ── individual tests ─────────────────────────────────────────────────────

    def is_stationary(self, s: pd.Series) -> bool:
        """ADF rejects unit-root AND KPSS fails to reject stationarity."""
        s = s.dropna()
        if len(s) < self.MIN_OBS:
            return True
        try:
            adf_p = adfuller(s, autolag="AIC")[1]
            # nlags="auto" can fail on very short series; cap explicitly
            nlags = max(1, min(int(len(s) ** 0.5), len(s) // 5))
            kpss_p = kpss(s, regression="c", nlags=nlags)[1]
            return adf_p < 0.05 and kpss_p > 0.05
        except Exception:
            return False

    def is_normal(self, s: pd.Series) -> bool:
        """Jarque-Bera test for normality (p > 0.05 → approximately normal)."""
        s = s.dropna()
        if len(s) < 8:
            return True
        try:
            _, p = stats.jarque_bera(s)
            return p > 0.05
        except Exception:
            return True

    def _passes(self, s: pd.Series) -> bool:
        return self.is_stationary(s) and self.is_normal(s)

    # ── public API ────────────────────────────────────────────────────────────

    def check_and_transform(
        self, series: pd.Series, name: str
    ) -> Tuple[pd.Series, TransformInfo]:
        """
        Return (transformed_series, TransformInfo).
        Raises ValueError if stationarity cannot be achieved.
        """
        info = TransformInfo()
        s = series.copy().astype(float).replace([np.inf, -np.inf], np.nan)

        if s.isna().all():
            raise ValueError("all values are null or infinite")

        s_filled = s.ffill().bfill()

        # ── no transform needed ───────────────────────────────────────────────
        if self._passes(s_filled):
            self._log(f"{name}: OK — no transform needed")
            return s, info

        # ── compute shift to make series strictly positive ───────────────────
        min_val = float(s_filled.min())
        shift   = max(0.0, -min_val + 1.0)

        # ── log ───────────────────────────────────────────────────────────────
        try:
            s_log = np.log(s_filled + shift)
            if self._passes(s_log):
                self._log(f"{name}: log transform applied (shift={shift:.4f})")
                info.method = "log"
                info.shift  = shift
                return np.log(s + shift), info
        except Exception:
            pass

        # ── Box-Cox ───────────────────────────────────────────────────────────
        try:
            s_bc_arr, lam = stats.boxcox(s_filled + shift)
            s_bc = pd.Series(s_bc_arr, index=s_filled.index)
            if self._passes(s_bc):
                self._log(
                    f"{name}: Box-Cox transform applied (λ={lam:.4f}, shift={shift:.4f})"
                )
                info.method   = "boxcox"
                info.lambda_  = lam
                info.shift    = shift
                return (
                    pd.Series(
                        stats.boxcox(s.fillna(s.median()) + shift, lmbda=lam),
                        index=s.index,
                    ).where(s.notna(), other=np.nan),
                    info,
                )
        except Exception:
            pass

        # ── sqrt ──────────────────────────────────────────────────────────────
        try:
            s_sqrt = np.sqrt(s_filled + shift)
            if self._passes(s_sqrt):
                self._log(f"{name}: sqrt transform applied (shift={shift:.4f})")
                info.method = "sqrt"
                info.shift  = shift
                return np.sqrt(s + shift), info
        except Exception:
            pass

        # ── first difference ──────────────────────────────────────────────────
        try:
            s_diff = s_filled.diff()
            if self._passes(s_diff.dropna()):
                self._log(f"{name}: first-difference transform applied")
                info.method = "diff"
                return s.diff(), info
        except Exception:
            pass

        # ── percentage growth rate ────────────────────────────────────────────
        try:
            s_pct = (s_filled + shift).pct_change() * 100
            if self._passes(s_pct.dropna()):
                self._log(
                    f"{name}: percentage growth rate transform applied (shift={shift:.4f})"
                )
                info.method = "pct_change"
                info.shift  = shift
                return (s + shift).pct_change() * 100, info
        except Exception:
            pass

        # ── stationary-only fallback (normality less critical for exog) ───────
        if self.is_stationary(s_filled):
            self._log(f"{name}: stationary but non-normal — kept without transform")
            return s, info

        raise ValueError(
            f"'{name}' cannot be made stationary after log / Box-Cox / sqrt / diff / pct_change transforms"
        )

    def inverse_transform(
        self, values: np.ndarray, info: TransformInfo
    ) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if info.method == "log":
            return np.exp(v) - info.shift
        if info.method == "boxcox":
            return inv_boxcox(v, info.lambda_) - info.shift
        if info.method == "sqrt":
            return np.clip(v, 0, None) ** 2 - info.shift
        if info.method == "diff":
            # cumulative sum from the last known level restores the original scale
            return np.cumsum(v) + info.anchor
        if info.method == "pct_change":
            # reconstruct levels: anchor_shifted * ∏(1 + pct_i/100) − shift
            anchor_shifted = info.anchor + info.shift
            return anchor_shifted * np.cumprod(1.0 + v / 100.0) - info.shift
        return v  # 'none'

    def check_residuals(self, model: pm.ARIMA) -> None:
        """Print residual diagnostic checks after model fitting."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            resid = np.array(model.resid())
            if len(resid) < 5:
                return
            _, norm_p = stats.jarque_bera(resid)
            lags = min(10, max(1, len(resid) // 5))
            lb_p = float(
                acorr_ljungbox(resid, lags=[lags], return_df=True)[
                    "lb_pvalue"
                ].iloc[0]
            )
            self._log(
                f"Residual normality   (Jarque-Bera): p={norm_p:.4f}"
                + ("  ✓" if norm_p > 0.05 else "  ⚠ non-normal")
            )
            self._log(
                f"Residual autocorr    (Ljung-Box):   p={lb_p:.4f}"
                + ("  ✓" if lb_p > 0.05 else "  ⚠ autocorrelated")
            )
        except Exception as e:
            self._log(f"Residual diagnostics skipped: {e}")

    def check_linearity(self, x: pd.Series, y: pd.Series, name: str) -> bool:
        """
        Harvey-Collier test for linearity of the y ~ x relationship.

        Both series are first-differenced before the test to remove trend and
        serial autocorrelation — raw time series would cause false positives
        because OLS residuals inherit the autocorrelation of y.

        A non-linear result is a WARNING only — the variable is kept because:
          (a) transforms may already have linearised the relationship, and
          (b) the ML models (RF/XGBoost) always run and capture non-linear effects.
        Returns True if linear (or if the test cannot be run).
        """
        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import linear_harvey_collier

            df = pd.DataFrame({"y": y.values, "x": x.values}).dropna()
            df = df.diff().dropna()  # first-difference removes trend + autocorrelation

            if len(df) < 15:
                self._log(f"{name}: linearity check skipped (< 15 obs after differencing)")
                return True

            X_sm = sm.add_constant(df["x"].values)
            ols  = sm.OLS(df["y"].values, X_sm).fit()
            _, p = linear_harvey_collier(ols)

            if p < 0.05:
                self._log(
                    f"{name}: ⚠ possible non-linearity (Harvey-Collier p={p:.4f})"
                    " — ML models will also capture non-linear effects"
                )
                return False
            self._log(f"{name}: linearity OK (Harvey-Collier p={p:.4f})")
            return True
        except Exception as e:
            self._log(f"{name}: linearity check skipped ({e})")
            return True


# ─────────────────────────────────────────────────────────────────────────────
# Exogenous variable selection
# ─────────────────────────────────────────────────────────────────────────────

class VariableSelector:
    """
    Selects exogenous regressors via:
      1. Pearson correlation filter (|r| >= corr_threshold)
      2. Iterative VIF pruning (removes highest-VIF variable until VIF < max_vif)
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    def select(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        corr_threshold: float = 0.10,
        max_vif: float = 10.0,
    ) -> List[str]:
        if X.empty:
            return []

        # ── Step 1: correlation filter ────────────────────────────────────────
        keep: List[str] = []
        for col in X.columns:
            try:
                r = abs(float(y.corr(X[col])))
                tag = "kept" if r >= corr_threshold else "dropped (low correlation)"
                self._log(f"{col}: |corr|={r:.3f} → {tag}")
                if r >= corr_threshold:
                    keep.append(col)
            except Exception:
                pass

        if not keep:
            self._log("No variables exceed correlation threshold — using pure ARIMA")
            return []

        # ── Step 2: VIF pruning ───────────────────────────────────────────────
        selected = list(keep)
        while len(selected) > 1:
            sub = X[selected].dropna()
            if len(sub) < 2:
                break
            try:
                vifs = [
                    variance_inflation_factor(sub.values, i)
                    for i in range(len(selected))
                ]
            except Exception:
                break
            mx = max(vifs)
            if mx <= max_vif:
                break
            drop_idx = vifs.index(mx)
            self._log(
                f"{selected[drop_idx]}: VIF={mx:.2f} > {max_vif} → dropped (multicollinearity)"
            )
            selected.pop(drop_idx)

        self._log(f"Final exogenous selection: {selected}")
        return selected


# ─────────────────────────────────────────────────────────────────────────────
# ARIMAX forecaster
# ─────────────────────────────────────────────────────────────────────────────

class ARIMAXForecaster:
    """
    Full pipeline:
      1. Transform dependent variable to satisfy ARIMA assumptions
      2. Transform and validate each candidate exogenous variable
      3. Select exogenous variables via correlation + VIF
      4. Auto-select ARIMA(p,d,q) order via AIC
      5. Forecast held-out N periods; evaluate accuracy
      6. Run residual diagnostics
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose        = verbose
        self.model_:        Optional[pm.ARIMA]          = None
        self.dep_transform_: TransformInfo               = TransformInfo()
        self.exog_transforms_: Dict[str, TransformInfo] = {}
        self.selected_exog_: List[str]                  = []
        self._checker  = AssumptionChecker(verbose)
        self._selector = VariableSelector(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}")

    def fit_and_forecast(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        n_forecast: int,
    ) -> ForecastResult:

        # ── 1. Dependent variable ─────────────────────────────────────────────
        print("\n[1/5] Checking dependent variable assumptions...")
        try:
            y_t, dep_info = self._checker.check_and_transform(y, str(y.name or "y"))
        except ValueError as e:
            print(f"  WARNING: {e} — proceeding without transform")
            y_t, dep_info = y.copy().astype(float), TransformInfo()
        # diff / pct_change inverse transforms need the last training-period level
        if dep_info.method in ("diff", "pct_change"):
            dep_info.anchor = float(y.iloc[-(n_forecast + 1)])
        self.dep_transform_ = dep_info

        # ── 2. Exogenous candidates ───────────────────────────────────────────
        print("\n[2/5] Checking exogenous variable assumptions (stationarity, normality, linearity)...")
        valid: Dict[str, Tuple[pd.Series, TransformInfo]] = {}
        for col in X.columns:
            try:
                x_t, x_info = self._checker.check_and_transform(X[col], col)
                # Linearity check uses training slice only (no look-ahead)
                self._checker.check_linearity(
                    x_t.iloc[:-n_forecast], y_t.iloc[:-n_forecast], col
                )
                valid[col] = (x_t, x_info)
            except ValueError as e:
                self._log(f"{col}: DROPPED — {e}")

        X_t = (
            pd.DataFrame({k: v[0] for k, v in valid.items()})
            if valid
            else pd.DataFrame(index=X.index)
        )
        self.exog_transforms_ = {k: v[1] for k, v in valid.items()}

        # ── 3. Split train / test ─────────────────────────────────────────────
        y_train_t = y_t.iloc[:-n_forecast]

        # ── 4. Select exogenous (evaluated on training slice only) ────────────
        print("\n[3/5] Selecting exogenous variables...")
        if not X_t.empty:
            self.selected_exog_ = self._selector.select(
                y_train_t, X_t.iloc[:-n_forecast]
            )
        else:
            self.selected_exog_ = []

        exog_train = (
            X_t[self.selected_exog_].iloc[:-n_forecast].values
            if self.selected_exog_
            else None
        )
        exog_test = (
            X_t[self.selected_exog_].iloc[-n_forecast:].values
            if self.selected_exog_
            else None
        )

        # ── 5. Fit ARIMAX ─────────────────────────────────────────────────────
        print("\n[4/5] Fitting ARIMAX model (auto-selecting p, d, q via AIC)...")
        y_in = y_train_t.ffill().bfill().values

        try:
            model = pm.auto_arima(
                y_in,
                exogenous=exog_train,
                seasonal=False,
                stepwise=True,
                information_criterion="aic",
                max_p=5, max_q=5, max_d=2,
                error_action="ignore",
                suppress_warnings=True,
            )
        except Exception as e:
            self._log(f"ARIMAX failed ({e}); retrying as pure ARIMA")
            model = pm.auto_arima(
                y_in, seasonal=False, stepwise=True,
                max_p=5, max_q=5, max_d=2,
                error_action="ignore", suppress_warnings=True,
            )
            exog_test        = None
            self.selected_exog_ = []

        self.model_ = model
        self._log(f"Selected model: ARIMA{model.order}  |  AIC = {model.aic():.2f}")

        # ── 6. Residual diagnostics ───────────────────────────────────────────
        print("\n[5/5] Checking model residuals...")
        self._checker.check_residuals(model)

        # ── 7. Forecast + inverse-transform ───────────────────────────────────
        fc_t, ci_t = model.predict(
            n_periods=n_forecast,
            exogenous=exog_test,
            return_conf_int=True,
        )
        fc    = self._checker.inverse_transform(fc_t,        dep_info)
        ci_lo = self._checker.inverse_transform(ci_t[:, 0],  dep_info)
        ci_hi = self._checker.inverse_transform(ci_t[:, 1],  dep_info)
        actuals = y.values[-n_forecast:]

        # ── 8. Accuracy metrics ───────────────────────────────────────────────
        mae  = mean_absolute_error(actuals, fc)
        rmse = float(np.sqrt(mean_squared_error(actuals, fc)))
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = float(
                np.mean(
                    np.abs(np.where(actuals != 0, (actuals - fc) / actuals, 0.0))
                ) * 100
            )

        print(f"\n  Accuracy on held-out {n_forecast} period(s):")
        print(f"    MAE  : {mae:.4f}")
        print(f"    RMSE : {rmse:.4f}")
        print(f"    MAPE : {mape:.2f}%")

        idx = y.index[-n_forecast:]
        return ForecastResult(
            forecast      = pd.Series(fc,    index=idx, name="ARIMAX_forecast"),
            ci_lower      = pd.Series(ci_lo, index=idx, name="ci_lower"),
            ci_upper      = pd.Series(ci_hi, index=idx, name="ci_upper"),
            actuals       = pd.Series(actuals, index=idx, name="actuals"),
            train_actuals = y.iloc[:-n_forecast],
            order         = model.order,
            aic           = model.aic(),
            mae           = mae,
            rmse          = rmse,
            mape          = mape,
            selected_exog   = list(self.selected_exog_),
            exog_transforms = {
                k: v for k, v in self.exog_transforms_.items()
                if k in self.selected_exog_
            },
            dep_transform = dep_info,
            n_forecast    = n_forecast,
            model         = model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Optional ML comparison
# ─────────────────────────────────────────────────────────────────────────────

class MLForecaster:
    """
    Backtesting comparison using Random Forest or XGBoost.

    Approach: supervised learning on lag features. Because this is a
    backtesting scenario (we hold out the last N actuals), the lag
    features for the test period are constructed from true prior values —
    no recursive multi-step error accumulation.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}")

    def _build_feature_matrix(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame],
        n_lags: int,
    ) -> pd.DataFrame:
        df = pd.DataFrame({"y": y.values})
        for i in range(1, n_lags + 1):
            df[f"lag_{i}"] = df["y"].shift(i)
        if exog is not None and not exog.empty:
            for col in exog.columns:
                df[col] = exog[col].values
        return df.dropna().reset_index(drop=True)

    def fit_and_forecast(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame],
        n_forecast: int,
        method: str = "rf",
    ) -> MLResult:
        n_lags = max(1, min(5, len(y) - n_forecast - 2))
        feat_df = self._build_feature_matrix(y, exog, n_lags)

        exog_cols   = [c for c in feat_df.columns if c not in ["y"] and not c.startswith("lag_")]
        feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)] + exog_cols

        n_train = len(feat_df) - n_forecast
        if n_train <= 0:
            raise ValueError("Not enough data for ML forecaster after lag creation")

        X_train = feat_df.iloc[:n_train][feature_cols].fillna(0)
        y_train = feat_df.iloc[:n_train]["y"]
        X_test  = feat_df.iloc[n_train:][feature_cols].fillna(0)

        # ── choose estimator ──────────────────────────────────────────────────
        if method == "xgb":
            from xgboost import XGBRegressor   # raises ImportError if not installed
            clf   = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
            label = "XGBoost"
        else:
            from sklearn.ensemble import RandomForestRegressor
            clf   = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            label = "RandomForest"

        clf.fit(X_train, y_train)
        preds   = clf.predict(X_test)
        actuals = y.values[-n_forecast:]

        mae  = mean_absolute_error(actuals, preds)
        rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = float(
                np.mean(
                    np.abs(np.where(actuals != 0, (actuals - preds) / actuals, 0.0))
                ) * 100
            )

        self._log(
            f"{label}: MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%"
        )

        idx = y.index[-n_forecast:]
        return MLResult(
            method   = label,
            forecast = pd.Series(preds,   index=idx, name=f"{label}_forecast"),
            actuals  = pd.Series(actuals, index=idx, name="actuals"),
            mae      = mae,
            rmse     = rmse,
            mape     = mape,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model ranking utility
# ─────────────────────────────────────────────────────────────────────────────

def rank_models(
    arimax: ForecastResult,
    ml: List[MLResult],
) -> List[Dict]:
    """
    Combine ARIMAX and ML results and sort by RMSE ascending (best = rank 1).

    Each entry contains:
        rank, label, forecast, ci_lower*, ci_upper*, mae, rmse, mape, is_arimax
    (* ci_lower / ci_upper are None for ML models)
    """
    rows: List[Dict] = [
        {
            "label":    f"ARIMAX{arimax.order}",
            "forecast": arimax.forecast,
            "ci_lower": arimax.ci_lower,
            "ci_upper": arimax.ci_upper,
            "mae":      arimax.mae,
            "rmse":     arimax.rmse,
            "mape":     arimax.mape,
            "is_arimax": True,
        }
    ]
    for r in ml:
        rows.append(
            {
                "label":    r.method,
                "forecast": r.forecast,
                "ci_lower": None,
                "ci_upper": None,
                "mae":      r.mae,
                "rmse":     r.rmse,
                "mape":     r.mape,
                "is_arimax": False,
            }
        )
    rows.sort(key=lambda x: x["rmse"])
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows
