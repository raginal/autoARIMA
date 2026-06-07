"""
evaluation.py
Model selection, accuracy metrics, ensembling and forecast guardrails.

Why this module exists
──────────────────────
The most recent periods are scored against lag-corrupted "actuals", so they are
NOT a trustworthy basis for choosing a model. Instead every model is ranked by a
**rolling-origin backtest over the settled (fully-reported) history**, and the
forecast for the unreliable tail is produced separately as the deliverable.

Public surface
──────────────
- metrics: ``rmse``, ``mae``, ``mape``, ``smape``, ``mase``
- ``rolling_backtest()`` — out-of-sample evaluation of one model over settled data
- ``ModelEval`` / ``VariableResult`` — structures consumed by ``exports.py``
- ``evaluate_variable()`` — run the full registry for one dependent variable
- ``build_ensemble()`` — inverse-error weighted combination of the strongest models
- ``flag_implausible()`` — guardrail flags for unreasonable forecast moves
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecaster import BaseForecaster, Forecast, VariableSelector, build_models


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse(a: np.ndarray, f: np.ndarray) -> float:
    a, f = _finite_pair(a, f)
    return float(np.sqrt(np.mean((a - f) ** 2))) if len(a) else float("nan")


def mae(a: np.ndarray, f: np.ndarray) -> float:
    a, f = _finite_pair(a, f)
    return float(np.mean(np.abs(a - f))) if len(a) else float("nan")


def mape(a: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute percentage error; periods with actual==0 are excluded
    (returning 0 there would silently mask error)."""
    a, f = _finite_pair(a, f)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100)


def smape(a: np.ndarray, f: np.ndarray) -> float:
    a, f = _finite_pair(a, f)
    denom = np.abs(a) + np.abs(f)
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(a[mask] - f[mask]) / denom[mask]) * 100)


def mase(a: np.ndarray, f: np.ndarray, y_train: np.ndarray, period: int = 1) -> float:
    """Mean Absolute Scaled Error: MAE scaled by the in-sample seasonal-naive MAE.
    <1 ⇒ better than naive, >1 ⇒ worse. Scale-free, so it is comparable across
    dependent variables of different magnitudes."""
    a, f = _finite_pair(a, f)
    yt = np.asarray(y_train, dtype=float)
    yt = yt[np.isfinite(yt)]
    m = period if (period >= 1 and len(yt) > period) else 1
    scale = np.mean(np.abs(yt[m:] - yt[:-m])) if len(yt) > m else np.nan
    if not np.isfinite(scale) or scale == 0:
        scale = np.mean(np.abs(np.diff(yt))) if len(yt) > 1 else np.nan
    if not np.isfinite(scale) or scale == 0 or len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - f)) / scale)


def _finite_pair(a, f) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    f = np.asarray(f, dtype=float)
    mask = np.isfinite(a) & np.isfinite(f)
    return a[mask], f[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Result structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelEval:
    name: str
    final: Forecast                          # forecast for the (unreliable) tail
    metrics: Dict[str, float]                # backtest metrics (the ranking basis)
    pooled_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    pooled_actual: np.ndarray = field(default_factory=lambda: np.array([]))
    flags: List[str] = field(default_factory=list)
    rank: int = 0
    meta: Dict = field(default_factory=dict)


@dataclass
class VariableResult:
    dep_col: str
    time_col: str
    period: int
    tail_index: pd.Index                     # the forecast periods (deliverable)
    tail_actuals: pd.Series                  # reported but unreliable — display only
    train_actuals: pd.Series                 # settled history (for charts)
    models: List[ModelEval]                  # ranked, best first
    selected_exog: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-origin backtest
# ─────────────────────────────────────────────────────────────────────────────

def _fold_cutoffs(n: int, h: int, n_folds: int, period: int) -> List[int]:
    """Deterministic expanding-window cutoffs near the end of the settled data.
    Deterministic in (n, h, n_folds, period) so every model is scored on the
    identical folds — a prerequisite for pooling them into an ensemble."""
    min_train = max(10, 2 * period, h + 2)
    cutoffs = []
    for i in range(n_folds):
        c = n - h - i
        if c < min_train:
            break
        cutoffs.append(c)
    return sorted(cutoffs)


def rolling_backtest(
    model: BaseForecaster,
    y_settled: pd.Series,
    X_settled: Optional[pd.DataFrame],
    h: int,
    n_folds: int,
    period: int,
) -> Dict[str, object]:
    """
    Evaluate one model out-of-sample over the settled history. Returns pooled
    metrics plus the pooled per-point predictions/actuals (NaN where a fold failed)
    so the same folds can be reused to score an ensemble.
    """
    y_settled = pd.Series(y_settled).astype(float)
    n = len(y_settled)
    cutoffs = _fold_cutoffs(n, h, n_folds, period)

    pooled_pred: List[float] = []
    pooled_actual: List[float] = []
    for c in cutoffs:
        y_tr = y_settled.iloc[:c]
        actual = y_settled.iloc[c:c + h].values
        if model.uses_exog and X_settled is not None and not X_settled.empty:
            X_tr = X_settled.iloc[:c]
            X_fu = X_settled.iloc[c:c + h]
        else:
            X_tr = X_fu = None
        try:
            fc = model.fit_predict(y_tr, X_tr, X_fu, h)
            pred = np.asarray(fc.point, dtype=float)
        except Exception:
            pred = np.full(h, np.nan)
        if len(pred) < h:
            pred = np.concatenate([pred, np.full(h - len(pred), np.nan)])
        pooled_pred.extend(pred[:h].tolist())
        pooled_actual.extend(actual.tolist())

    pp = np.asarray(pooled_pred, dtype=float)
    pa = np.asarray(pooled_actual, dtype=float)
    return {
        "metrics": {
            "mase":  mase(pa, pp, y_settled.values, period),
            "smape": smape(pa, pp),
            "rmse":  rmse(pa, pp),
            "mae":   mae(pa, pp),
            "mape":  mape(pa, pp),
        },
        "pooled_pred": pp,
        "pooled_actual": pa,
        "n_folds": len(cutoffs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def build_ensemble(
    members: List[ModelEval],
    y_train_scale: np.ndarray,
    period: int,
    top_k: int = 3,
) -> Optional[ModelEval]:
    """
    Inverse-MASE weighted combination of the strongest non-naive models. The
    ensemble is scored on the *same* pooled backtest folds as its members, so its
    metric is directly comparable. Combinations are consistently more robust than
    any single model.
    """
    elig = [
        m for m in members
        if m.name not in ("Naive", "SeasonalNaive")
        and np.isfinite(m.metrics.get("mase", np.nan))
        and m.pooled_pred.size
    ]
    if len(elig) < 2:
        return None

    elig.sort(key=lambda m: m.metrics["mase"])
    chosen = elig[:max(2, min(top_k, len(elig)))]
    weights = np.array([1.0 / max(m.metrics["mase"], 1e-6) for m in chosen])
    weights /= weights.sum()

    # pooled backtest prediction (folds are aligned across models)
    L = min(m.pooled_pred.size for m in chosen)
    stack = np.vstack([m.pooled_pred[:L] for m in chosen])
    wcol = weights.reshape(-1, 1)
    pooled = np.nansum(stack * wcol, axis=0) / np.nansum(np.where(np.isfinite(stack), wcol, 0), axis=0)
    pa = chosen[0].pooled_actual[:L]

    # deliverable forecast
    hlen = min(m.final.point.size for m in chosen)
    fstack = np.vstack([m.final.point[:hlen] for m in chosen])
    final_point = np.nansum(fstack * wcol, axis=0) / np.nansum(
        np.where(np.isfinite(fstack), wcol, 0), axis=0)

    return ModelEval(
        name="Ensemble",
        final=Forecast(point=final_point,
                       meta={"members": [m.name for m in chosen],
                             "weights": dict(zip([m.name for m in chosen], np.round(weights, 3)))}),
        metrics={
            "mase":  mase(pa, pooled, y_train_scale, period),
            "smape": smape(pa, pooled),
            "rmse":  rmse(pa, pooled),
            "mae":   mae(pa, pooled),
            "mape":  mape(pa, pooled),
        },
        pooled_pred=pooled,
        pooled_actual=pa,
        meta={"members": [m.name for m in chosen]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────────────────────────────────

def flag_implausible(
    point: np.ndarray,
    y_history: np.ndarray,
    k: float = 3.0,
    band: float = 0.5,
) -> List[str]:
    """
    Return a per-step reason string ('' = fine) flagging forecast moves that look
    unreasonable: a step jump larger than ``k`` × the largest historical one-step
    move, or a level leaving the historical range by more than ``band`` × its span.
    Values are never modified — flags are advisory.
    """
    point = np.asarray(point, dtype=float)
    hist = np.asarray(y_history, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) < 3:
        return [""] * len(point)

    last = hist[-1]
    steps = np.abs(np.diff(hist))
    hist_step = np.nanmax(steps) if steps.size else np.nanstd(hist)
    if not np.isfinite(hist_step) or hist_step == 0:
        hist_step = np.nanstd(hist) or 1.0
    lo, hi = np.nanmin(hist), np.nanmax(hist)
    span = (hi - lo) or abs(hi) or 1.0

    reasons: List[str] = []
    prev = last
    for v in point:
        msgs = []
        if np.isfinite(v):
            if abs(v - prev) > k * hist_step:
                msgs.append(f"jump {v - prev:+.2f} > {k}×hist step ({hist_step:.2f})")
            if v < lo - band * span or v > hi + band * span:
                msgs.append(f"outside historical range [{lo:.2f}, {hi:.2f}]")
            prev = v
        reasons.append("; ".join(msgs))
    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_models(models: List[ModelEval]) -> List[ModelEval]:
    """Sort by backtest MASE ascending (NaN last); assign 1-based ranks."""
    def key(m: ModelEval):
        v = m.metrics.get("mase", np.nan)
        return (np.isnan(v), v if np.isfinite(v) else np.inf)
    ordered = sorted(models, key=key)
    for i, m in enumerate(ordered):
        m.rank = i + 1
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration for one dependent variable
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variable(
    y: pd.Series,
    X: pd.DataFrame,
    dep_col: str,
    time_col: str,
    n_forecast: int,
    period: int,
    n_folds: int = 3,
    verbose: bool = True,
) -> VariableResult:
    """
    Full pipeline for one dependent variable:
      1. split off the unreliable tail (the deliverable horizon),
      2. select exogenous regressors once on the settled training data,
      3. for each model: final deliverable forecast + rolling backtest,
      4. add the inverse-error ensemble, rank by backtest MASE, flag implausible moves.
    """
    y = pd.Series(y).astype(float)
    h = n_forecast
    settled_y = y.iloc[:-h]
    settled_X = X.iloc[:-h] if X is not None and not X.empty else pd.DataFrame(index=settled_y.index)
    future_X = X.iloc[-h:] if X is not None and not X.empty else None
    tail_index = y.index[-h:]
    tail_actuals = y.iloc[-h:]

    # ── exogenous selection (once, on settled training only) ───────────────────
    selected: List[str] = []
    if not settled_X.empty:
        selector = VariableSelector(verbose=verbose)
        selected = selector.select(settled_y, settled_X, X_future=future_X)
    X_sel_settled = settled_X[selected] if selected else None
    X_sel_future = future_X[selected] if (selected and future_X is not None) else None

    evals: List[ModelEval] = []
    for model in build_models(period=period, verbose=verbose):
        if verbose:
            print(f"\n  → {model.name}")
        X_tr = X_sel_settled if model.uses_exog else None
        X_fu = X_sel_future if model.uses_exog else None

        # final deliverable forecast first (ARIMAX caches its order here)
        try:
            final = model.fit_predict(settled_y, X_tr, X_fu, h)
        except ImportError:
            if verbose:
                print(f"    {model.name} unavailable — skipped")
            continue
        except Exception as e:
            if verbose:
                print(f"    {model.name} failed: {e}")
            continue

        bt = rolling_backtest(model, settled_y, X_tr, h, n_folds, period)
        flags = flag_implausible(final.point, settled_y.values)
        if verbose:
            mt = bt["metrics"]
            print(f"    backtest MASE={mt['mase']:.3f}  sMAPE={mt['smape']:.2f}%  RMSE={mt['rmse']:.3f}")
        evals.append(ModelEval(
            name=model.name, final=final, metrics=bt["metrics"],
            pooled_pred=bt["pooled_pred"], pooled_actual=bt["pooled_actual"],
            flags=flags, meta=final.meta,
        ))

    # ── ensemble ────────────────────────────────────────────────────────────────
    ens = build_ensemble(evals, settled_y.values, period)
    if ens is not None:
        ens.flags = flag_implausible(ens.final.point, settled_y.values)
        evals.append(ens)

    ranked = rank_models(evals)
    return VariableResult(
        dep_col=dep_col, time_col=time_col, period=period,
        tail_index=tail_index, tail_actuals=tail_actuals,
        train_actuals=settled_y, models=ranked, selected_exog=selected,
    )
