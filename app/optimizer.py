"""Optimization strategies.

Strategies:
  1. equal_weights          - 100/N each (baseline, no solver needed)
  2. risk_parity            - equalize risk contributions (SLSQP)
  3. minimize_drawdown      - minimize max historical drawdown (multi-start SLSQP)
  4. minimize_volatility    - minimum-variance portfolio (SLSQP)
  5. maximize_sharpe_ratio  - max risk-adjusted return (multi-start SLSQP)
  6. optimize_factor_exposure (bonus) - max/min a factor beta (linear objective)

All weight-vector math is in fractions (0-1). Percent conversion happens
at the API boundary.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import metrics as M

STRATEGY_ALIASES = {
    "equal_weights": "equal_weights",
    "equal_weight": "equal_weights",
    "equal": "equal_weights",
    "risk_parity": "risk_parity",
    "riskparity": "risk_parity",
    "minimize_drawdown": "minimize_drawdown",
    "min_drawdown": "minimize_drawdown",
    "min_dd": "minimize_drawdown",
    "minimize_volatility": "minimize_volatility",
    "min_volatility": "minimize_volatility",
    "min_vol": "minimize_volatility",
    "minimum_volatility": "minimize_volatility",
    "maximize_sharpe_ratio": "maximize_sharpe_ratio",
    "max_sharpe": "maximize_sharpe_ratio",
    "maximize_sharpe": "maximize_sharpe_ratio",
    "sharpe": "maximize_sharpe_ratio",
    "optimize_factor_exposure": "optimize_factor_exposure",
    "factor_exposure": "optimize_factor_exposure",
    "max_factor": "optimize_factor_exposure",
}

SUPPORTED_STRATEGIES = sorted(set(STRATEGY_ALIASES.values()))


class InfeasibleError(ValueError):
    """Raised when constraints cannot be satisfied."""


def normalize_strategy(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in STRATEGY_ALIASES:
        raise ValueError(
            f"unsupported optimization_strategy '{name}'. "
            f"Supported: {SUPPORTED_STRATEGIES}"
        )
    return STRATEGY_ALIASES[key]


def _norm_weight_dict(d: dict | None) -> dict | None:
    """Normalize per-ticker weight dict keys to uppercase tickers."""
    if not d:
        return None
    return {str(k).strip().upper(): v for k, v in d.items()}


def _bounds(
    n: int,
    tickers: list[str],
    min_weight: float | None,
    max_weight: float | None,
    min_weights: dict | None,
    max_weights: dict | None,
) -> list[tuple[float, float]]:
    """Build per-asset (lo, hi) bounds in fractions."""
    min_weights = _norm_weight_dict(min_weights)
    max_weights = _norm_weight_dict(max_weights)
    bounds = []
    for i, t in enumerate(tickers):
        lo = 0.0 if min_weight is None else min_weight / 100.0
        hi = 1.0 if max_weight is None else max_weight / 100.0
        if min_weights and t in min_weights:
            lo = min_weights[t] / 100.0
        if max_weights and t in max_weights:
            hi = max_weights[t] / 100.0
        if lo < 0 or hi > 1 or lo > hi:
            raise InfeasibleError(
                f"infeasible weight bounds for {t}: min={lo*100:.2f}%, max={hi*100:.2f}%"
            )
        bounds.append((lo, hi))
    lo_sum = sum(b[0] for b in bounds)
    hi_sum = sum(b[1] for b in bounds)
    if lo_sum > 1.0 + 1e-9 or hi_sum < 1.0 - 1e-9:
        raise InfeasibleError(
            f"infeasible weight bounds: min weights sum to {lo_sum*100:.2f}%, "
            f"max weights sum to {hi_sum*100:.2f}% (must bracket 100%)"
        )
    return bounds


def _portfolio_level_constraints(
    returns_matrix: np.ndarray,
    div_yields: np.ndarray | None,
    periods_per_year: int,
    min_cagr: float | None = None,
    min_vol: float | None = None,
    max_vol: float | None = None,
    max_dd: float | None = None,
    min_div_yield: float | None = None,
) -> list[dict]:
    """SLSQP inequality constraints (fun(w) >= 0 means feasible)."""
    cons: list[dict] = []
    if min_div_yield is not None:
        if div_yields is None or bool(np.isnan(div_yields).any()):
            raise InfeasibleError(
                "min_dividend_yield constraint requires dividend_yield for every security"
            )
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w, _y=np.asarray(div_yields): float(
                    np.dot(w, _y) - min_div_yield / 100.0
                ),
            }
        )
    if min_cagr is not None:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w, _r=returns_matrix, _p=periods_per_year: float(
                    M.cagr(_r @ w, _p) - min_cagr / 100.0
                ),
            }
        )
    if min_vol is not None:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w, _r=returns_matrix, _p=periods_per_year: float(
                    M.annualized_volatility(_r @ w, _p) - min_vol / 100.0
                ),
            }
        )
    if max_vol is not None:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w, _r=returns_matrix, _p=periods_per_year: float(
                    max_vol / 100.0 - M.annualized_volatility(_r @ w, _p)
                ),
            }
        )
    if max_dd is not None:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w, _r=returns_matrix: float(
                    max_dd / 100.0 - M.max_drawdown(_r @ w)
                ),
            }
        )
    return cons


def _check_feasible(w: np.ndarray, bounds, extra_cons, tol: float = 1e-6) -> None:
    if abs(w.sum() - 1.0) > 1e-4:
        raise InfeasibleError(f"solver returned weights summing to {w.sum()*100:.2f}%")
    for (lo, hi), wi, i in zip(bounds, w, range(len(w))):
        if wi < lo - 1e-4 or wi > hi + 1e-4:
            raise InfeasibleError(f"weight {i}={wi*100:.2f}% violates bounds")
    for c in extra_cons:
        if float(c["fun"](w)) < -1e-4:
            raise InfeasibleError(
                "portfolio-level constraint infeasible (e.g. min CAGR / vol range / "
                "max drawdown / min dividend yield cannot be met)"
            )


def _solve(objective, x0_list, bounds, constraints, method="SLSQP"):
    best = None
    best_val = float("inf")
    for x0 in x0_list:
        try:
            res = minimize(
                objective,
                np.asarray(x0, dtype=float),
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
        except Exception:
            continue
        if not res.success:
            continue
        w = np.clip(res.x, 0, 1)
        if abs(w.sum() - 1) > 1e-6:
            w = w / w.sum()
        # verify SLSQP inequality constraints manually (tolerance)
        ok = all(float(c["fun"](w)) >= -1e-5 for c in constraints if c["type"] == "ineq")
        if not ok:
            continue
        if float(res.fun) < best_val:
            best_val = float(res.fun)
            best = w
    if best is None:
        raise InfeasibleError(
            "optimization failed: constraints may be infeasible - "
            "try relaxing min/max weights or portfolio-level constraints"
        )
    return best / best.sum()


# ---------------------------------------------------------------- strategies

def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def risk_parity_weights(
    returns_matrix: np.ndarray,
    bounds,
    extra_cons,
) -> np.ndarray:
    cov = M.covariance_matrix(returns_matrix)
    n = returns_matrix.shape[1]

    def objective(w):
        w = np.asarray(w)
        port_var = float(w @ cov @ w)
        if port_var <= 1e-14:
            return float(np.sum((w - 1.0 / n) ** 2))
        rc = w * (cov @ w) / port_var  # risk contributions sum to 1
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}] + list(extra_cons)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0s = [np.full(n, 1.0 / n)]
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = lo + rng.random(n) * (hi - lo)
        x0s.append(x / x.sum())
    return _solve(objective, x0s, bounds, cons)


def min_volatility_weights(returns_matrix, bounds, extra_cons) -> np.ndarray:
    cov = M.covariance_matrix(returns_matrix)
    n = returns_matrix.shape[1]

    def objective(w):
        return float(np.sqrt(max(float(w @ cov @ w), 0.0)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}] + list(extra_cons)
    # start from equal + single-asset corners (helps when min-vol is 100% bonds)
    x0s = [np.full(n, 1.0 / n)]
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1.0
        # project corner onto bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x = np.clip(x, lo, hi)
        x = x / x.sum()
        x0s.append(x)
    return _solve(objective, x0s, bounds, cons)


def max_sharpe_weights(
    returns_matrix, bounds, extra_cons, risk_free_annual=0.0, periods_per_year=12
) -> np.ndarray:
    n = returns_matrix.shape[1]

    def objective(w):
        s = float(
            M.sharpe_ratio(returns_matrix @ w, risk_free_annual, periods_per_year)
        )
        if np.isposinf(s):
            return -1e6
        if np.isneginf(s) or np.isnan(s):
            return 1e6
        return -s

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}] + list(extra_cons)
    x0s = [np.full(n, 1.0 / n)]
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.random(n)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x = lo + (x * (hi - lo))
        x = x / x.sum()
        x0s.append(x)
    return _solve(objective, x0s, bounds, cons)


def min_drawdown_weights(returns_matrix, bounds, extra_cons) -> np.ndarray:
    n = returns_matrix.shape[1]

    def objective(w):
        return float(M.max_drawdown(returns_matrix @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}] + list(extra_cons)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0s = [np.full(n, 1.0 / n)]
    try:
        x0s.append(min_volatility_weights(returns_matrix, bounds, []))
    except Exception:
        pass
    rng = np.random.default_rng(2)
    for _ in range(5):
        x = lo + rng.random(n) * (hi - lo)
        x0s.append(x / x.sum())
    return _solve(objective, x0s, bounds, cons)


def factor_exposure_weights(
    returns_matrix,
    bounds,
    extra_cons,
    factor_data: dict[str, np.ndarray],
    target_factor: str = "momentum",
    direction: str = "maximize",
) -> np.ndarray:
    from .factors import factor_loading_matrix

    C, names = factor_loading_matrix(
        returns_matrix,
        np.asarray(factor_data["momentum"]),
        np.asarray(factor_data["value"]),
        np.asarray(factor_data["size"]),
    )
    idx = names.index(target_factor)
    c = C[idx, :]
    sign = -1.0 if direction == "maximize" else 1.0

    def objective(w):
        return float(sign * np.dot(c, w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}] + list(extra_cons)
    n = returns_matrix.shape[1]
    x0s = [np.full(n, 1.0 / n)]
    # corner leaning toward highest loading often optimal for linear objective
    order = np.argsort(c if direction == "maximize" else -c)[::-1]
    x = np.array([b[0] for b in bounds], dtype=float)
    remaining = 1.0 - x.sum()
    for i in order:
        hi = bounds[i][1]
        add = min(remaining, hi - x[i])
        if add > 0:
            x[i] += add
            remaining -= add
    x0s.append(x / x.sum())
    return _solve(objective, x0s, bounds, cons)
