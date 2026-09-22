"""Portfolio performance metrics operating on periodic return series.

All returns are decimal (e.g. 0.01 == 1% for the period).
Annualization assumes `periods_per_year` (default 12 == monthly data,
which is the Finominal Portfolio Optimizer convention).
"""

from __future__ import annotations

import numpy as np


def portfolio_series(returns_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted portfolio return series: R @ w. R is (T, N), w is (N,)."""
    return returns_matrix @ weights


def cumulative_wealth(series: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(series, dtype=float))


def cagr(series: np.ndarray, periods_per_year: int = 12) -> float:
    """Compound annual growth rate from a periodic return series."""
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        return 0.0
    wealth = np.prod(1.0 + s)
    if wealth <= 0:
        return -1.0
    return float(wealth ** (periods_per_year / s.size) - 1.0)


def annualized_volatility(series: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualized std-dev of periodic returns (sample std, ddof=1)."""
    s = np.asarray(series, dtype=float)
    if s.size < 2:
        return 0.0
    return float(np.std(s, ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(series: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (0.20 == 20% peak-to-trough).

    Computed on cumulative wealth including a starting value of 1.0.
    """
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        return 0.0
    cum = np.concatenate([[1.0], cumulative_wealth(s)])
    running_max = np.maximum.accumulate(cum)
    # avoid div-by-zero (should not happen since wealth starts at 1.0)
    dd = (cum - running_max) / np.where(running_max == 0, 1.0, running_max)
    return float(-np.min(dd))


def sharpe_ratio(
    series: np.ndarray,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """Annualized Sharpe ratio. risk_free_annual is decimal (0.02 == 2%)."""
    s = np.asarray(series, dtype=float)
    if s.size < 2:
        return 0.0
    rf_per_period = risk_free_annual / periods_per_year
    excess = s - rf_per_period
    vol = float(np.std(s, ddof=1))
    if vol == 0:
        mean_excess = float(np.mean(excess))
        if mean_excess > 0:
            return float("inf")
        if mean_excess < 0:
            return float("-inf")
        return 0.0
    return float(np.mean(excess) / vol * np.sqrt(periods_per_year))


def covariance_matrix(returns_matrix: np.ndarray) -> np.ndarray:
    """Sample covariance (N x N) of the (T, N) return matrix."""
    r = np.asarray(returns_matrix, dtype=float)
    if r.shape[0] < 2:
        n = r.shape[1]
        return np.eye(n) * 1e-8
    cov = np.cov(r, rowvar=False)
    cov = np.atleast_2d(cov)
    # Regularize slightly so SLSQP / risk-parity never sees a singular matrix.
    n = cov.shape[0]
    cov = cov + np.eye(n) * 1e-10
    return cov
