"""Factor beta estimation via OLS.

Model:  portfolio_return = alpha + b_mom * Momentum + b_val * Value + b_size * Size + e

Uses the common (trailing) window between the portfolio series and the
factor series so mismatched history lengths still work.
"""

from __future__ import annotations

import numpy as np


def _common_window(*series: np.ndarray) -> list[np.ndarray]:
    n = min(len(s) for s in series)
    return [np.asarray(s, dtype=float)[-n:] for s in series]


def factor_betas(
    portfolio_returns: np.ndarray,
    momentum: np.ndarray,
    value: np.ndarray,
    size: np.ndarray,
) -> dict[str, float]:
    """OLS betas (with intercept) of portfolio returns on 3 factors."""
    rp, mom, val, siz = _common_window(portfolio_returns, momentum, value, size)
    t = len(rp)
    if t < 5:  # fewer obs than params+1 -> fall back to covariances
        out: dict[str, float] = {}
        for name, f in (("momentum", mom), ("value", val), ("size", siz)):
            vf = float(np.var(f))
            out[name] = float(np.cov(rp, f, ddof=1)[0, 1] / vf) if vf > 0 else 0.0
        return out
    X = np.column_stack([np.ones(t), mom, val, siz])  # (T, 4)
    coef, *_ = np.linalg.lstsq(X, rp, rcond=None)
    return {
        "momentum": float(coef[1]),
        "value": float(coef[2]),
        "size": float(coef[3]),
    }


def factor_loading_matrix(
    returns_matrix: np.ndarray,
    momentum: np.ndarray,
    value: np.ndarray,
    size: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Precompute C (3 x N) with beta_vector(w) = C @ w.

    Because portfolio returns are linear in w (r_p = R w) and OLS betas are
    linear in r_p, the factor beta for any weight vector is a linear function
    of w. We exploit this so factor-exposure optimization is exact and fast.

    Returns (C, ["momentum", "value", "size"]) with rows in that order.
    """
    r = np.asarray(returns_matrix, dtype=float)
    mom = np.asarray(momentum, dtype=float)
    val = np.asarray(value, dtype=float)
    siz = np.asarray(size, dtype=float)
    # Common trailing window across funds AND factors (funds may have
    # shorter histories than the factor series or vice versa).
    t_common = min(r.shape[0], len(mom), len(val), len(siz))
    mom, val, siz = mom[-t_common:], val[-t_common:], siz[-t_common:]
    r_common = r[-t_common:, :]
    F = np.column_stack([mom, val, siz])  # (T, 3)
    X = np.column_stack([np.ones(t_common), F])  # (T, 4)
    # M maps a return series -> [alpha, b_mom, b_val, b_size].
    M, *_ = np.linalg.lstsq(X, np.eye(t_common), rcond=None)  # (4, T)
    M_betas = M[1:, :]  # (3, T)
    C = M_betas @ r_common  # (3, N), rows = mom/val/size
    return C, ["momentum", "value", "size"]
