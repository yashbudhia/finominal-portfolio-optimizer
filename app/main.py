"""FastAPI entrypoint: POST /api/v1/optimize."""

from __future__ import annotations

import math

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import metrics as M
from .factors import factor_betas
from .models import (
    AllocationChange,
    OptimizeRequest,
    OptimizeResponse,
    PortfolioStats,
)
from .optimizer import (
    SUPPORTED_STRATEGIES,
    InfeasibleError,
    equal_weights,
    factor_exposure_weights,
    max_sharpe_weights,
    min_drawdown_weights,
    min_volatility_weights,
    normalize_strategy,
    risk_parity_weights,
    _bounds,
    _check_feasible,
    _portfolio_level_constraints,
)

app = FastAPI(
    title="Portfolio Optimizer API",
    version="1.0.0",
    description="Replicates the Finominal Portfolio Optimizer engine.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SECURITY_MASTER = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "AGG": "iShares Core US Aggregate Bond ETF",
    "GLD": "SPDR Gold Shares",
    "DPG": "Duff & Phelps Utility and Infrastructure Fund Inc",
}


def _round_weights(w: np.ndarray) -> list[float]:
    """Round to 2 decimals while forcing the sum to exactly 100.00."""
    pct = w * 100.0
    floored = np.floor(pct * 100) / 100.0
    remainder = round(100.0 - floored.sum(), 2)
    cents = int(round(remainder * 100))
    if cents != 0:
        frac = pct * 100 - np.floor(pct * 100)
        order = np.argsort(frac)
        step = 1 if cents > 0 else -1
        for k in range(abs(cents)):
            floored[order[-(k % len(order)) - 1]] += step * 0.01
    return [round(float(x), 2) for x in floored]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/strategies")
def strategies():
    return {"strategies": SUPPORTED_STRATEGIES}


@app.post("/api/v1/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    try:
        strategy = normalize_strategy(req.optimization_strategy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tickers = [s.ticker for s in req.securities]
    names = [s.security_name or SECURITY_MASTER.get(s.ticker, s.ticker) for s in req.securities]
    current_w = np.array([s.current_weight / 100.0 for s in req.securities], dtype=float)

    # Align all return histories to the trailing common window (funds often
    # have different inception dates). Truncating from the front keeps dates aligned.
    min_len = min(len(s.returns) for s in req.securities)
    R = np.column_stack([np.asarray(s.returns[-min_len:], dtype=float) for s in req.securities])
    if not np.isfinite(R).all():
        raise HTTPException(status_code=400, detail="returns must all be finite numbers")

    c = req.constraints
    try:
        # Normalize per-ticker keys (tickers are uppercased at input) and
        # reject unknown tickers instead of silently ignoring typos.
        min_w = {str(k).strip().upper(): v for k, v in c.min_weights.items()} if c and c.min_weights else None
        max_w = {str(k).strip().upper(): v for k, v in c.max_weights.items()} if c and c.max_weights else None
        for label, d in (("min_weights", min_w), ("max_weights", max_w)):
            if d:
                unknown = sorted(set(d) - set(tickers))
                if unknown:
                    raise HTTPException(
                        status_code=400,
                        detail=f"constraints.{label} has unknown tickers {unknown} "
                        f"(request tickers: {tickers})",
                    )
        bounds = _bounds(
            len(tickers),
            tickers,
            c.min_weight if c else None,
            c.max_weight if c else None,
            min_w,
            max_w,
        )
        div_yields = np.array(
            [
                (s.dividend_yield / 100.0 if s.dividend_yield is not None else np.nan)
                for s in req.securities
            ]
        )
        has_div = not bool(np.isnan(div_yields).any())
        extra = _portfolio_level_constraints(
            R,
            div_yields if has_div or (c and c.min_dividend_yield is not None) else None,
            req.periods_per_year,
            min_cagr=c.min_cagr if c else None,
            min_vol=c.min_volatility if c else None,
            max_vol=c.max_volatility if c else None,
            max_dd=c.max_drawdown if c else None,
            min_div_yield=c.min_dividend_yield if c else None,
        )
    except InfeasibleError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        if strategy == "equal_weights":
            w = equal_weights(len(tickers))
            # equal weights can still violate user bounds -> project then check
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            if bool(((w < lo - 1e-9) | (w > hi + 1e-9)).any()):
                raise InfeasibleError(
                    "equal weights violate min/max weight constraints "
                    "(equal weight infeasible under these bounds)"
                )
            _check_feasible(w, bounds, extra)
        elif strategy == "risk_parity":
            w = risk_parity_weights(R, bounds, extra)
        elif strategy == "minimize_volatility":
            w = min_volatility_weights(R, bounds, extra)
        elif strategy == "maximize_sharpe_ratio":
            w = max_sharpe_weights(
                R, bounds, extra, req.risk_free_rate, req.periods_per_year
            )
        elif strategy == "minimize_drawdown":
            w = min_drawdown_weights(R, bounds, extra)
        elif strategy == "optimize_factor_exposure":
            if req.factor_data is None:
                raise HTTPException(
                    status_code=400,
                    detail="optimize_factor_exposure requires factor_data "
                    "(momentum, value, size return series)",
                )
            obj = req.factor_objective
            w = factor_exposure_weights(
                R,
                bounds,
                extra,
                {
                    "momentum": np.asarray(req.factor_data.momentum),
                    "value": np.asarray(req.factor_data.value),
                    "size": np.asarray(req.factor_data.size),
                },
                target_factor=(obj.factor if obj else "momentum"),
                direction=(obj.direction if obj else "maximize"),
            )
        else:  # pragma: no cover - normalize_strategy guards this
            raise HTTPException(status_code=400, detail="unsupported strategy")
        _check_feasible(w, bounds, extra)
    except InfeasibleError as e:
        raise HTTPException(status_code=422, detail=str(e))

    w = w / w.sum()
    if bool((w < -1e-6).any()):
        raise HTTPException(status_code=422, detail="negative weights produced (no short selling)")

    rounded = _round_weights(w)
    changes = [
        AllocationChange(
            ticker=t,
            security_name=n,
            current_weight=round(float(cw), 2),
            optimized_weight=ow,
            change=round(float(ow - cw), 2),
        )
        for t, n, cw, ow in zip(tickers, names, [s.current_weight for s in req.securities], rounded)
    ]

    def stats(weights_frac: np.ndarray) -> PortfolioStats:
        series = R @ weights_frac
        sharpe = M.sharpe_ratio(series, req.risk_free_rate, req.periods_per_year)
        return PortfolioStats(
            cagr=round(float(M.cagr(series, req.periods_per_year)) * 100, 2),
            volatility=round(float(M.annualized_volatility(series, req.periods_per_year)) * 100, 2),
            max_drawdown=round(float(M.max_drawdown(series)) * 100, 2),
            sharpe_ratio=round(float(sharpe), 4) if math.isfinite(sharpe) else None,
            dividend_yield=round(float(np.dot(weights_frac, div_yields)) * 100, 2)
            if has_div
            else None,
        )

    factor_out = None
    if req.factor_data is not None:
        cur_series = R @ current_w
        opt_series = R @ w
        factor_out = {
            "current_portfolio": factor_betas(
                cur_series,
                np.asarray(req.factor_data.momentum),
                np.asarray(req.factor_data.value),
                np.asarray(req.factor_data.size),
            ),
            "optimized_portfolio": factor_betas(
                opt_series,
                np.asarray(req.factor_data.momentum),
                np.asarray(req.factor_data.value),
                np.asarray(req.factor_data.size),
            ),
        }

    return OptimizeResponse(
        optimization_strategy=strategy,
        allocation_changes=changes,
        current_portfolio=stats(current_w),
        optimized_portfolio=stats(w),
        factor_betas=factor_out,
    )
