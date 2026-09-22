"""Build the 6 required test-scenario request bodies from sample_history.json.

Run:  python sample_requests/build_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

HIST = json.loads(Path(__file__).resolve().parent.parent.joinpath("data", "sample_history.json").read_text())
BY_TICKER = {s["ticker"]: s for s in HIST["securities"]}


def sec(ticker: str, w: float) -> dict:
    s = BY_TICKER[ticker]
    return {
        "ticker": ticker,
        "security_name": s["security_name"],
        "current_weight": w,
        "returns": s["returns"],
        "dividend_yield": s["dividend_yield"],
    }


def base(secs: list[dict], strategy: str, **kw) -> dict:
    body: dict = {
        "securities": secs,
        "optimization_strategy": strategy,
        "periods_per_year": 12,
        "risk_free_rate": 0.0,
    }
    body.update(kw)
    return body


CASES = {
    # 1: equal-weight sanity check
    "case1_equal_weights": base(
        [sec("IEFA", 25), sec("SPY", 75)], "equal_weights"
    ),
    # 2: risk parity across asset classes
    "case2_risk_parity": base(
        [sec("VEA", 25), sec("AGG", 75)], "risk_parity"
    ),
    # 3: lowest-risk portfolio
    "case3_min_volatility": base(
        [sec("SPY", 60), sec("AGG", 30), sec("GLD", 10)],
        "minimize_volatility",
    ),
    # 3b: minimize drawdown (same inputs, required strategy #3)
    "case3b_min_drawdown": base(
        [sec("SPY", 60), sec("AGG", 30), sec("GLD", 10)],
        "minimize_drawdown",
    ),
    # 4: risk-adjusted return, no constraints
    "case4_max_sharpe": base(
        [sec("IEFA", 20), sec("GLD", 20), sec("AGG", 20), sec("VEA", 20), sec("SPY", 20)],
        "maximize_sharpe_ratio",
    ),
    # 5: max sharpe + portfolio + security constraints
    "case5_max_sharpe_constrained": base(
        [sec("IEFA", 20), sec("GLD", 20), sec("AGG", 20), sec("VEA", 20), sec("SPY", 20)],
        "maximize_sharpe_ratio",
        constraints={
            "min_weight": 5,
            "max_weight": 40,
            "min_dividend_yield": 2.50,
        },
    ),
    # 6 (bonus): maximize momentum exposure + betas
    "case6_factor_momentum": base(
        [sec("IEFA", 20), sec("GLD", 20), sec("AGG", 20), sec("VEA", 20), sec("SPY", 20)],
        "optimize_factor_exposure",
        factor_data=HIST["factor_data"],
        factor_objective={"factor": "momentum", "direction": "maximize"},
    ),
}


def main() -> None:
    outdir = Path(__file__).resolve().parent
    for name, body in CASES.items():
        (outdir / f"{name}.json").write_text(json.dumps(body, indent=2))
        print(f"wrote {name}.json")


if __name__ == "__main__":
    main()
