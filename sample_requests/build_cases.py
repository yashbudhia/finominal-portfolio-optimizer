"""Build the 6 required test-scenario request bodies from the real Data.xlsx.

Funds have different inception dates and the factor calendar differs from the
fund calendar, so each scenario is aligned on the intersection of the dates
actually present for the tickers (and factors) it uses.

Run:  python data/load_excel.py && python sample_requests/build_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIST = json.loads((ROOT / "data" / "real_history.json").read_text())
FUNDS = HIST["funds"]
FACTORS = HIST["factors"]
PPY = HIST["periods_per_year"]


def _aligned(tickers: list[str], with_factors: bool = False):
    """Common trading dates across the given funds (and factors if asked)."""
    common = set(FUNDS[tickers[0]]["dates"])
    for t in tickers[1:]:
        common &= set(FUNDS[t]["dates"])
    if with_factors:
        common &= set(FACTORS["dates"])
    dates = sorted(common)
    idx = {t: {d: i for i, d in enumerate(FUNDS[t]["dates"])} for t in tickers}
    series = {t: [FUNDS[t]["returns"][idx[t][d]] for d in dates] for t in tickers}
    factors = None
    if with_factors:
        fidx = {d: i for i, d in enumerate(FACTORS["dates"])}
        factors = {
            k: [FACTORS[k][fidx[d]] for d in dates] for k in ("momentum", "value", "size")
        }
    return dates, series, factors


def base(holdings: dict[str, float], strategy: str, with_factors: bool = False, **kw) -> dict:
    tickers = list(holdings)
    dates, series, factors = _aligned(tickers, with_factors)
    body: dict = {
        "securities": [
            {
                "ticker": t,
                "security_name": FUNDS[t]["security_name"],
                "current_weight": holdings[t],
                "returns": series[t],
                "dividend_yield": FUNDS[t]["dividend_yield"],
            }
            for t in tickers
        ],
        "optimization_strategy": strategy,
        "periods_per_year": PPY,
        "risk_free_rate": 0.0,
        "_history": {"start": dates[0], "end": dates[-1], "observations": len(dates)},
    }
    if factors is not None:
        body["factor_data"] = factors
    body.update(kw)
    return body


ALL_FIVE = {"IEFA": 20, "GLD": 20, "AGG": 20, "VEA": 20, "SPY": 20}

CASES = {
    # 1: equal-weight sanity check
    "case1_equal_weights": base({"IEFA": 25, "SPY": 75}, "equal_weights"),
    # 2: risk parity across asset classes
    "case2_risk_parity": base({"VEA": 25, "AGG": 75}, "risk_parity"),
    # 3: lowest-risk portfolio
    "case3_min_volatility": base(
        {"SPY": 60, "AGG": 30, "GLD": 10}, "minimize_volatility"
    ),
    # 3b: minimize drawdown (same inputs, required strategy #3)
    "case3b_min_drawdown": base(
        {"SPY": 60, "AGG": 30, "GLD": 10}, "minimize_drawdown"
    ),
    # 4: risk-adjusted return, no constraints
    "case4_max_sharpe": base(ALL_FIVE, "maximize_sharpe_ratio"),
    # 5: max sharpe + portfolio + security constraints
    "case5_max_sharpe_constrained": base(
        ALL_FIVE,
        "maximize_sharpe_ratio",
        constraints={"min_weight": 5, "max_weight": 40, "min_dividend_yield": 2.50},
    ),
    # 6 (bonus): maximize momentum exposure + betas
    "case6_factor_momentum": base(
        ALL_FIVE,
        "optimize_factor_exposure",
        with_factors=True,
        factor_objective={"factor": "momentum", "direction": "maximize"},
    ),
}


def main() -> None:
    outdir = Path(__file__).resolve().parent
    for name, body in CASES.items():
        (outdir / f"{name}.json").write_text(json.dumps(body, indent=2))
        h = body["_history"]
        print(f"wrote {name}.json  {h['observations']:5d} obs  {h['start']} -> {h['end']}")


if __name__ == "__main__":
    main()
