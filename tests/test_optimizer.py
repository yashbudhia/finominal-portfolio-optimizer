"""Acceptance tests: sums, bounds, constraints, factor betas, error paths."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
REQ = Path(__file__).resolve().parent.parent / "sample_requests"


def _post(name: str):
    body = json.loads((REQ / name).read_text())
    r = client.post("/api/v1/optimize", json=body)
    assert r.status_code == 200, r.text
    return r.json(), body


def _assert_valid(j):
    weights = [a["optimized_weight"] for a in j["allocation_changes"]]
    assert abs(sum(weights) - 100.0) < 1e-6
    assert all(w >= -1e-9 for w in weights)
    for a in j["allocation_changes"]:
        assert {"ticker", "security_name", "current_weight", "optimized_weight", "change"} <= set(a)
        assert abs(a["change"] - (a["optimized_weight"] - a["current_weight"])) < 0.02


def test_case1_equal_weights():
    j, _ = _post("case1_equal_weights.json")
    _assert_valid(j)
    assert j["optimization_strategy"] == "equal_weights"
    for a in j["allocation_changes"]:
        assert a["optimized_weight"] == 50.00


def test_case2_risk_parity():
    j, _ = _post("case2_risk_parity.json")
    _assert_valid(j)
    # risk parity must hold both assets with AGG (low vol) overweight vs 50/50
    agg = next(a for a in j["allocation_changes"] if a["ticker"] == "AGG")
    assert agg["optimized_weight"] > 50


def test_case3_min_volatility():
    j, _ = _post("case3_min_volatility.json")
    _assert_valid(j)
    cur_vol = j["current_portfolio"]["volatility"]
    opt_vol = j["optimized_portfolio"]["volatility"]
    assert opt_vol < cur_vol


def test_case3b_min_drawdown():
    j, _ = _post("case3b_min_drawdown.json")
    _assert_valid(j)
    assert j["optimized_portfolio"]["max_drawdown"] <= j["current_portfolio"]["max_drawdown"]


def test_case4_max_sharpe():
    j, _ = _post("case4_max_sharpe.json")
    _assert_valid(j)
    assert j["optimized_portfolio"]["sharpe_ratio"] >= j["current_portfolio"]["sharpe_ratio"]


def test_case5_constraints_respected():
    j, _ = _post("case5_max_sharpe_constrained.json")
    _assert_valid(j)
    for a in j["allocation_changes"]:
        assert 5.0 - 1e-6 <= a["optimized_weight"] <= 40.0 + 1e-6
    assert j["optimized_portfolio"]["dividend_yield"] >= 2.50 - 1e-6


def test_case6_factor_momentum():
    j, _ = _post("case6_factor_momentum.json")
    _assert_valid(j)
    assert j["factor_betas"] is not None
    cur = j["factor_betas"]["current_portfolio"]["momentum"]
    opt = j["factor_betas"]["optimized_portfolio"]["momentum"]
    assert opt > cur


def test_unsupported_strategy_400():
    body = json.loads((REQ / "case1_equal_weights.json").read_text())
    body["optimization_strategy"] = "maximize_vibes"
    r = client.post("/api/v1/optimize", json=body)
    assert r.status_code == 400


def test_infeasible_constraints_422():
    body = json.loads((REQ / "case5_max_sharpe_constrained.json").read_text())
    body["constraints"] = {"min_weight": 40, "max_weight": 40, "min_dividend_yield": 99}
    r = client.post("/api/v1/optimize", json=body)
    assert r.status_code == 422


def test_per_ticker_keys_case_insensitive():
    body = json.loads((REQ / "case4_max_sharpe.json").read_text())
    body["optimization_strategy"] = "minimize_volatility"
    body["constraints"] = {"min_weights": {"spy": 80}}  # lowercase must work
    r = client.post("/api/v1/optimize", json=body)
    assert r.status_code == 200, r.text
    spy = next(a for a in r.json()["allocation_changes"] if a["ticker"] == "SPY")
    assert spy["optimized_weight"] >= 80 - 1e-6


def test_unknown_ticker_in_constraints_400():
    body = json.loads((REQ / "case4_max_sharpe.json").read_text())
    body["constraints"] = {"min_weights": {"SYP": 10}}  # typo, not silently ignored
    r = client.post("/api/v1/optimize", json=body)
    assert r.status_code == 400


def test_factor_optimization_with_longer_factor_history():
    """Funds with a shorter history than the factor series must still work.

    This is the normal real-world case (different inception dates) and used to
    raise a matmul shape error surfacing as a 500.
    """
    body = json.loads((REQ / "case6_factor_momentum.json").read_text())
    for s in body["securities"]:
        s["returns"] = s["returns"][:30]
        s["dates"] = s["dates"][:30]
    r = client.post("/api/v1/optimize", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["factor_betas"]["optimized_portfolio"]["momentum"] > j["factor_betas"]["current_portfolio"]["momentum"]


def test_dates_length_mismatch_rejected():
    body = json.loads((REQ / "case1_equal_weights.json").read_text())
    body["securities"][0]["dates"] = body["securities"][0]["dates"][:-5]
    r = client.post("/api/v1/optimize", json=body)
    assert r.status_code == 422, r.text


def test_betas_returned_for_every_strategy():
    """factor_betas is not exclusive to the factor-exposure strategy."""
    for name in ("case1_equal_weights.json", "case3_min_volatility.json"):
        j, _ = _post(name)
        assert j["factor_betas"] is not None, name
        for side in ("current_portfolio", "optimized_portfolio"):
            assert set(j["factor_betas"][side]) == {"momentum", "value", "size"}


def test_betas_computed_on_true_date_overlap():
    """Betas must use the date intersection, not the trailing N of each series.

    The shipped factor calendar ends before the fund calendar, so aligning by
    position would pair mismatched observations. This recomputes the OLS fit
    independently on the correct overlap and demands an exact match.
    """
    import numpy as np

    body = json.loads((REQ / "case4_max_sharpe.json").read_text())
    j, _ = _post("case4_max_sharpe.json")

    secs = body["securities"]
    fd = body["factor_data"]
    dates = set(secs[0]["dates"])
    for s in secs[1:]:
        dates &= set(s["dates"])
    fpos = {d: i for i, d in enumerate(fd["dates"])}
    common = [d for d in sorted(dates) if d in fpos]
    assert len(common) < len(dates), "test is only meaningful when calendars differ"

    cols = []
    for s in secs:
        pos = {d: i for i, d in enumerate(s["dates"])}
        cols.append([s["returns"][pos[d]] for d in common])
    weights = np.array([s["current_weight"] for s in secs]) / 100.0
    rp = np.column_stack(cols) @ weights
    F = np.column_stack(
        [[fd[k][fpos[d]] for d in common] for k in ("momentum", "value", "size")]
    )
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(common)), F]), rp, rcond=None)

    got = j["factor_betas"]["current_portfolio"]
    for i, k in enumerate(("momentum", "value", "size"), start=1):
        assert abs(got[k] - coef[i]) < 1e-9, (k, got[k], coef[i])
