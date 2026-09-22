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
    body = json.loads((REQ / "case6_factor_momentum.json").read_text())
    for s in body["securities"]:
        s["returns"] = s["returns"][:30]  # funds shorter than 120-obs factors
    r = client.post("/api/v1/optimize", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["factor_betas"]["optimized_portfolio"]["momentum"] > j["factor_betas"]["current_portfolio"]["momentum"]
