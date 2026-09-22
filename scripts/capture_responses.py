"""Capture the API response for every scenario into sample_responses/.

These are committed so the repo carries the required "API responses for the
required test scenarios" deliverable without needing the server running.

Run:  python scripts/capture_responses.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `python scripts/capture_responses.py`

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

REQ = ROOT / "sample_requests"
OUT = ROOT / "sample_responses"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    client = TestClient(app)
    summary = []

    for src in sorted(REQ.glob("case*.json")):
        body = json.loads(src.read_text())
        r = client.post("/api/v1/optimize", json=body)
        r.raise_for_status()
        resp = r.json()
        (OUT / src.name).write_text(json.dumps(resp, indent=2))

        weights = "  ".join(
            f"{a['ticker']} {a['optimized_weight']:.2f}" for a in resp["allocation_changes"]
        )
        summary.append((src.stem, body["optimization_strategy"], weights))
        print(f"wrote sample_responses/{src.name}  ({weights})")

    # error paths, so the repo also evidences the required validation behaviour
    errors = {
        "error_unsupported_strategy": (
            {
                "securities": [
                    {"ticker": "SPY", "current_weight": 100, "returns": [0.01, -0.01]}
                ],
                "optimization_strategy": "maximize_vibes",
            },
            400,
        ),
        "error_infeasible_constraints": (
            {
                "securities": [
                    {
                        "ticker": "SPY",
                        "current_weight": 50,
                        "returns": [0.01, -0.01],
                        "dividend_yield": 0.99,
                    },
                    {
                        "ticker": "AGG",
                        "current_weight": 50,
                        "returns": [0.002, 0.003],
                        "dividend_yield": 3.97,
                    },
                ],
                "optimization_strategy": "maximize_sharpe_ratio",
                "constraints": {"min_dividend_yield": 99},
            },
            422,
        ),
        "error_unknown_ticker_in_constraints": (
            {
                "securities": [
                    {"ticker": "SPY", "current_weight": 50, "returns": [0.01, -0.01]},
                    {"ticker": "AGG", "current_weight": 50, "returns": [0.002, 0.003]},
                ],
                "optimization_strategy": "minimize_volatility",
                "constraints": {"min_weights": {"SYP": 10}},
            },
            400,
        ),
    }
    for name, (payload, expected) in errors.items():
        r = client.post("/api/v1/optimize", json=payload)
        assert r.status_code == expected, (name, r.status_code, r.text)
        (OUT / f"{name}.json").write_text(
            json.dumps({"request": payload, "http_status": r.status_code, "response": r.json()}, indent=2)
        )
        print(f"wrote sample_responses/{name}.json  (HTTP {r.status_code})")

    print("\nsummary:")
    for stem, strategy, weights in summary:
        print(f"  {stem:34s} {strategy:26s} {weights}")


if __name__ == "__main__":
    main()
