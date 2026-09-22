"""Generate deterministic synthetic monthly history for the 5 funds + 3 factors.

The assignment ships an Excel file; this repo was built without it, so this
script creates a realistic stand-in (120 months) with the same tickers the
test scenarios use. Means/vols/correlations mimic the real assets:
bonds (AGG) low-vol, equities higher-vol and correlated, gold diversifying.

Run:  python data/make_sample_data.py
Output: data/sample_history.json  (used by tests + sample requests)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TICKERS = ["SPY", "IEFA", "VEA", "AGG", "GLD"]
NAMES = {
    "SPY": "State Street SPDR S&P 500 ETF Trust",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "VEA": "Vanguard Developed Markets Index Fund;ETF",
    "AGG": "iShares Core US Aggregate Bond ETF",
    "GLD": "SPDR Gold Shares",
}
# Dividend yields from the assignment file (Image 1). GLD cell was blank -> 0.0
# so the Min Dividend Yield constraint math still works.
DIV_YIELDS = {"SPY": 0.99, "IEFA": 3.28, "VEA": 2.03, "AGG": 3.97, "GLD": 0.0}

# monthly means (~ SPY 10% ann, AGG 3% ann, ...)
MEANS = np.array([0.008, 0.005, 0.0052, 0.0025, 0.004])
VOLS = np.array([0.045, 0.050, 0.049, 0.012, 0.050])
CORR = np.array(
    [
        [1.00, 0.80, 0.82, 0.10, 0.05],
        [0.80, 1.00, 0.96, 0.08, 0.10],
        [0.82, 0.96, 1.00, 0.08, 0.10],
        [0.10, 0.08, 0.08, 1.00, 0.05],
        [0.05, 0.10, 0.10, 0.05, 1.00],
    ]
)

FACTOR_MEANS = {"momentum": 0.003, "value": 0.002, "size": 0.0015}
FACTOR_VOLS = {"momentum": 0.030, "value": 0.025, "size": 0.028}


def main(months: int = 120, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    cov = np.outer(VOLS, VOLS) * CORR
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((months, len(TICKERS)))
    R = MEANS + z @ L.T  # (T, N)

    # Factors correlated with the equity block so betas are meaningful.
    spy = R[:, 0]
    factors = {}
    for k in ("momentum", "value", "size"):
        noise = rng.standard_normal(months) * FACTOR_VOLS[k] * 0.8
        factors[k] = list(
            (FACTOR_MEANS[k] + 0.35 * (spy - MEANS[0]) * (FACTOR_VOLS[k] / VOLS[0]) + noise).astype(float)
        )

    out = {
        "periods_per_year": 12,
        "securities": [
            {
                "ticker": t,
                "security_name": NAMES[t],
                "returns": [float(x) for x in R[:, i]],
                "dividend_yield": DIV_YIELDS[t],
            }
            for i, t in enumerate(TICKERS)
        ],
        "factor_data": factors,
    }
    p = Path(__file__).with_name("sample_history.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p} ({months} months)")


if __name__ == "__main__":
    main()
