"""Convert the assignment Excel file into an /optimize request body.

The assignment ships historical returns + factor returns + dividend yields in
Excel. Sheet/column names may vary, so this is a starting template: point it
at your file, adjust sheet names, then run:

    python data/load_excel.py path/to/history.xlsx --weights SPY=60,AGG=30,GLD=10

It prints a JSON request body you can POST to /api/v1/optimize.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def parse_weights(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in spec.split(","):
        t, w = part.split("=")
        out[t.strip().upper()] = float(w)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="Path to assignment Excel file")
    ap.add_argument("--weights", default="SPY=20,IEFA=20,VEA=20,AGG=20,GLD=20")
    ap.add_argument("--strategy", default="minimize_volatility")
    ap.add_argument("--returns-sheet", default=0, help="Sheet with fund returns (Date + tickers as columns)")
    ap.add_argument("--factors-sheet", default=1, help="Sheet with Momentum/Value/Size columns")
    ap.add_argument("--yields", default="", help="Optional 'SPY=1.3,AGG=3.2,...'")
    args = ap.parse_args()

    rets = pd.read_excel(args.xlsx, sheet_name=args.returns_sheet)
    # Assume first column is Date; remaining columns are tickers.
    tickers = [c for c in rets.columns if str(c).lower() != "date"]
    weights = parse_weights(args.weights)
    yields = parse_weights(args.yields) if args.yields else {}

    securities = []
    for t in tickers:
        s = rets[t].dropna().astype(float).tolist()
        # Excel may store percents (1.5 == 1.5%) or decimals; normalize >1 to percent.
        if s and max(abs(x) for x in s) > 1.0:
            s = [x / 100.0 for x in s]
        securities.append(
            {
                "ticker": str(t).upper(),
                "current_weight": float(weights.get(str(t).upper(), 0.0)),
                "returns": s,
                **({"dividend_yield": yields[str(t).upper()]} if str(t).upper() in yields else {}),
            }
        )

    body: dict = {
        "securities": securities,
        "optimization_strategy": args.strategy,
        "periods_per_year": 12,
    }
    try:
        fac = pd.read_excel(args.xlsx, sheet_name=args.factors_sheet)
        cols = {str(c).lower(): c for c in fac.columns}
        if all(k in cols for k in ("momentum", "value", "size")):
            body["factor_data"] = {
                k: fac[cols[k]].dropna().astype(float).tolist() for k in ("momentum", "value", "size")
            }
    except Exception as e:
        print(f"warning: could not read factors sheet: {e}", file=sys.stderr)

    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
