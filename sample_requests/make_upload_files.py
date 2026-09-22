"""Emit upload files for the live tool's "Load Portfolio" button, one per scenario.

Schema copied from the tool's own `Sample Portfolio` download (example_us.xlsx):
a single sheet with exactly two columns, `Ticker` and `Percentage` (whole
numbers summing to 100). Security Name is resolved by the tool from the ticker.

Min/Max Weight are NOT importable - the template has no columns for them, so
case 5's 5% / 40% bounds must be typed into the Holdings table by hand. Same
for everything on the Configure Optimization tab (strategy, Min Dividend
Yield, factor objective).

Run:  python sample_requests/make_upload_files.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "uploads"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for src in sorted(HERE.glob("case*.json")):
        body = json.loads(src.read_text())
        df = pd.DataFrame(
            {
                "Ticker": [s["ticker"] for s in body["securities"]],
                "Percentage": [s["current_weight"] for s in body["securities"]],
            }
        )
        dest = OUT / f"{src.stem}.xlsx"
        df.to_excel(dest, index=False, sheet_name="Sheet1")

        manual = []
        c = body.get("constraints") or {}
        if c.get("min_weight") is not None or c.get("max_weight") is not None:
            manual.append(f"Min/Max Weight {c.get('min_weight')}/{c.get('max_weight')} per row")
        if c.get("min_dividend_yield") is not None:
            manual.append(f"Min Dividend Yield {c['min_dividend_yield']}%")
        if body.get("factor_objective"):
            fo = body["factor_objective"]
            manual.append(f"factor {fo['factor']}/{fo['direction']}")
        print(
            f"wrote uploads/{dest.name}  {len(df)} holdings, total {df.Percentage.sum():g}%"
            f"  | strategy={body['optimization_strategy']}"
            + (f"  | TYPE BY HAND: {'; '.join(manual)}" if manual else "")
        )


if __name__ == "__main__":
    main()
