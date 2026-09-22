"""Emit CSVs for the live tool's "Load Portfolio" button, one per scenario.

Columns mirror the Import Portfolio table headers exactly:
    Ticker, Security Name, Allocation, Min Weight, Max Weight

Only case 5 has per-row Min/Max Weight (5% / 40%); the rest are blank.
Portfolio-level settings (strategy, Min Dividend Yield, factor objective) are
NOT part of the holdings table - set those on the Configure Optimization tab.

Run:  python sample_requests/make_upload_csvs.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "uploads"
HEADER = ["Ticker", "Security Name", "Allocation", "Min Weight", "Max Weight"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for src in sorted(HERE.glob("case*.json")):
        body = json.loads(src.read_text())
        c = body.get("constraints") or {}
        gmin, gmax = c.get("min_weight"), c.get("max_weight")
        per_min = {k.upper(): v for k, v in (c.get("min_weights") or {}).items()}
        per_max = {k.upper(): v for k, v in (c.get("max_weights") or {}).items()}

        rows = []
        for s in body["securities"]:
            t = s["ticker"]
            rows.append(
                [
                    t,
                    s.get("security_name", ""),
                    f"{s['current_weight']:g}",
                    _fmt(per_min.get(t, gmin)),
                    _fmt(per_max.get(t, gmax)),
                ]
            )

        dest = OUT / f"{src.stem}.csv"
        with dest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(rows)

        note = f"strategy={body['optimization_strategy']}"
        if c.get("min_dividend_yield") is not None:
            note += f", min_dividend_yield={c['min_dividend_yield']}%"
        if body.get("factor_objective"):
            fo = body["factor_objective"]
            note += f", factor={fo['factor']}/{fo['direction']}"
        print(f"wrote uploads/{dest.name}  {len(rows)} holdings  ({note})")


def _fmt(v) -> str:
    return "" if v is None else f"{v:g}"


if __name__ == "__main__":
    main()
