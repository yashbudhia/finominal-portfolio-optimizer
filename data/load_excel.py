"""Convert the assignment's Data.xlsx into date-indexed JSON history.

The file ships three sheets:

    Fund Info      ticker | fund_name | dividend_yield      (yield as decimal)
    Fund Returns   date | total_return | ticker             (long format, DAILY)
    Factor Returns date | total_return | index_ticker        (Momentum/Value/Size)

Returns are DAILY and each fund has a different inception date, so series are
kept date-indexed here and aligned per-scenario by build_cases.py.

    python data/load_excel.py                    # writes data/real_history.json
    python data/load_excel.py --xlsx other.xlsx  # different source file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FACTOR_NAMES = {
    "Momentum Factor": "momentum",
    "Value Factor": "value",
    "Size Factor": "size",
}
# Daily observations -> annualize with trading days, not months.
PERIODS_PER_YEAR = 252


def load(xlsx: Path) -> dict:
    info = pd.read_excel(xlsx, sheet_name="Fund Info")
    rets = pd.read_excel(xlsx, sheet_name="Fund Returns")
    facs = pd.read_excel(xlsx, sheet_name="Factor Returns")

    info["ticker"] = info["ticker"].astype(str).str.strip().str.upper()
    # dividend_yield is a decimal in the file (0.03278); the API takes percent.
    # GLD's cell is blank -> 0.0 so the min-dividend-yield constraint still works.
    yields = {
        r.ticker: round(float(r.dividend_yield) * 100.0, 4)
        if pd.notna(r.dividend_yield)
        else 0.0
        for r in info.itertuples()
    }
    names = {r.ticker: str(r.fund_name).strip() for r in info.itertuples()}

    rets["ticker"] = rets["ticker"].astype(str).str.strip().str.upper()
    rets["date"] = pd.to_datetime(rets["date"])
    wide = (
        rets.pivot_table(index="date", columns="ticker", values="total_return")
        .sort_index()  # source file is newest-first
    )

    funds = {}
    for t in wide.columns:
        s = wide[t].dropna()
        funds[t] = {
            "security_name": names.get(t, t),
            "dividend_yield": yields.get(t, 0.0),
            "dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "returns": [float(x) for x in s.values],
        }

    facs["index_ticker"] = facs["index_ticker"].astype(str).str.strip()
    facs["date"] = pd.to_datetime(facs["date"])
    fwide = (
        facs.pivot_table(index="date", columns="index_ticker", values="total_return")
        .rename(columns=FACTOR_NAMES)
        .sort_index()
    )
    missing = [k for k in ("momentum", "value", "size") if k not in fwide.columns]
    if missing:
        raise SystemExit(f"Factor Returns sheet is missing {missing}; found {list(fwide.columns)}")
    fwide = fwide[["momentum", "value", "size"]].dropna()

    return {
        "source": xlsx.name,
        "periods_per_year": PERIODS_PER_YEAR,
        "funds": funds,
        "factors": {
            "dates": [d.strftime("%Y-%m-%d") for d in fwide.index],
            "momentum": [float(x) for x in fwide["momentum"].values],
            "value": [float(x) for x in fwide["value"].values],
            "size": [float(x) for x in fwide["size"].values],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(HERE / "Data.xlsx"))
    ap.add_argument("--out", default=str(HERE / "real_history.json"))
    args = ap.parse_args()

    hist = load(Path(args.xlsx))
    Path(args.out).write_text(json.dumps(hist))

    print(f"wrote {args.out}")
    for t, f in sorted(hist["funds"].items()):
        print(
            f"  {t:5s} {len(f['returns']):5d} obs  "
            f"{f['dates'][0]} -> {f['dates'][-1]}  div {f['dividend_yield']:.2f}%"
        )
    fa = hist["factors"]
    print(f"  factors {len(fa['dates']):5d} obs  {fa['dates'][0]} -> {fa['dates'][-1]}")


if __name__ == "__main__":
    main()
