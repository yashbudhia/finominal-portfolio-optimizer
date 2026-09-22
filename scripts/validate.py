"""Validate all 6 assignment scenarios against the running API.

Usage:
    uvicorn app.main:app --reload          # terminal 1
    python scripts/validate.py             # terminal 2
    python scripts/validate.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import urllib.request

CASE_FILES = sorted(glob.glob("sample_requests/case*.json"))


def post(base: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        base.rstrip("/") + "/api/v1/optimize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:500]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()

    failures = 0
    for f in CASE_FILES:
        body = json.loads(open(f).read())
        status, resp = post(args.base_url, body)
        print(f"\n==== {f} -> HTTP {status}")
        if status != 200:
            print("  FAILED:", resp)
            failures += 1
            continue
        total = sum(a["optimized_weight"] for a in resp["allocation_changes"])
        neg = [a for a in resp["allocation_changes"] if a["optimized_weight"] < -1e-9]
        print(f"  strategy={resp['optimization_strategy']} sum={total:.2f}")
        for a in resp["allocation_changes"]:
            print(f"    {a['ticker']:5s} cur={a['current_weight']:6.2f} opt={a['optimized_weight']:6.2f} chg={a['change']:+6.2f}")
        if abs(total - 100) > 1e-6 or neg:
            print("  FAILED acceptance: sum/negativity")
            failures += 1
        if "case5" in f:
            bad = [a for a in resp["allocation_changes"] if not (5 - 1e-6 <= a["optimized_weight"] <= 40 + 1e-6)]
            dy = resp["optimized_portfolio"]["dividend_yield"]
            print(f"  dividend_yield={dy} (need >= 2.50)")
            if bad or (dy is not None and dy < 2.50 - 1e-6):
                print("  FAILED case5 constraints")
                failures += 1
        if "case6" in f:
            cur = resp["factor_betas"]["current_portfolio"]["momentum"]
            opt = resp["factor_betas"]["optimized_portfolio"]["momentum"]
            print(f"  momentum beta {cur:.4f} -> {opt:.4f}")
            if not opt > cur:
                print("  FAILED case6 momentum increase")
                failures += 1
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
