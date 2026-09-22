# Portfolio Optimizer API

REST API replicating the core engine behind Finominal's Portfolio Optimizer.
Implements all 5 required strategies + the bonus Factor Exposure strategy,
with weight and portfolio-level constraints and factor betas.

## Strategies

| # | Name (`optimization_strategy`) | Method |
|---|---|---|
| 1 | `equal_weights` | 100/N each |
| 2 | `risk_parity` | equalize risk contributions `w·(Σw)/σ²` via SLSQP |
| 3 | `minimize_drawdown` | minimize max peak-to-trough drawdown (multi-start SLSQP) |
| 4 | `minimize_volatility` | minimum-variance `min √(wᵀΣw)` via SLSQP |
| 5 | `maximize_sharpe_ratio` | `max mean/vol`, risk-free 0% default |
| 6 (bonus) | `optimize_factor_exposure` | max/min Momentum/Value/Size beta (exact linear objective `β=Cw`) |

Strategy names are case/space-insensitive (`"Maximize Sharpe"`, `"max-sharpe"` etc. all work).

## Quickstart

```bash
cd portfolio-optimizer
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# NOTE: pin numpy<2 (scipy wheels in this env are built against numpy 1.x)
uvicorn app.main:app --reload
# docs: http://localhost:8000/docs
```

## API

### `POST /api/v1/optimize`

Request (weights/yields in **percent**, returns as **decimals**):

```json
{
  "securities": [
    {"ticker": "SPY", "security_name": "SPDR S&P 500 ETF Trust",
     "current_weight": 60, "returns": [0.012, -0.02, 0.03], "dividend_yield": 1.3},
    {"ticker": "AGG", "current_weight": 30, "returns": [0.003, 0.004, 0.002], "dividend_yield": 3.2},
    {"ticker": "GLD", "current_weight": 10, "returns": [0.01, 0.02, -0.01], "dividend_yield": 0.0}
  ],
  "optimization_strategy": "minimize_volatility",
  "constraints": {
    "min_weight": 5, "max_weight": 40,
    "min_cagr": 3.0, "min_volatility": 2.0, "max_volatility": 15.0,
    "max_drawdown": 20.0, "min_dividend_yield": 2.5
  },
  "factor_data": {"momentum": [...], "value": [...], "size": [...]},
  "factor_objective": {"factor": "momentum", "direction": "maximize"},
  "risk_free_rate": 0.0,
  "periods_per_year": 12
}
```

`constraints.min_weights / max_weights` (per-ticker dicts) override the global
`min_weight / max_weight`. Return histories are aligned to the trailing common
window, so funds with different inception dates still work.

Response:

```json
{
  "optimization_strategy": "minimize_volatility",
  "allocation_changes": [
    {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF",
     "current_weight": 50.0, "optimized_weight": 37.65, "change": -12.35}
  ],
  "current_portfolio": {"cagr": 5.1, "volatility": 8.2, "max_drawdown": 12.0, "sharpe_ratio": 0.6, "dividend_yield": 2.1},
  "optimized_portfolio": {...},
  "factor_betas": {
    "current_portfolio": {"value": 0.09, "momentum": 0.10, "size": 0.12},
    "optimized_portfolio": {"value": 0.14, "momentum": 0.18, "size": 0.08}
  }
}
```

`factor_betas` is returned whenever `factor_data` is supplied (any strategy).
Factor model: `r_p = α + β_m·Mom + β_v·Val + β_s·Size + ε` (OLS with intercept).

Errors: `400` unsupported strategy / malformed input, `422` infeasible
constraints (clear message, never silent invalid weights). Weights always sum
to 100.00 (largest-remainder rounding) and are never negative (no short selling).

### `GET /health`, `GET /strategies`

## Test scenarios (assignment §Required Test Scenarios)

Sample requests live in `sample_requests/case*.json`, built from
`data/sample_history.json` (120 months synthetic history for
SPY/IEFA/VEA/AGG/GLD + 3 factors — stand-in for the assignment Excel file;
swap in real data via `data/load_excel.py`):

```bash
python data/make_sample_data.py        # regenerate synthetic history
python sample_requests/build_cases.py  # rebuild case JSONs
python -m pytest tests -q              # 9 acceptance tests
python scripts/validate.py             # same checks vs running server
```

| Case | File | Strategy |
|---|---|---|
| 1 | `case1_equal_weights.json` | equal_weights |
| 2 | `case2_risk_parity.json` | risk_parity |
| 3 | `case3_min_volatility.json` + `case3b_min_drawdown.json` | min vol / min DD |
| 4 | `case4_max_sharpe.json` | max Sharpe, no constraints |
| 5 | `case5_max_sharpe_constrained.json` | max Sharpe + DivY≥2.5%, 5–40% each |
| 6 | `case6_factor_momentum.json` | maximize Momentum + betas |

Example:

```bash
curl -s http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d @sample_requests/case3_min_volatility.json
```

## Validating vs the live tool

1. Open https://finominal.com/portfolio-optimizer/US
2. Enter the same tickers/weights as the case JSON, same strategy + constraints.
3. Run Optimization → compare **Review Results → Allocation Changes**.
4. (Bonus) Compare **Comparison → Factor Betas**.
5. Expect close match; <0.1% float diffs OK. Note case 6 uses only 3 factors
   here vs the tool's broader model — check direction (Momentum up) not exact match.

To use your real Excel file: `python data/load_excel.py history.xlsx --weights SPY=60,AGG=30,GLD=10 --strategy minimize_volatility`.

## Project layout

```
app/main.py        FastAPI routes, rounding, stats, error mapping
app/models.py      Pydantic schemas
app/optimizer.py   6 strategies (scipy SLSQP) + constraint builders
app/metrics.py     CAGR / vol / max-DD / Sharpe / covariance
app/factors.py     OLS betas + linear loading matrix C
data/              sample_history.json, make_sample_data.py, load_excel.py
sample_requests/   case1..case6 JSON + build_cases.py
tests/             acceptance tests (sums, bounds, case 5/6 rules, 400/422)
scripts/validate.py  live-server validator for screenshots
```

## Assumptions / trade-offs

- Monthly returns, `periods_per_year=12` (configurable). Annualization only
  affects reported stats/constraints, not the optimum.
- Covariance is sample covariance + `1e-10` diagonal regularization.
- Non-convex objectives (drawdown, Sharpe) use multi-start SLSQP (equal +
  min-vol + random seeds) — fast and deterministic, global optimum not guaranteed.
- Factor betas are OLS with intercept on the trailing common window.
- `risk_free_rate` is annual decimal (default 0 per spec).

## Submission checklist

- [ ] `pip install -r requirements.txt` + `uvicorn app.main:app` screenshot
- [ ] `curl` outputs for ≥3 strategies
- [ ] Live-tool screenshots for the 6 cases
- [ ] 3–5 min Loom: code walkthrough, decisions, shortcuts
