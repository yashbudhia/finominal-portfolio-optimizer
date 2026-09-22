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

Sample requests live in `sample_requests/case*.json`, built from the real
assignment data in `data/Data.xlsx` (**daily** total returns for
SPY/IEFA/VEA/AGG/GLD + Momentum/Value/Size factor returns, plus dividend
yields):

```bash
python data/load_excel.py              # Data.xlsx -> data/real_history.json
python sample_requests/build_cases.py  # rebuild case JSONs
python -m pytest tests -q              # 12 acceptance tests
python scripts/validate.py             # same checks vs running server
```

Funds have different inception dates (SPY 1993, AGG 2003, GLD 2004, VEA 2007,
IEFA 2012) and the factor calendar ends a few days earlier than the fund
calendar, so each scenario is aligned on the **intersection of dates actually
present** for the tickers (and factors) it uses. The window used is recorded in
each case file under `_history`:

| Case | Tickers | Window | Obs |
|---|---|---|---|
| 1 | IEFA, SPY | 2012-10-23 → 2026-05-27 | 3414 |
| 2 | VEA, AGG | 2007-07-26 → 2026-05-27 | 4739 |
| 3 / 3b | SPY, AGG, GLD | 2004-11-18 → 2026-05-27 | 5413 |
| 4 / 5 | all five | 2012-10-23 → 2026-05-27 | 3414 |
| 6 | all five + factors | 2012-10-23 → 2026-05-22 | 3412 |

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

## Results on the real data

| Case | Strategy | Optimized weights | Effect |
|---|---|---|---|
| 1 | equal_weights | IEFA 50.00, SPY 50.00 | exact 100/N |
| 2 | risk_parity | AGG 79.88, VEA 20.12 | vol 6.89% → 6.26% |
| 3 | minimize_volatility | AGG 91.21, SPY 6.92, GLD 1.87 | vol 11.75% → **5.01%** |
| 3b | minimize_drawdown | AGG 56.43, GLD 31.76, SPY 11.81 | maxDD 33.66% → **15.65%** |
| 4 | maximize_sharpe_ratio | SPY 45.54, AGG 34.36, GLD 20.10, IEFA/VEA 0 | Sharpe 0.820 → **1.030** |
| 5 | max Sharpe + constraints | AGG 40.00, SPY 33.18, IEFA 14.68, GLD 7.14, VEA 5.00 | Sharpe 0.820 → 0.901, DivY **2.50%**, all within 5–40% |
| 6 | optimize_factor_exposure | VEA 100.00 | Momentum β 0.1316 → **0.1873** |

Case 6 goes to a corner because maximizing `β = Cw` is *linear* in `w`: with no
weight caps the optimum is always 100% in the single highest-loading fund. Add
`constraints.max_weight` (or per-ticker caps) to get a diversified tilt — that
is the honest behavior of the objective, not a solver artifact.

## Validating vs the live tool

1. Open https://finominal.com/portfolio-optimizer/US
2. Enter the same tickers/weights as the case JSON, same strategy + constraints.
3. Run Optimization → compare **Review Results → Allocation Changes**.
4. (Bonus) Compare **Comparison → Factor Betas**.
5. Expect close match; <0.1% float diffs OK. Note case 6 uses only 3 factors
   here vs the tool's broader model — check direction (Momentum up) not exact match.

If the tool's numbers differ materially, the likely causes are return frequency
(the data is daily; the tool may resample to monthly), `ddof` in the covariance
/ volatility estimate, and the risk-free assumption in Sharpe. All three are
single-line changes: `periods_per_year` in the request, `ddof` in
`app/metrics.py`, `risk_free_rate` in the request.

## Project layout

```
app/main.py        FastAPI routes, rounding, stats, error mapping
app/models.py      Pydantic schemas
app/optimizer.py   6 strategies (scipy SLSQP) + constraint builders
app/metrics.py     CAGR / vol / max-DD / Sharpe / covariance
app/factors.py     OLS betas + linear loading matrix C
data/              Data.xlsx (assignment data) + load_excel.py parser
sample_requests/   case1..case6 JSON + build_cases.py
tests/             acceptance tests (sums, bounds, case 5/6 rules, 400/422)
scripts/validate.py  live-server validator for screenshots
```

## Assumptions / trade-offs

- Daily returns, `periods_per_year=252` (configurable; the case files set it
  from the data). Annualization only affects reported stats/constraints, not the
  optimum.
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
