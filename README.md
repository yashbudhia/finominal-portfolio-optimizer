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

See [Validation against the live tool](#validation-against-the-live-tool) below
for the optimized weights of all six scenarios alongside the reference tool's
output.

Case 6 goes to a corner because maximizing `β = Cw` is *linear* in `w`: with no
weight caps the optimum is always 100% in the single highest-loading fund. Add
`constraints.max_weight` (or per-ticker caps) to get a diversified tilt — that
is the honest behavior of the objective, not a solver artifact. The live tool
behaves the same way.

## Validation against the live tool

All six scenarios were run through https://finominal.com/portfolio-optimizer/US
on the same window (End Date set to 2026-05-27, where `Data.xlsx` ends) and
compared against this API. Tolerance in the brief is 0.1%.

| Case | Strategy | Ours | Live tool | Max diff |
|---|---|---|---|---|
| 1 | Equal Weights | IEFA 50.00, SPY 50.00 | IEFA 50.00, SPY 50.00 | **exact** |
| 2 | Risk Parity | AGG 79.88, VEA 20.12 | AGG 79.89, VEA 20.11 | 0.01 pp |
| 3 | Minimize Volatility | AGG 91.21, SPY 6.92, GLD 1.87 | AGG 91.11, SPY 6.94, GLD 1.95 | 0.10 pp |
| 3b | Minimize Drawdown | AGG 56.43, GLD 31.76, SPY 11.81 | AGG 56.47, GLD 31.72, SPY 11.80 | 0.04 pp |
| 4 | Maximize Sharpe | SPY 69.39, GLD 30.61 | SPY 69.40, GLD 30.60 | 0.01 pp |
| 5 | Max Sharpe + constraints | AGG 40.00, SPY 36.24, IEFA 13.76, GLD 5.00, VEA 5.00 | AGG 40.00, SPY 39.36, IEFA 10.64, GLD 5.00, VEA 5.00 | 3.12 pp — see below |
| 6 | Factor Exposure (Momentum) | VEA 100.00 | GLD 100.00 | n/a — see below |

Cases 1–4 (including 3b) all match within the 0.1% tolerance. Cases 5 and 6
differ for input reasons that are identified and quantified below, not because
of the optimizer.

### Three conventions reverse-engineered from the tool

**1. Risk-free rate is ~1.57%, not 0%.** The spec permits "0% or a standard
value". At 0% this API returned SPY 45.55 / AGG 34.36 / GLD 20.10 for case 4
against the tool's SPY 69.40 / GLD 30.60. Solving max-Sharpe over the same
window at `rf=1.57%` reproduces the tool to 0.01 pp, so that value is used in
the generated case files. Sensitivity is ~0.021 pp of SPY weight per basis
point of `rf`, so the rate is only identifiable to ~0.5 bp from weights
displayed to two decimals — hence 1.57% rather than a falsely precise figure.

**2. The tool optimizes on constant-weight returns but *reports* buy-and-hold
statistics.** Two separate code paths on their side. For case 4's equal-weight
current portfolio (CAGR / vol / maxDD):

| | CAGR | Vol | MaxDD |
|---|---|---|---|
| Live tool | 9.30 | 12.28 | 25.49 |
| Buy & hold (weights drift) | 9.20 | 12.25 | 25.42 |
| Constant weight `R@w` (ours) | 8.67 | 10.87 | 22.28 |

Buy-and-hold reproduces their max drawdown to within 0.01 pp across every
series tested, while annual rebalancing (their own dropdown setting) fits
*worse* than buy-and-hold. Yet their optimized **weights** match a
constant-weight objective exactly.

Case 3b settles this independently. Minimize Drawdown is the only
*path-dependent* objective, so if the tool optimized buy-and-hold paths its
weights would have to differ — instead they agree to 0.04 pp, while its
reported drawdown for those same weights is −18.49% against our 15.65%. The
optimizer works on constant-weight paths; only the results panel is
buy-and-hold.

This API reports constant-weight stats throughout, which is self-consistent
with what it optimizes; the `current_portfolio` / `optimized_portfolio` blocks
are therefore not directly comparable to the tool's results panel, though the
weights are.

**3. The shipped dividend yields are ~2.7% stale.** Scaling the yields in
`Data.xlsx` by 1.027 reproduces the tool's case 5 answer to 0.25 pp. That
factor has three independent derivations that agree: the tool's displayed
current yield (2.64% vs 2.57% from the file on the same basis), a best fit on
an unconstrained yield-only run, and a best fit on the fully-bounded case 5.
The tool pulls live yields; the file is a dated snapshot.

### Why case 5 differs by 3.12 pp (and why it is not a bug)

Both engines bind identically — AGG pinned at its 40% cap, GLD and VEA both at
the 5% floor — leaving exactly 50.00 to split between SPY and IEFA. IEFA yields
3.28% against SPY's 0.99%, so the split is determined entirely by how much IEFA
is needed to clear the 2.50% yield floor, and that depends on the yield vintage.

Two checks confirm the cause is input data, not the solver:

- The tool's answer scores **2.4286%** under the yields in `Data.xlsx` — below
  the 2.50% floor. It is *infeasible* given the data we were told to use, so
  this API cannot return it without violating the constraint.
- Our answer was verified against 101,207 independently sampled feasible
  portfolios: Sharpe 0.734151 vs 0.726396 for the best sample. SLSQP is finding
  the global optimum, not a local one.

### Why case 6 differs

The tool's Factor Exposure panel exposes **five** factors — Value, Momentum,
Low Volatility, Quality, Size — while `Data.xlsx` supplies only three, and its
factor return series are internal rather than the ones shipped. Under the
momentum series we were given, the single-fund loadings rank:

```
VEA 0.1873 > SPY 0.1733 > IEFA 0.1696 > GLD 0.1408 > AGG -0.0131
```

Maximizing a linear objective on the simplex puts 100% in the top-ranked fund,
so this API returns VEA 100% while the tool returns GLD 100%. Both are corner
solutions of the same form. Factor betas, current → optimized:

| | Momentum | Value | Size |
|---|---|---|---|
| Ours (3-factor) | 0.1316 → 0.1873 | 0.1651 → 0.3503 | −0.0576 → −0.0559 |
| Live tool (5-factor) | 0.00 → 0.07 | 0.04 → −0.17 | 0.02 → 0.00 |

Momentum exposure increases in both, which is the criterion the brief sets for
this case ("not expected to match the live tool exactly... the goal is to show
a clear approach"). The differing current-portfolio betas for the *identical*
equal-weight portfolio isolate the model difference from the optimization.

### Committed API responses

The full JSON response for every scenario is checked in under
`sample_responses/`, so the results can be inspected without running anything:

```
sample_responses/case1_equal_weights.json          ... case6_factor_momentum.json
sample_responses/error_unsupported_strategy.json          HTTP 400
sample_responses/error_infeasible_constraints.json        HTTP 422
sample_responses/error_unknown_ticker_in_constraints.json HTTP 400
```

Regenerate with `python scripts/capture_responses.py`. Live-tool screenshots are
in `screenshots/`.

### Reproducing

```bash
uvicorn app.main:app --reload     # terminal 1
python scripts/validate.py        # terminal 2 - checks all 6 acceptance rules
```

In the tool: load `sample_requests/uploads/caseN.xlsx` via **Load Portfolio**,
set End Date to 27/05/2026, pick the matching strategy, and for case 5 type
Min/Max Weight 5/40 into all five rows plus Min Dividend Yield 2.50% (the
upload template carries only Ticker and Percentage, so constraints are manual).

## Project layout

```
app/main.py        FastAPI routes, rounding, stats, error mapping
app/models.py      Pydantic schemas
app/optimizer.py   6 strategies (scipy SLSQP) + constraint builders
app/metrics.py     CAGR / vol / max-DD / Sharpe / covariance
app/factors.py     OLS betas + linear loading matrix C
data/              Data.xlsx (assignment data) + load_excel.py parser
sample_requests/   case1..case6 request bodies + build_cases.py + uploads/*.xlsx
sample_responses/  committed API responses for all 6 cases + the 400/422 paths
screenshots/       live-tool screenshots
tests/             acceptance tests (sums, bounds, case 5/6 rules, 400/422)
scripts/validate.py          live-server validator
scripts/capture_responses.py regenerates sample_responses/
```

## Submission checklist

- [x] Public GitHub repo with source + `README.md`
- [x] All 6 required scenarios run and compared against the live tool
- [x] API responses for every scenario committed (`sample_responses/`)
- [x] Error paths evidenced (400 unsupported strategy, 400 unknown ticker, 422 infeasible)
- [x] 15 unit tests passing (`python -m pytest tests -q`)
- [ ] Live-tool screenshots added to `screenshots/`
- [ ] 3–5 min Loom walkthrough

## Assumptions / trade-offs

- Daily returns, `periods_per_year=252` (configurable; the case files set it
  from the data). Annualization only affects reported stats/constraints, not the
  optimum.
- Covariance is sample covariance + `1e-10` diagonal regularization.
- Non-convex objectives (drawdown, Sharpe) use multi-start SLSQP (equal +
  min-vol + random seeds) — fast and deterministic, global optimum not guaranteed.
  Spot-checked on case 5 against 101,207 random feasible portfolios: SLSQP won.
- Factor betas are OLS with intercept on the common date window.
- `risk_free_rate` is an annual decimal. The generated case files use **0.0157**,
  calibrated from the live tool's own case 4 answer (see Validation below); the
  spec permits "0% or a standard value".
- Portfolio statistics are computed on the constant-weight series `R@w`, matching
  what the optimizer optimizes. The live tool reports buy-and-hold statistics
  instead, so its results *panel* is not directly comparable even where the
  weights agree — quantified in the Validation section.
- A blank dividend-yield cell (GLD) is treated as 0.0 and included in the
  portfolio yield. The live tool excludes zero-yield funds from its *displayed*
  yield but includes them in the *constraint*, which is the behavior implemented
  here.
