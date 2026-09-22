"""Request/response schemas for the optimizer API."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SecurityInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. SPY")
    security_name: Optional[str] = Field(None, description="Human readable name")
    current_weight: float = Field(..., description="Current weight in percent (0-100)")
    returns: List[float] = Field(
        ..., description="Periodic historical returns as decimals (0.01 == 1%)"
    )
    dates: Optional[List[str]] = Field(
        None,
        description="Optional ISO dates matching `returns`. When supplied for every "
        "security, histories are aligned by date instead of by trailing position, "
        "which matters when calendars differ.",
    )
    dividend_yield: Optional[float] = Field(
        None, description="Dividend yield in percent (e.g. 2.5 == 2.5%)"
    )

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker must be non-empty")
        return v

    @model_validator(mode="after")
    def _dates_match(self) -> "SecurityInput":
        if self.dates is not None and len(self.dates) != len(self.returns):
            raise ValueError(
                f"{self.ticker}: dates has {len(self.dates)} entries but returns has "
                f"{len(self.returns)}"
            )
        return self


class FactorData(BaseModel):
    momentum: List[float]
    value: List[float]
    size: List[float]
    dates: Optional[List[str]] = Field(
        None,
        description="Optional ISO dates matching the factor series. When supplied "
        "alongside security dates, betas are computed on the true date overlap.",
    )

    @model_validator(mode="after")
    def _nonempty(self) -> "FactorData":
        for k in ("momentum", "value", "size"):
            if len(getattr(self, k)) == 0:
                raise ValueError(f"factor_data.{k} must be non-empty")
        n = len(self.momentum)
        if not (len(self.value) == len(self.size) == n):
            raise ValueError("factor_data series must all be the same length")
        if self.dates is not None and len(self.dates) != n:
            raise ValueError(
                f"factor_data.dates has {len(self.dates)} entries but series have {n}"
            )
        return self


class FactorObjective(BaseModel):
    factor: Literal["momentum", "value", "size"] = "momentum"
    direction: Literal["maximize", "minimize"] = "maximize"

    @field_validator("factor", mode="before")
    @classmethod
    def _lower_factor(cls, v: str) -> str:
        return str(v).lower()

    @field_validator("direction", mode="before")
    @classmethod
    def _lower_dir(cls, v: str) -> str:
        return str(v).lower()


class PortfolioConstraints(BaseModel):
    """All weights / yields / returns in PERCENT (0-100 scale).

    min_weight / max_weight apply globally; min_weights / max_weights
    override per ticker.
    """

    min_weight: Optional[float] = None
    max_weight: Optional[float] = None
    min_weights: Optional[Dict[str, float]] = None
    max_weights: Optional[Dict[str, float]] = None
    min_cagr: Optional[float] = None
    min_volatility: Optional[float] = None
    max_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    min_dividend_yield: Optional[float] = None


class OptimizeRequest(BaseModel):
    securities: List[SecurityInput] = Field(..., min_length=1)
    optimization_strategy: str = Field(
        ..., description="e.g. minimize_volatility, maximize_sharpe, ..."
    )
    constraints: Optional[PortfolioConstraints] = None
    factor_data: Optional[FactorData] = None
    factor_objective: Optional[FactorObjective] = None
    risk_free_rate: float = Field(
        0.0, description="Annual risk-free rate as decimal (0.02 == 2%)"
    )
    periods_per_year: int = Field(12, ge=1, le=252)

    @field_validator("optimization_strategy", mode="before")
    @classmethod
    def _norm_strategy(cls, v: str) -> str:
        return str(v).strip().lower().replace(" ", "_").replace("-", "_")

    @model_validator(mode="after")
    def _checks(self) -> "OptimizeRequest":
        tickers = [s.ticker for s in self.securities]
        if len(set(tickers)) != len(tickers):
            raise ValueError(f"duplicate tickers: {tickers}")
        for s in self.securities:
            if len(s.returns) < 2:
                raise ValueError(f"{s.ticker}: need at least 2 return observations")
            if not (0 <= s.current_weight <= 100):
                raise ValueError(f"{s.ticker}: current_weight must be 0-100")
        total = sum(s.current_weight for s in self.securities)
        if abs(total - 100.0) > 1.0:
            raise ValueError(
                f"current weights must sum to ~100 (got {total:.2f})"
            )
        return self


class AllocationChange(BaseModel):
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float


class PortfolioStats(BaseModel):
    cagr: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None


class OptimizeResponse(BaseModel):
    optimization_strategy: str
    allocation_changes: List[AllocationChange]
    current_portfolio: Optional[PortfolioStats] = None
    optimized_portfolio: Optional[PortfolioStats] = None
    factor_betas: Optional[Dict[str, Dict[str, float]]] = None
