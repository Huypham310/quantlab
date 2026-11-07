"""Request / response models. Validation happens here so bad input is a 422, not a failed job."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from quantlab.api.jobs import validate_strategy

Timeframe = Literal["1h", "1d", "1m"]


class Bar(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class SymbolInfo(BaseModel):
    source: str
    symbol: str
    timeframe: str
    bars: int
    first: datetime
    last: datetime


class StrategyInfo(BaseModel):
    name: str
    params: list[str]
    grid: dict[str, list]


class BacktestRequest(BaseModel):
    source: str = "coinbase"
    symbol: str = Field(min_length=1, max_length=32)
    timeframe: Timeframe = "1h"
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    vol_target: float | None = Field(default=None, gt=0, le=5)
    band: float = Field(default=0.0, ge=0, le=1)
    fee_bps: float = Field(default=10.0, ge=0, le=500)
    slippage_bps: float = Field(default=5.0, ge=0, le=500)
    cash: float = Field(default=10_000.0, gt=0)
    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> BacktestRequest:
        validate_strategy(self.strategy, self.params)  # ValueError -> 422
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must be before end")
        return self


class TrainRequest(BaseModel):
    source: str = "coinbase"
    symbol: str = Field(min_length=1, max_length=32)
    timeframe: Timeframe = "1h"
    model: Literal["lgbm", "lstm"] = "lgbm"
    horizon: int = Field(default=24, ge=1, le=24 * 7)
    train_days: int = Field(default=365, ge=7, le=3650)
    test_days: int = Field(default=90, ge=1, le=365)
    # cap trials: one request shouldn't tie up the worker for an hour
    trials: int = Field(default=10, ge=1, le=50)
    seed: int = 0

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class PaperOrderRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0)
    type: Literal["market", "limit"] = "market"
    limit_price: float | None = Field(default=None, gt=0)

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _limit_needs_price(self) -> PaperOrderRequest:
        if (self.type == "limit") != (self.limit_price is not None):
            raise ValueError("limit orders need limit_price, market orders must not have one")
        return self


class PaperOrder(BaseModel):
    id: int
    symbol: str
    side: str
    qty: float
    type: str
    limit_price: float | None
    created_ms: int
    status: str
    filled_qty: float
    avg_price: float
    reason: str
    marketable: bool


class JobCreated(BaseModel):
    job_id: str
    kind: Literal["backtest", "train"]


class JobStatus(BaseModel):
    job_id: str
    status: str  # PENDING / STARTED / PROGRESS / SUCCESS / FAILURE
    progress: dict | None = None
    result: dict | None = None
    error: str | None = None
