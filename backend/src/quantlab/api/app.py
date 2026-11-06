"""QuantLab API.

    uvicorn quantlab.api.app:app --port 8000

REST for market data, strategies and jobs; WebSockets for live trades and job progress.
Backtests and training run on the Celery worker (see quantlab.worker), never in the request.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from redis import asyncio as aioredis
from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine

from quantlab.api.jobs import strategy_params
from quantlab.api.schemas import (
    BacktestRequest,
    Bar,
    JobCreated,
    JobStatus,
    StrategyInfo,
    SymbolInfo,
    TrainRequest,
)
from quantlab.backtest.strategies import LIBRARY
from quantlab.config import get_settings
from quantlab.data.store import bars as bars_table
from quantlab.data.store import make_engine
from quantlab.stream.trades import parse_entry
from quantlab.worker import celery_app, run_backtest, train_model

app = FastAPI(title="QuantLab", version="0.1.0")


@lru_cache
def _engine() -> Engine:
    return make_engine(get_settings().database_url)


def get_engine() -> Engine:
    return _engine()


def get_redis() -> aioredis.Redis:
    return aioredis.from_url(get_settings().redis_url)


def _utc(ts: datetime) -> datetime:
    # sqlite returns naive datetimes; everything we store is UTC
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


@app.get("/health")
def health(engine: Engine = Depends(get_engine)) -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db = True
    except Exception:  # noqa: BLE001 - health check reports, never raises
        db = False
    return {"status": "ok" if db else "degraded", "database": db}


SYMBOLS_TTL_S = 5.0
_symbols_cache: dict[int, tuple[float, list[SymbolInfo]]] = {}


@app.get("/symbols", response_model=list[SymbolInfo])
def symbols(engine: Engine = Depends(get_engine)) -> list[SymbolInfo]:
    """Cached for a few seconds: the GROUP BY scans every bar (17 ms on 52k rows) and the
    answer only changes when an ingest runs."""
    hit = _symbols_cache.get(id(engine))
    if hit and time.monotonic() - hit[0] < SYMBOLS_TTL_S:
        return hit[1]
    out = _load_symbols(engine)
    _symbols_cache[id(engine)] = (time.monotonic(), out)
    return out


def _load_symbols(engine: Engine) -> list[SymbolInfo]:
    b = bars_table.c
    q = (
        select(b.source, b.symbol, b.timeframe, func.count(), func.min(b.ts), func.max(b.ts))
        .group_by(b.source, b.symbol, b.timeframe)
        .order_by(b.symbol, b.timeframe)
    )
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    return [
        SymbolInfo(source=s, symbol=sym, timeframe=tf, bars=n, first=_utc(lo), last=_utc(hi))
        for s, sym, tf, n, lo, hi in rows
    ]


@app.get("/bars/{symbol}", response_model=list[Bar])
def get_bars(
    symbol: str,
    timeframe: str = "1h",
    source: str = "coinbase",
    limit: int = Query(500, ge=1, le=5000),
    end: datetime | None = None,
    engine: Engine = Depends(get_engine),
) -> list[Bar]:
    """The newest `limit` bars (before `end` if given), oldest first."""
    b = bars_table.c
    q = select(b.ts, b.open, b.high, b.low, b.close, b.volume).where(
        b.source == source, b.symbol == symbol.upper(), b.timeframe == timeframe
    )
    if end is not None:
        q = q.where(b.ts < end)
    # newest first + LIMIT uses the (source, symbol, timeframe, ts) primary key index
    with engine.connect() as conn:
        rows = conn.execute(q.order_by(b.ts.desc()).limit(limit)).all()
    if not rows:
        raise HTTPException(404, f"no {timeframe} bars for {symbol.upper()} from {source}")
    # pydantic costs ~0.9 ms for 500 bars here (measured); the SQL round trip is ~3.4 ms
    return [
        Bar(ts=_utc(r[0]), open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5])
        for r in reversed(rows)
    ]


@app.get("/strategies", response_model=list[StrategyInfo])
def strategies() -> list[StrategyInfo]:
    out = [StrategyInfo(name="buy_and_hold", params=[], grid={})]
    for name, (_, grid) in LIBRARY.items():
        out.append(StrategyInfo(name=name, params=strategy_params(name), grid=grid))
    return out


@app.post("/backtests", response_model=JobCreated, status_code=202)
def create_backtest(req: BacktestRequest) -> JobCreated:
    task = run_backtest.delay(req.model_dump(mode="json"))
    return JobCreated(job_id=task.id, kind="backtest")


@app.post("/models/train", response_model=JobCreated, status_code=202)
def create_training(req: TrainRequest) -> JobCreated:
    task = train_model.delay(req.model_dump(mode="json"))
    return JobCreated(job_id=task.id, kind="train")


def job_status(job_id: str) -> JobStatus:
    res = AsyncResult(job_id, app=celery_app)
    state = res.state
    out = JobStatus(job_id=job_id, status=state)
    if state == "PROGRESS" and isinstance(res.info, dict):
        out.progress = res.info
    elif state == "SUCCESS":
        out.result = res.result
    elif state == "FAILURE":
        # keep the message, not the traceback: it goes straight to the browser
        out.error = f"{type(res.result).__name__}: {res.result}"
    return out


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    """Celery can't tell an unknown id from a queued job, so both read as PENDING."""
    return job_status(job_id)


@app.websocket("/ws/jobs/{job_id}")
async def ws_job(ws: WebSocket, job_id: str, interval: float = 0.5) -> None:
    """Pushes the job status whenever it changes, and closes once the job is done."""
    await ws.accept()
    last = None
    try:
        while True:
            status = await asyncio.to_thread(job_status, job_id)
            payload = status.model_dump(mode="json")
            if payload != last:
                await ws.send_json(payload)
                last = payload
            if status.status in ("SUCCESS", "FAILURE"):
                break
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
    await ws.close()


@app.websocket("/ws/trades/{symbol}")
async def ws_trades(
    ws: WebSocket,
    symbol: str,
    since: str = "$",
    limit: int | None = None,
    redis: aioredis.Redis = Depends(get_redis),
) -> None:
    """Live trades for one symbol, straight from the gateway's Redis Stream.

    `since` is a stream id to start after ("$" = only new trades, "0" = from the start);
    `limit` closes the socket after that many trades (snapshots, tests).
    """
    await ws.accept()
    stream = get_settings().trades_stream
    symbol = symbol.upper()
    last_id, sent = since, 0
    try:
        while limit is None or sent < limit:
            # BLOCK keeps one round trip per batch instead of polling
            resp = await redis.xread({stream: last_id}, count=200, block=1000)
            for _, entries in resp or []:
                for msg_id, fields in entries:
                    last_id = msg_id
                    try:
                        trade = parse_entry(fields)
                    except ValueError:
                        continue  # the consumer counts bad entries; the browser doesn't care
                    if trade.symbol != symbol:
                        continue
                    await ws.send_text(json.dumps(asdict(trade)))
                    sent += 1
                    if limit is not None and sent >= limit:
                        break
    except WebSocketDisconnect:
        return
    finally:
        await redis.aclose()
    await ws.close()
