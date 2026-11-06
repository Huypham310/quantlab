from datetime import UTC, datetime, timedelta

import fakeredis
import pytest
from fastapi.testclient import TestClient

from quantlab.api import app as api
from quantlab.api.jobs import backtest_job, downsample, validate_strategy
from quantlab.data.store import upsert_bars
from quantlab.worker import celery_app
from tests.conftest import make_bars
from tests.test_model import random_bars


@pytest.fixture
def client(engine, monkeypatch):
    # run jobs inline and keep their results in memory, no broker / redis needed
    celery_app.conf.update(
        task_always_eager=True,
        task_store_eager_result=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )
    upsert_bars(engine, random_bars(800, seed=1), "coinbase", "ETH-USD", "1h")
    upsert_bars(engine, make_bars(30), "coinbase", "BTC-USD", "1h")
    api.app.dependency_overrides[api.get_engine] = lambda: engine
    api._symbols_cache.clear()  # engine ids can be reused between tests
    # the worker opens its own engine from the request's db url
    monkeypatch.setattr("quantlab.worker.settings.database_url", str(engine.url))
    with TestClient(api.app) as c:
        yield c
    api.app.dependency_overrides.clear()


def test_health_and_symbols(client):
    assert client.get("/health").json() == {"status": "ok", "database": True}
    rows = client.get("/symbols").json()
    assert [(r["symbol"], r["bars"]) for r in rows] == [("BTC-USD", 30), ("ETH-USD", 800)]
    assert rows[0]["first"].endswith(("Z", "+00:00"))


def test_symbols_are_cached_briefly(client, engine, monkeypatch):
    first = client.get("/symbols").json()
    upsert_bars(engine, make_bars(5), "coinbase", "SOL-USD", "1h")
    assert client.get("/symbols").json() == first  # still cached
    monkeypatch.setattr(api, "SYMBOLS_TTL_S", 0.0)
    assert "SOL-USD" in [r["symbol"] for r in client.get("/symbols").json()]


def test_bars_are_the_newest_n_oldest_first(client):
    bars = client.get("/bars/eth-usd", params={"limit": 5}).json()
    assert len(bars) == 5
    ts = [b["ts"] for b in bars]
    assert ts == sorted(ts)
    every = client.get("/bars/ETH-USD", params={"limit": 5000}).json()
    assert len(every) == 800 and every[-1] == bars[-1]
    assert client.get("/bars/DOGE-USD").status_code == 404
    assert client.get("/bars/ETH-USD", params={"limit": 0}).status_code == 422


def test_strategies_list_their_params(client):
    rows = {r["name"]: r for r in client.get("/strategies").json()}
    assert rows["momentum"]["params"] == ["lookback", "short"]
    assert "buy_and_hold" in rows and rows["sma_cross"]["grid"]["fast"] == [12, 24, 48]


def test_backtest_job_round_trip(client):
    r = client.post(
        "/backtests",
        json={"symbol": "ETH-USD", "strategy": "momentum", "params": {"lookback": 24}, "band": 0.1},
    )
    assert r.status_code == 202
    job = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "SUCCESS"
    res = job["result"]
    assert res["bars"] == 800 and len(res["equity"]) <= 500
    assert set(res["metrics"]) >= {"sharpe", "max_drawdown", "trades"}


def test_bad_backtest_requests_are_422(client):
    for body in (
        {"symbol": "ETH-USD", "strategy": "astrology"},
        {"symbol": "ETH-USD", "strategy": "momentum", "params": {"lookback": 24, "x": 1}},
        {"symbol": "ETH-USD", "strategy": "momentum"},  # lookback is required
        {"symbol": "ETH-USD", "strategy": "buy_and_hold", "band": 2},
        {"symbol": "ETH-USD", "strategy": "buy_and_hold", "timeframe": "5m"},
    ):
        assert client.post("/backtests", json=body).status_code == 422, body


def test_failed_job_reports_the_error(client):
    r = client.post("/backtests", json={"symbol": "NOPE-USD", "strategy": "buy_and_hold"})
    job = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "FAILURE"
    assert "not enough bars" in job["error"]


def test_training_job_and_its_websocket(client):
    r = client.post(
        "/models/train",
        json={"symbol": "eth-usd", "model": "lgbm", "train_days": 10, "test_days": 5, "trials": 1},
    )
    job_id = r.json()["job_id"]
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        msg = ws.receive_json()
    assert msg["status"] == "SUCCESS"
    report = msg["result"]
    # 800 bars - 169 warm-up - 240 train, in 120-bar test windows
    assert len(report["folds"]) == 4
    assert report["backtest"]["buy_and_hold"]["trades"] == 1


def test_trades_websocket_filters_by_symbol(client):
    server = fakeredis.FakeServer()
    sync = fakeredis.FakeRedis(server=server)
    now = int(datetime.now(UTC).timestamp() * 1000)
    for i, sym in enumerate(["BTC-USD", "ETH-USD", "BTC-USD", "BTC-USD"]):
        sync.xadd(
            "trades",
            {"exchange": "coinbase", "symbol": sym, "trade_id": i, "price": 100 + i,
             "size": 0.5, "side": "buy", "ts_ms": now + i},
        )  # fmt: skip
    sync.xadd("trades", {"symbol": "BTC-USD", "garbage": "1"})  # skipped, not fatal
    api.app.dependency_overrides[api.get_redis] = lambda: fakeredis.aioredis.FakeRedis(
        server=server
    )
    with client.websocket_connect("/ws/trades/btc-usd?since=0&limit=3") as ws:
        got = [ws.receive_json() for _ in range(3)]
    assert [t["trade_id"] for t in got] == [0, 2, 3]
    assert got[0]["price"] == 100.0 and got[0]["side"] == "buy"


def test_loadtest_against_the_app_in_process(client):
    import asyncio

    import httpx

    from quantlab.api import loadtest

    # the fixture has both symbols the dashboard mix asks for
    r = asyncio.run(loadtest.run("http://test", 3, 4, transport=httpx.ASGITransport(app=api.app)))
    assert r["requests"] == 12 and r["errors"] == 0
    assert 0 < r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"]


def test_job_helpers():
    validate_strategy("buy_and_hold", {})
    with pytest.raises(ValueError):
        validate_strategy("buy_and_hold", {"lookback": 3})
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ts = [t0 + timedelta(hours=i) for i in range(2000)]
    import numpy as np

    pts = downsample(ts, np.arange(2000.0), 100)
    assert len(pts) == 100 and pts[-1]["value"] == 1999.0
    res = backtest_job(make_bars(50), {"strategy": "buy_and_hold"})
    assert res["metrics"]["total_return"] > 0  # make_bars only goes up
