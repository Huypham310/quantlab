"""Smoke tests for the Prometheus endpoint: the metric names the Grafana dashboard queries
must exist, and the values must move when the API is used."""

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from quantlab.api import app as api
from quantlab.api import metrics
from quantlab.data.store import upsert_bars
from quantlab.paper.broker import PaperBroker
from quantlab.worker import celery_app
from tests.test_model import random_bars


@pytest.fixture
def client(engine, monkeypatch):
    celery_app.conf.update(
        task_always_eager=True,
        task_store_eager_result=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )
    # fresh broker per test so the gauges start from known values
    broker = PaperBroker()
    monkeypatch.setattr(api, "paper_broker", broker)
    api.app.dependency_overrides[api.get_engine] = lambda: engine
    api.app.dependency_overrides[api.get_broker] = lambda: broker
    with TestClient(api.app) as c:
        c.broker = broker
        yield c
    api.app.dependency_overrides.clear()


def scrape(client) -> dict[str, list]:
    """Parse /metrics into {sample name: [(labels, value), ...]}."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    out: dict[str, list] = {}
    for family in text_string_to_metric_families(resp.text):
        for s in family.samples:
            out.setdefault(s.name, []).append((s.labels, s.value))
    return out


def value(samples, name, **labels):
    for lab, v in samples[name]:
        if all(lab.get(k) == want for k, want in labels.items()):
            return v
    raise AssertionError(f"{name}{labels} not found")


def test_metrics_endpoint_exposes_expected_names(client):
    client.get("/health")
    samples = scrape(client)
    expected = [
        "quantlab_http_request_duration_seconds_bucket",
        "quantlab_http_request_duration_seconds_count",
        "quantlab_paper_equity_usd",
        "quantlab_paper_cash_usd",
        "quantlab_paper_open_orders",
        "quantlab_paper_killed",
        "quantlab_paper_orders",
        "quantlab_paper_limit_breaches",
    ]
    for name in expected:
        assert name in samples, f"missing {name}"


def test_request_latency_uses_route_template(client):
    count = "quantlab_http_request_duration_seconds_count"
    # the registry is process-wide, so compare against what other tests already recorded
    try:
        before = value(scrape(client), count, route="/bars/{symbol}", status="404")
    except AssertionError:
        before = 0
    client.get("/bars/BTC-USD")  # 404 here (empty db) but still a request
    client.get("/bars/DOGE-USD")
    client.get("/nope")
    samples = scrape(client)
    # two different symbols collapse into one series, otherwise cardinality explodes
    assert value(samples, count, route="/bars/{symbol}", status="404") == before + 2
    assert value(samples, count, route="unmatched", status="404") >= 1
    # the scrape itself must not be measured
    assert all(lab["route"] != "/metrics" for lab, _ in samples[count])


def test_paper_gauges_follow_the_broker(client):
    b = client.broker
    b.on_trade("BTC-USD", 100.0, 1)
    # 97 is inside the 5% price band and below the last price, so the order rests
    order = b.submit("BTC-USD", "buy", 1.0, "limit", 97.0)
    assert order.status == "open"
    s = scrape(client)
    assert value(s, "quantlab_paper_open_orders") == 1
    assert value(s, "quantlab_paper_killed") == 0
    assert value(s, "quantlab_paper_equity_usd") == pytest.approx(100_000.0)
    assert value(s, "quantlab_paper_orders", status="open") == 1

    b.cancel(order.id)
    b.killed = True
    s = scrape(client)
    assert value(s, "quantlab_paper_open_orders") == 0
    assert value(s, "quantlab_paper_killed") == 1
    assert value(s, "quantlab_paper_orders", status="cancelled") == 1


def test_job_counters(client, engine, monkeypatch):
    # the worker opens its own engine from the request's db url
    monkeypatch.setattr("quantlab.worker.settings.database_url", str(engine.url))
    upsert_bars(engine, random_bars(800, seed=1), "coinbase", "ETH-USD", "1h")
    ok = {"symbol": "ETH-USD", "strategy": "momentum", "params": {"lookback": 24}}
    assert client.post("/backtests", json=ok).status_code == 202
    assert client.post("/backtests", json={**ok, "symbol": "NOPE-USD"}).status_code == 202

    s = scrape(client)
    assert value(s, "quantlab_jobs_submitted_total", kind="backtest") >= 2
    # eager mode fires the same signals the real worker does
    assert value(s, "quantlab_jobs_total", task="quantlab.backtest", status="success") >= 1
    assert value(s, "quantlab_jobs_total", task="quantlab.backtest", status="failure") >= 1
    assert "quantlab_job_duration_seconds_bucket" in s


def test_consumer_stats_are_recorded():
    from quantlab.stream.consumer import Stats

    c = metrics.CONSUMER_COUNTERS["trades"]
    before = c._value.get()
    metrics.record_consumer_stats(Stats(trades=3, duplicates=1), open_bars=2)
    assert c._value.get() == before + 3
    assert metrics.CONSUMER_OPEN_BARS._value.get() == 2
