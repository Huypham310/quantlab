"""Prometheus metrics for the backend: HTTP latency, paper-account gauges, jobs, stream consumer.

All metrics live in one private registry (not the global default) so tests and `--reload`
can import this module without "duplicated timeseries" errors, and so /metrics only exposes
what we define here plus the process collector.

Who exposes what:
  - API process:      http_*, paper_*, jobs_submitted_total          -> GET /metrics on :8000
  - Celery worker:    jobs_total, job_duration_seconds               -> :9101 (start_metrics_server)
  - Stream consumer:  consumer_*                                     -> :9102 (--metrics-port)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    disable_created_metrics,
    generate_latest,
    start_http_server,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

# the *_created series just double the size of every scrape and nothing here uses them
disable_created_metrics()

# registry riêng thay vì registry mặc định: import lại không bị lỗi trùng tên metric
REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)

# API calls are mostly 5-50 ms (see the load test in the README), so the buckets are dense
# below 100 ms and still reach the multi-second tail.
LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

HTTP_LATENCY = Histogram(
    "quantlab_http_request_duration_seconds",
    "HTTP request latency by method, route template and status code",
    ["method", "route", "status"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

# ---- jobs ------------------------------------------------------------------------------------
JOBS_SUBMITTED = Counter(
    "quantlab_jobs_submitted_total",
    "Jobs queued through the API",
    ["kind"],
    registry=REGISTRY,
)
JOBS_DONE = Counter(
    "quantlab_jobs_total",
    "Celery tasks that finished, by task name and outcome",
    ["task", "status"],
    registry=REGISTRY,
)
JOB_DURATION = Histogram(
    "quantlab_job_duration_seconds",
    "Wall time of a Celery task (backtests are ms-seconds, training runs minutes)",
    ["task"],
    buckets=(0.1, 0.5, 1, 5, 15, 60, 180, 600, 1800),
    registry=REGISTRY,
)

# ---- stream consumer -------------------------------------------------------------------------
CONSUMER_COUNTERS = {
    name: Counter(f"quantlab_consumer_{name}_total", f"Consumer: {desc}", registry=REGISTRY)
    for name, desc in {
        "messages": "stream entries read",
        "trades": "trades added to a bar",
        "duplicates": "trades dropped as duplicates (reconnect replays)",
        "late": "trades dropped for arriving after their bar closed",
        "invalid": "malformed entries dropped",
        "bars_written": "1-minute bars written to the database",
    }.items()
}
CONSUMER_OPEN_BARS = Gauge(
    "quantlab_consumer_open_bars",
    "1-minute bars still being built (waiting for the grace period)",
    registry=REGISTRY,
)
CONSUMER_LAST_POLL = Gauge(
    "quantlab_consumer_last_poll_timestamp_seconds",
    "Unix time of the consumer's last poll",
    registry=REGISTRY,
)


def record_consumer_stats(stats, open_bars: int) -> None:
    """Add one poll's Stats (quantlab.stream.consumer.Stats) to the counters."""
    for name, counter in CONSUMER_COUNTERS.items():
        n = getattr(stats, name, 0)
        if n:
            counter.inc(n)
    CONSUMER_OPEN_BARS.set(open_bars)
    CONSUMER_LAST_POLL.set(time.time())


# ---- paper account ---------------------------------------------------------------------------
class PaperCollector(Collector):
    """Reads the broker when Prometheus scrapes, so the broker needs no metrics hooks at all
    (and the gauges can't drift from the real state)."""

    def __init__(self, get_broker: Callable[[], object]) -> None:
        self._get_broker = get_broker

    def collect(self) -> Iterable[GaugeMetricFamily]:
        broker = self._get_broker()  # lấy broker mới nhất mỗi lần scrape

        def gauge(name: str, doc: str, value: float) -> GaugeMetricFamily:
            g = GaugeMetricFamily(name, doc)
            g.add_metric([], float(value))
            return g

        yield gauge("quantlab_paper_equity_usd", "Paper equity (cash + positions)", broker.equity())
        yield gauge("quantlab_paper_cash_usd", "Paper account cash", broker.cash)
        yield gauge("quantlab_paper_open_orders", "Resting paper orders", len(broker.open))
        # 1 while the daily-loss kill switch is on
        yield gauge("quantlab_paper_killed", "1 if the kill switch is active", int(broker.killed))
        yield gauge("quantlab_paper_fees_paid_usd", "Fees paid so far", broker.fees_paid)
        # must stay 0: fills that pushed a position past its limit
        yield gauge("quantlab_paper_limit_breaches", "Position-limit breaches", broker.breaches)

        by_status = GaugeMetricFamily(
            "quantlab_paper_orders", "Paper orders by status", labels=["status"]
        )
        counts: dict[str, int] = {}
        for order in list(broker.orders.values()):  # list(): the broker mutates from other threads
            counts[order.status] = counts.get(order.status, 0) + 1
        for status in ("open", "filled", "cancelled", "rejected"):
            by_status.add_metric([status], counts.get(status, 0))
        yield by_status


def register_paper(get_broker: Callable[[], object]) -> None:
    REGISTRY.register(PaperCollector(get_broker))


# ---- HTTP middleware -------------------------------------------------------------------------
class MetricsMiddleware:
    """Pure ASGI middleware (BaseHTTPMiddleware would buffer and slow every request).

    Labels use the route *template* ("/bars/{symbol}"), not the raw path, otherwise every
    symbol or job id would create a new time series.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        # websockets are long-lived: a latency histogram makes no sense for them
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return

        status = 500  # if the app raises before sending a response, count it as a 500
        start = time.perf_counter()

        async def send_wrapper(message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # the router stores the matched route in the scope; 404s have none
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            HTTP_LATENCY.labels(scope["method"], path, str(status)).observe(
                time.perf_counter() - start
            )


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def start_metrics_server(port: int) -> None:
    """For processes without an HTTP server of their own (worker, consumer). 0 = disabled."""
    if port:
        start_http_server(port, registry=REGISTRY)
