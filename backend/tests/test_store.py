from datetime import timedelta

import polars as pl

from quantlab.data import ingest as ingest_module
from quantlab.data.store import latest_ts, load_bars, upsert_bars
from tests.conftest import T0, make_bars


def test_upsert_then_load_roundtrip(engine):
    bars = make_bars(5)
    upsert_bars(engine, bars, "coinbase", "BTC-USD", "1h")
    loaded = load_bars(engine, "coinbase", "BTC-USD", "1h")
    assert loaded.equals(bars)


def test_upsert_is_idempotent(engine):
    bars = make_bars(5)
    upsert_bars(engine, bars, "coinbase", "BTC-USD", "1h")
    upsert_bars(engine, bars, "coinbase", "BTC-USD", "1h")
    assert load_bars(engine, "coinbase", "BTC-USD", "1h").height == 5


def test_symbols_and_sources_are_separate(engine):
    upsert_bars(engine, make_bars(3), "coinbase", "BTC-USD", "1h")
    upsert_bars(engine, make_bars(2), "kraken", "BTC-USD", "1h")
    upsert_bars(engine, make_bars(4), "coinbase", "ETH-USD", "1h")
    assert load_bars(engine, "kraken", "BTC-USD", "1h").height == 2
    assert load_bars(engine, "coinbase", "ETH-USD", "1h").height == 4


def test_load_range_is_half_open(engine):
    upsert_bars(engine, make_bars(10), "coinbase", "BTC-USD", "1h")
    df = load_bars(
        engine, "coinbase", "BTC-USD", "1h", T0 + timedelta(hours=2), T0 + timedelta(hours=5)
    )
    assert df["ts"].to_list() == [T0 + timedelta(hours=h) for h in (2, 3, 4)]


def test_latest_ts(engine):
    assert latest_ts(engine, "coinbase", "BTC-USD", "1h") is None
    upsert_bars(engine, make_bars(4), "coinbase", "BTC-USD", "1h")
    assert latest_ts(engine, "coinbase", "BTC-USD", "1h") == T0 + timedelta(hours=3)


class FakeClient:
    """Returns hourly bars for whatever range is asked, and records the requests."""

    requests: list = []

    def fetch_bars(self, symbol, timeframe, start, end):
        FakeClient.requests.append((symbol, start, end))
        hours = int((end - start).total_seconds() // 3600)
        return make_bars(hours, start=start)


def test_ingest_is_incremental(engine, monkeypatch):
    monkeypatch.setitem(ingest_module.SOURCES, "fake", FakeClient)
    FakeClient.requests = []

    first = ingest_module.ingest(engine, "fake", ["BTC-USD"], "1h", days=2)
    assert first["BTC-USD"] == 48

    second = ingest_module.ingest(engine, "fake", ["BTC-USD"], "1h", days=2)
    # already up to date -> no new request (one more at most if an hour boundary just passed)
    assert second["BTC-USD"] <= 1
    assert len(FakeClient.requests) <= 2
    stored = load_bars(engine, "fake", "BTC-USD", "1h")
    assert stored["ts"].is_unique().all()
    assert isinstance(stored, pl.DataFrame)
