import fakeredis
import pytest

from quantlab.data.store import load_bars
from quantlab.stream.aggregator import BAR_MS
from quantlab.stream.consumer import SOURCE, TradeConsumer

M0 = 1_793_556_000_000


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def xadd(r, tid, ts_ms, price, symbol="BTC-USD", size=1.0):
    """Write an entry exactly like the Go gateway does."""
    return r.xadd(
        "trades",
        {
            "exchange": "coinbase",
            "symbol": symbol,
            "trade_id": str(tid),
            "price": str(price),
            "size": str(size),
            "side": "buy",
            "ts_ms": str(ts_ms),
        },
    )


def make(r, engine, name="worker-1"):
    c = TradeConsumer(r, engine, consumer=name, grace_ms=0, block_ms=10)
    c.ensure_group()
    return c


def pending(r) -> int:
    return r.xpending("trades", "bars-1m")["pending"]


def test_writes_closed_bars_and_acks_their_trades(r, engine):
    for i, (off, price) in enumerate([(0, 100), (20_000, 104), (50_000, 101)]):
        xadd(r, i, M0 + off, price)
    xadd(r, 9, M0 + BAR_MS + 1, 102)  # next minute closes the first bar

    c = make(r, engine)
    stats = c.poll()
    assert (stats.trades, stats.bars_written) == (4, 1)

    bars = load_bars(engine, SOURCE, "BTC-USD", "1m")
    assert bars.select("open", "high", "low", "close", "volume").row(0) == (100, 104, 100, 101, 3)
    # the 3 trades of the written bar are acked, the open bar's trade is still pending
    assert pending(r) == 1


def test_bad_entries_are_dropped_and_acked(r, engine):
    r.xadd("trades", {"symbol": "BTC-USD", "price": "oops"})
    stats = make(r, engine).poll()
    assert stats.invalid == 1
    assert pending(r) == 0


def test_restart_replays_pending_trades_without_losing_or_doubling(r, engine):
    xadd(r, 1, M0, 100)
    xadd(r, 2, M0 + 30_000, 105)
    first = make(r, engine)
    first.poll()
    assert pending(r) == 2  # bar still open when the process "dies"

    # new process, same consumer name: replays its pending trades, then reads new ones
    xadd(r, 3, M0 + 59_000, 103)
    xadd(r, 4, M0 + BAR_MS + 1, 99)
    second = make(r, engine)
    while second.poll().messages:
        pass

    bars = load_bars(engine, SOURCE, "BTC-USD", "1m")
    assert bars.height == 1
    assert bars.select("open", "high", "close", "volume").row(0) == (100, 105, 103, 3)
    assert pending(r) == 1  # only the newest minute's trade


def test_pending_replay_moves_forward(r, engine):
    # more pending entries than one batch: replay must not re-read the same batch forever
    for i in range(25):
        xadd(r, i, M0 + i * 100, 100 + i)
    c = TradeConsumer(r, engine, grace_ms=0, batch=10, block_ms=10)
    c.ensure_group()
    c.poll()
    c.poll()
    c.poll()  # all 25 delivered, nothing closed yet -> all pending

    restarted = TradeConsumer(r, engine, grace_ms=0, batch=10, block_ms=10)
    seen = 0
    for _ in range(5):
        seen += restarted.poll().messages
    assert seen == 25


def test_writing_the_same_bar_twice_is_harmless(r, engine):
    # crash after the db write but before the ack -> the bar gets rebuilt and written again
    xadd(r, 1, M0, 100)
    xadd(r, 2, M0 + BAR_MS, 101)
    make(r, engine).poll()
    assert load_bars(engine, SOURCE, "BTC-USD", "1m").height == 1
    r.xgroup_destroy("trades", "bars-1m")  # forget the acks, like a crash would

    stats = make(r, engine).poll()
    assert stats.bars_written == 1
    bars = load_bars(engine, SOURCE, "BTC-USD", "1m")
    assert bars.height == 1 and bars["close"][0] == 100


def test_existing_group_is_reused(r, engine):
    make(r, engine)
    make(r, engine)  # BUSYGROUP is fine
    assert len(r.xinfo_groups("trades")) == 1
