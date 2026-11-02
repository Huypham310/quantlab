from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from quantlab.data.bars import BAR_SCHEMA
from quantlab.stream.aggregator import BAR_MS, AddResult, MinuteBarAggregator, bars_frame
from quantlab.stream.trades import Trade

M0 = 1_793_556_000_000  # a minute boundary (2026-11-01 18:00 UTC)


def trade(tid: int, ts_ms: int, price: float, size: float = 1.0, symbol: str = "BTC-USD"):
    return Trade("coinbase", symbol, tid, price, size, "buy", ts_ms)


def feed(agg: MinuteBarAggregator, trades: list[Trade]) -> list[AddResult]:
    return [agg.add(t, f"{i}-0") for i, t in enumerate(trades)]


def test_builds_ohlcv_by_trade_time_not_arrival_order():
    agg = MinuteBarAggregator(grace_ms=0)
    feed(
        agg,
        [
            trade(3, M0 + 30_000, 105),  # arrives first but isn't the first trade
            trade(1, M0 + 1_000, 100),
            trade(4, M0 + 59_999, 102),
            trade(2, M0 + 10_000, 95, size=2.5),
            trade(5, M0 + BAR_MS, 110),  # next minute, pushes the watermark past the 1st bar
        ],
    )
    [bar] = agg.pop_closed()
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (100, 105, 95, 102, 5.5)
    assert bar.ts == datetime(2026, 11, 1, 18, 0, tzinfo=UTC)
    assert bar.msg_ids == ["0-0", "1-0", "2-0", "3-0"]


def test_same_timestamp_trades_are_ordered_by_trade_id():
    agg = MinuteBarAggregator(grace_ms=0)
    feed(agg, [trade(8, M0, 101), trade(7, M0, 100), trade(9, M0 + BAR_MS, 1)])
    [bar] = agg.pop_closed()
    assert (bar.open, bar.close) == (100, 101)


def test_waits_for_the_grace_period():
    agg = MinuteBarAggregator(grace_ms=5_000)
    feed(agg, [trade(1, M0, 100), trade(2, M0 + BAR_MS + 4_999, 101)])
    assert agg.pop_closed() == []
    # a trade from the old minute can still arrive during the grace period
    assert agg.add(trade(3, M0 + 59_000, 99), "x") is AddResult.ADDED
    agg.add(trade(4, M0 + BAR_MS + 5_000, 101), "y")
    [bar] = agg.pop_closed()
    assert bar.close == 99 and bar.volume == 2


def test_duplicates_are_counted_once_but_still_acked():
    agg = MinuteBarAggregator(grace_ms=0)
    results = feed(agg, [trade(1, M0, 100), trade(1, M0, 100), trade(2, M0 + BAR_MS, 1)])
    assert results[1] is AddResult.DUPLICATE
    [bar] = agg.pop_closed()
    assert bar.volume == 1
    assert bar.msg_ids == ["0-0", "1-0"]


def test_trades_for_an_already_written_bar_are_late():
    agg = MinuteBarAggregator(grace_ms=0)
    feed(agg, [trade(1, M0, 100), trade(2, M0 + BAR_MS, 101)])
    agg.pop_closed()
    assert agg.add(trade(3, M0 + 30_000, 50), "z") is AddResult.LATE
    # older than the last written bar counts as late too, even if that minute was empty
    assert agg.add(trade(4, M0 - BAR_MS, 50), "w") is AddResult.LATE


def test_symbols_are_separate():
    agg = MinuteBarAggregator(grace_ms=0)
    feed(
        agg,
        [
            trade(1, M0, 100),
            trade(1, M0 + 5, 2000, symbol="ETH-USD"),  # same trade id on another product
            trade(2, M0 + BAR_MS, 101),
        ],
    )
    bars = agg.pop_closed()
    assert [(b.symbol, b.open) for b in bars] == [("BTC-USD", 100), ("ETH-USD", 2000)]


def test_bars_frame_matches_the_store_schema():
    agg = MinuteBarAggregator(grace_ms=0)
    feed(agg, [trade(1, M0, 100), trade(2, M0 + BAR_MS, 101), trade(3, M0 + 2 * BAR_MS, 1)])
    df = bars_frame(agg.pop_closed())
    assert df.schema == BAR_SCHEMA
    assert df["close"].to_list() == [100, 101]


# any arrival order of the same trades must give the same bars
trades_st = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=3 * BAR_MS - 1),  # offset inside 3 minutes
        st.floats(min_value=1, max_value=1e5, allow_nan=False),
        st.floats(min_value=1e-6, max_value=10, allow_nan=False),
    ),
    min_size=1,
    max_size=60,
)


@given(trades_st, st.randoms(use_true_random=False))
def test_order_of_arrival_doesnt_matter(raw, rnd):
    trades = [trade(i, M0 + off, p, s) for i, (off, p, s) in enumerate(raw)]
    closer = trade(10_000, M0 + 10 * BAR_MS, 1)

    def run(ts):
        agg = MinuteBarAggregator(grace_ms=0)
        feed(agg, [*ts, closer])
        return [
            (b.start_ms, b.open, b.high, b.low, b.close, round(b.volume, 9))
            for b in agg.pop_closed()
        ]

    shuffled = trades[:]
    rnd.shuffle(shuffled)
    assert run(trades) == run(shuffled)
