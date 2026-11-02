"""Builds 1-minute OHLCV bars from a trade stream, keyed on event time (the trade's own timestamp).

A bar for minute [t, t+60s) is only emitted once the newest trade seen is past t+60s+grace,
so trades that arrive a little out of order still land in the right bar. Using event time
instead of the wall clock also means replaying a backlog after a restart gives the same bars.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import polars as pl

from quantlab.data.bars import BAR_SCHEMA
from quantlab.stream.trades import Trade

BAR_MS = 60_000


class AddResult(Enum):
    ADDED = "added"
    DUPLICATE = "duplicate"  # same trade_id again, e.g. Coinbase's last_match after a reconnect
    LATE = "late"  # its bar was already written


@dataclass
class OpenBar:
    symbol: str
    start_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    # (ts_ms, trade_id) of the trades that set open/close, so out-of-order trades sort correctly
    first: tuple[int, int]
    last: tuple[int, int]
    trade_ids: set[int] = field(default_factory=set)
    # stream ids to ack once this bar is safely in the database
    msg_ids: list[str] = field(default_factory=list)

    @property
    def ts(self) -> datetime:
        return datetime.fromtimestamp(self.start_ms / 1000, tz=UTC)

    def add(self, t: Trade) -> None:
        key = (t.ts_ms, t.trade_id)
        if key < self.first:
            self.first, self.open = key, t.price
        if key > self.last:
            self.last, self.close = key, t.price
        self.high = max(self.high, t.price)
        self.low = min(self.low, t.price)
        self.volume += t.size
        self.trade_ids.add(t.trade_id)


class MinuteBarAggregator:
    def __init__(self, grace_ms: int = 5_000) -> None:
        self.grace_ms = grace_ms
        self.watermark_ms = 0  # newest trade time seen so far
        self._open: dict[tuple[str, int], OpenBar] = {}
        # newest bar start already emitted per symbol; anything at or before it is late
        self._emitted_until: dict[str, int] = {}

    def add(self, trade: Trade, msg_id: str) -> AddResult:
        start = trade.ts_ms - trade.ts_ms % BAR_MS
        if start <= self._emitted_until.get(trade.symbol, -1):
            return AddResult.LATE

        self.watermark_ms = max(self.watermark_ms, trade.ts_ms)
        bar = self._open.get((trade.symbol, start))
        if bar is None:
            key = (trade.ts_ms, trade.trade_id)
            bar = OpenBar(
                trade.symbol,
                start,
                trade.price,
                trade.price,
                trade.price,
                trade.price,
                0.0,
                key,
                key,
            )
            self._open[(trade.symbol, start)] = bar
        elif trade.trade_id in bar.trade_ids:
            # vẫn giữ msg id để ack cùng bar, không thì message nằm pending mãi
            bar.msg_ids.append(msg_id)
            return AddResult.DUPLICATE

        bar.add(trade)
        bar.msg_ids.append(msg_id)
        return AddResult.ADDED

    def pop_closed(self) -> list[OpenBar]:
        """Remove and return bars whose minute ended more than `grace_ms` before the watermark."""
        cutoff = self.watermark_ms - self.grace_ms - BAR_MS
        done = [b for b in self._open.values() if b.start_ms <= cutoff]
        for b in sorted(done, key=lambda b: b.start_ms):
            del self._open[(b.symbol, b.start_ms)]
            self._emitted_until[b.symbol] = max(self._emitted_until.get(b.symbol, -1), b.start_ms)
        return sorted(done, key=lambda b: (b.symbol, b.start_ms))

    @property
    def open_bars(self) -> int:
        return len(self._open)


def bars_frame(bars: list[OpenBar]) -> pl.DataFrame:
    """Closed bars of ONE symbol as the canonical bar frame used by the store."""
    rows = [(b.ts, b.open, b.high, b.low, b.close, b.volume) for b in bars]
    return pl.DataFrame(rows, schema=list(BAR_SCHEMA), orient="row").cast(BAR_SCHEMA)
