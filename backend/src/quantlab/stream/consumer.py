"""Consumes the trades stream with a Redis consumer group and writes 1-minute bars.

Delivery is at-least-once: a message is only acked after the bar it belongs to has been
written. If the process dies, its unacked messages stay in the group's pending list and get
replayed on restart; writing the same bar twice is a no-op (upsert on the bar's key).

    python -m quantlab.stream.consumer --consumer worker-1
"""

import argparse
import logging
import signal
import time
from collections import defaultdict
from dataclasses import dataclass

import redis
from sqlalchemy.engine import Engine

from quantlab.config import get_settings
from quantlab.data.store import init_db, make_engine, upsert_bars
from quantlab.stream.aggregator import AddResult, MinuteBarAggregator, OpenBar, bars_frame
from quantlab.stream.trades import parse_entry

log = logging.getLogger("quantlab.stream")

# live bars get their own source so they can be compared with Coinbase's REST candles
SOURCE = "coinbase-ws"
TIMEFRAME = "1m"


@dataclass
class Stats:
    messages: int = 0
    trades: int = 0
    duplicates: int = 0
    late: int = 0
    invalid: int = 0
    bars_written: int = 0

    def __iadd__(self, other: "Stats") -> "Stats":
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self


class TradeConsumer:
    def __init__(
        self,
        client: redis.Redis,
        engine: Engine,
        stream: str = "trades",
        group: str = "bars-1m",
        consumer: str = "worker-1",
        grace_ms: int = 5_000,
        batch: int = 1_000,
        block_ms: int = 2_000,
    ) -> None:
        self.r = client
        self.engine = engine
        self.stream, self.group, self.consumer = stream, group, consumer
        self.batch, self.block_ms = batch, block_ms
        self.agg = MinuteBarAggregator(grace_ms)
        # start by replaying our own pending messages (left over from a crash), then new ones
        self._pending_cursor: str | None = "0"

    def ensure_group(self) -> None:
        try:
            # id "0": a brand new group also picks up trades written before the consumer started
            self.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except redis.ResponseError as err:
            if "BUSYGROUP" not in str(err):
                raise

    def _read(self) -> list[tuple[str, dict]]:
        if self._pending_cursor is not None:
            resp = self.r.xreadgroup(
                self.group, self.consumer, {self.stream: self._pending_cursor}, count=self.batch
            )
            entries = resp[0][1] if resp else []
            if not entries:
                self._pending_cursor = None  # caught up, switch to new messages
                return self._read()
            # reading pending from "0" returns the same entries every time; move the cursor on
            self._pending_cursor = _id(entries[-1][0])
            return [(_id(i), f) for i, f in entries]

        resp = self.r.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=self.batch, block=self.block_ms
        )
        return [(_id(i), f) for i, f in resp[0][1]] if resp else []

    def poll(self) -> Stats:
        """Read one batch, feed the aggregator, write finished bars, ack what's safe."""
        stats = Stats()
        ack_now: list[str] = []
        for msg_id, fields in self._read():
            stats.messages += 1
            try:
                trade = parse_entry(fields)
            except (ValueError, KeyError) as err:
                # bad entries would sit in pending forever; log and drop them
                log.warning("dropping bad entry %s: %s", msg_id, err)
                stats.invalid += 1
                ack_now.append(msg_id)
                continue
            result = self.agg.add(trade, msg_id)
            if result is AddResult.ADDED:
                stats.trades += 1
            elif result is AddResult.DUPLICATE:
                stats.duplicates += 1
            else:
                stats.late += 1
                ack_now.append(msg_id)

        closed = self.agg.pop_closed()
        if closed:
            stats.bars_written = self._write(closed)
            ack_now.extend(m for b in closed for m in b.msg_ids)
        if ack_now:
            # only after the write above succeeded
            self.r.xack(self.stream, self.group, *ack_now)
        return stats

    def _write(self, closed: list[OpenBar]) -> int:
        by_symbol: dict[str, list[OpenBar]] = defaultdict(list)
        for b in closed:
            by_symbol[b.symbol].append(b)
        for symbol, bars in by_symbol.items():
            upsert_bars(self.engine, bars_frame(bars), SOURCE, symbol, TIMEFRAME)
        return len(closed)

    def run(self, should_stop=lambda: False, report_every: float = 60.0) -> Stats:
        self.ensure_group()
        total, last_report = Stats(), time.monotonic()
        while not should_stop():
            total += self.poll()
            if time.monotonic() - last_report >= report_every:
                log.info(
                    "trades=%d dup=%d late=%d invalid=%d bars=%d open=%d",
                    total.trades,
                    total.duplicates,
                    total.late,
                    total.invalid,
                    total.bars_written,
                    self.agg.open_bars,
                )
                last_report = time.monotonic()
        # open bars are NOT written on shutdown: their trades stay pending and get replayed
        return total


def _id(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else v


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Trades stream -> 1m bars")
    p.add_argument("--redis", default=settings.redis_url)
    p.add_argument("--db", default=settings.database_url)
    p.add_argument("--stream", default=settings.trades_stream)
    p.add_argument("--group", default="bars-1m")
    p.add_argument("--consumer", default="worker-1")
    p.add_argument("--grace-ms", type=int, default=5_000)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = make_engine(args.db)
    init_db(engine)
    consumer = TradeConsumer(
        redis.Redis.from_url(args.redis),
        engine,
        args.stream,
        args.group,
        args.consumer,
        grace_ms=args.grace_ms,
    )

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        log.info("stopping after the current batch")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    total = consumer.run(lambda: stopping)
    log.info("stopped: %d trades, %d bars written", total.trades, total.bars_written)


if __name__ == "__main__":
    main()
