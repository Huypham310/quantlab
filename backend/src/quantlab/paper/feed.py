"""Feeds the paper broker from the gateway's Redis Stream of trades."""

from __future__ import annotations

import logging

from redis import asyncio as aioredis

from quantlab.paper.broker import PaperBroker
from quantlab.stream.trades import parse_entry

log = logging.getLogger("quantlab.paper")


async def feed_trades(
    redis: aioredis.Redis,
    broker: PaperBroker,
    stream: str = "trades",
    since: str = "$",
    max_trades: int | None = None,
) -> int:
    """Reads trades forever (or `max_trades`) and hands each one to the broker.

    Single consumer, no group: the paper account lives in this process's memory, so after a
    restart there's nothing to resume and replaying old trades would fill orders on stale
    prices. Returns how many trades were applied.
    """
    last_id, applied = since, 0
    while max_trades is None or applied < max_trades:
        resp = await redis.xread({stream: last_id}, count=500, block=1000)
        for _, entries in resp or []:
            for msg_id, fields in entries:
                last_id = msg_id
                try:
                    t = parse_entry(fields)
                except ValueError:
                    continue
                if t.symbol not in broker.limits.symbols:
                    continue
                fills = broker.on_trade(t.symbol, t.price, t.ts_ms)
                for f in fills:
                    log.info("paper fill %s %s %.6f @ %.2f", f.side, f.symbol, f.qty, f.price)
                applied += 1
                if max_trades is not None and applied >= max_trades:
                    break
    return applied
