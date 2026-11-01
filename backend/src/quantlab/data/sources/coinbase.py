"""Coinbase Exchange public candles API (no API key needed).

Docs: GET /products/{product_id}/candles -> [[time, low, high, open, close, volume], ...],
newest first, at most 300 candles per request.
"""

import time
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl

from quantlab.data.bars import BAR_SCHEMA, TIMEFRAMES, empty_bars

BASE_URL = "https://api.exchange.coinbase.com"
MAX_CANDLES = 300
GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}


def parse_candles(rows: list[list[float]]) -> pl.DataFrame:
    """Convert raw Coinbase rows to the canonical bar frame (oldest first)."""
    if not rows:
        return empty_bars()
    # chú ý thứ tự cột của Coinbase là low, high, open, close (không phải OHLC)
    df = pl.DataFrame(rows, schema=["ts", "low", "high", "open", "close", "volume"], orient="row")
    return (
        df.with_columns(
            pl.from_epoch(pl.col("ts").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC")
        )
        .select(list(BAR_SCHEMA))
        .cast(BAR_SCHEMA)
        .sort("ts")
    )


class CoinbaseClient:
    name = "coinbase"

    def __init__(self, client: httpx.Client | None = None, pause: float = 0.35):
        # Coinbase asks for a User-Agent; pause keeps us well under the public rate limit
        self.client = client or httpx.Client(
            base_url=BASE_URL, timeout=20, headers={"User-Agent": "quantlab/0.1"}
        )
        self.pause = pause

    def fetch_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Download [start, end) in windows of 300 candles."""
        step = TIMEFRAMES[timeframe]
        if step not in GRANULARITIES:
            raise ValueError(f"Coinbase has no {timeframe} candles")
        window = timedelta(seconds=step * MAX_CANDLES)

        frames = []
        cursor = start.astimezone(UTC)
        end = end.astimezone(UTC)
        while cursor < end:
            window_end = min(cursor + window, end)
            res = self.client.get(
                f"/products/{symbol}/candles",
                params={
                    "granularity": step,
                    "start": cursor.isoformat(),
                    "end": window_end.isoformat(),
                },
            )
            res.raise_for_status()
            frames.append(parse_candles(res.json()))
            cursor = window_end
            if cursor < end and self.pause:
                time.sleep(self.pause)

        if not frames:
            return empty_bars()
        df = pl.concat(frames)
        # end is exclusive, the API is inclusive on both sides
        return df.filter(pl.col("ts") < end).unique("ts", keep="last").sort("ts")
