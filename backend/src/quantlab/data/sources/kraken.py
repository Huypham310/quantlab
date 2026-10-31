"""Kraken public OHLC API (no API key needed).

Returns at most the latest 720 candles per pair/interval, so it's only used for recent data and
to cross-check Coinbase prices, not for deep history.
"""

from datetime import UTC, datetime

import httpx
import polars as pl

from quantlab.data.bars import BAR_SCHEMA, TIMEFRAMES, empty_bars

BASE_URL = "https://api.kraken.com"
# Kraken intervals are in minutes
INTERVALS = {60: 1, 300: 5, 900: 15, 3600: 60, 21600: 240, 86400: 1440}
# Kraken gọi bitcoin là XBT
ASSET_ALIASES = {"BTC": "XBT"}


def to_kraken_pair(symbol: str) -> str:
    """'BTC-USD' -> 'XBTUSD'."""
    base, quote = symbol.split("-")
    return ASSET_ALIASES.get(base, base) + quote


def parse_ohlc(payload: dict) -> pl.DataFrame:
    """Convert a Kraken /OHLC response to the canonical bar frame."""
    if payload.get("error"):
        raise RuntimeError(f"Kraken error: {payload['error']}")
    result = payload["result"]
    # the pair key in the response isn't the one we asked for (XBTUSD -> XXBTZUSD), so take
    # whatever key isn't "last"
    rows = next(v for k, v in result.items() if k != "last")
    if not rows:
        return empty_bars()
    df = pl.DataFrame(
        rows,
        schema=["ts", "open", "high", "low", "close", "vwap", "volume", "count"],
        orient="row",
    )
    return (
        df.with_columns(pl.from_epoch(pl.col("ts").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC"))
        .select(list(BAR_SCHEMA))
        .cast(BAR_SCHEMA)  # prices come back as strings
        .sort("ts")
    )


class KrakenClient:
    name = "kraken"

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(base_url=BASE_URL, timeout=20)

    def fetch_bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pl.DataFrame:
        step = TIMEFRAMES[timeframe]
        if step not in INTERVALS:
            raise ValueError(f"Kraken has no {timeframe} candles")
        res = self.client.get(
            "/0/public/OHLC",
            params={
                "pair": to_kraken_pair(symbol),
                "interval": INTERVALS[step],
                "since": int(start.timestamp()),
            },
        )
        res.raise_for_status()
        df = parse_ohlc(res.json())
        # the newest candle is still forming, drop it along with anything outside the range
        df = df.head(max(df.height - 1, 0))
        return df.filter((pl.col("ts") >= start.astimezone(UTC)) & (pl.col("ts") < end.astimezone(UTC)))
