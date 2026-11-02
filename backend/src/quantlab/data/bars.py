"""Canonical OHLCV bar format shared by every data source, plus cleaning/validation."""

from dataclasses import dataclass

import polars as pl

# every source converts to exactly this schema before anything else touches the data
BAR_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# timeframe -> bar length in seconds
TIMEFRAMES: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


def empty_bars() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


@dataclass
class CleanReport:
    rows_in: int
    duplicates: int
    invalid: int
    gaps: int

    @property
    def rows_out(self) -> int:
        return self.rows_in - self.duplicates - self.invalid


def clean_bars(df: pl.DataFrame, timeframe: str) -> tuple[pl.DataFrame, CleanReport]:
    """Sort, drop duplicate timestamps and impossible candles, and count missing bars.

    Gaps are only counted, not filled: forward-filling prices would invent trades that never
    happened, and the backtester handles missing bars on its own.
    """
    rows_in = df.height
    df = df.select(list(BAR_SCHEMA)).cast(BAR_SCHEMA).sort("ts")

    deduped = df.unique(subset="ts", keep="last", maintain_order=True)
    duplicates = rows_in - deduped.height

    # high phải >= mọi giá khác, low phải <= mọi giá khác, volume không âm
    valid = (
        (pl.col("high") >= pl.max_horizontal("open", "close", "low"))
        & (pl.col("low") <= pl.min_horizontal("open", "close", "high"))
        & (pl.col("low") > 0)
        & (pl.col("volume") >= 0)
    )
    cleaned = deduped.filter(valid)
    invalid = deduped.height - cleaned.height

    gaps = 0
    if cleaned.height > 1:
        step = TIMEFRAMES[timeframe]
        diffs = cleaned["ts"].diff().dt.total_seconds().drop_nulls()
        # a diff of 3 bars means 2 bars are missing
        gaps = int(((diffs // step) - 1).clip(lower_bound=0).sum())

    return cleaned, CleanReport(rows_in, duplicates, invalid, gaps)
