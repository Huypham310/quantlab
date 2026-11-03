"""Compares the 1m bars built from the live stream with Coinbase's own 1m candles (REST).

If the gateway drops trades or the aggregator puts them in the wrong minute, it shows up here.

    python -m quantlab.stream.check --symbol BTC-USD
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta

import polars as pl

from quantlab.config import get_settings
from quantlab.data.sources.coinbase import CoinbaseClient
from quantlab.data.store import load_bars, make_engine
from quantlab.stream.consumer import SOURCE, TIMEFRAME


@dataclass
class Comparison:
    ws_bars: int
    rest_bars: int
    matched: int
    exact_ohlc: int  # all four prices identical
    close_diff_pct: float  # mean |ws - rest| / rest * 100 over matched bars
    volume_ratio: float  # median ws volume / rest volume

    def __str__(self) -> str:
        return (
            f"bars: {self.ws_bars} from the stream, {self.rest_bars} from REST, "
            f"{self.matched} matched\n"
            f"identical OHLC: {self.exact_ohlc}/{self.matched}\n"
            f"mean close diff: {self.close_diff_pct:.5f}%\n"
            f"median volume ratio (stream/REST): {self.volume_ratio:.4f}"
        )


def compare(ws: pl.DataFrame, rest: pl.DataFrame) -> Comparison:
    joined = ws.join(rest, on="ts", suffix="_rest")
    if joined.is_empty():
        return Comparison(ws.height, rest.height, 0, 0, float("nan"), float("nan"))
    same = (
        (pl.col("open") == pl.col("open_rest"))
        & (pl.col("high") == pl.col("high_rest"))
        & (pl.col("low") == pl.col("low_rest"))
        & (pl.col("close") == pl.col("close_rest"))
    )
    stats = joined.select(
        same.sum().alias("exact"),
        ((pl.col("close") - pl.col("close_rest")).abs() / pl.col("close_rest")).mean().alias("cd"),
        (pl.col("volume") / pl.col("volume_rest")).median().alias("vr"),
    ).row(0)
    return Comparison(
        ws.height, rest.height, joined.height, int(stats[0]), stats[1] * 100, stats[2]
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stream bars vs Coinbase REST candles")
    p.add_argument("--symbol", default="BTC-USD")
    p.add_argument("--db", default=get_settings().database_url)
    args = p.parse_args(argv)

    ws = load_bars(make_engine(args.db), SOURCE, args.symbol, TIMEFRAME)
    if ws.is_empty():
        raise SystemExit("no stream bars yet, run the gateway + consumer first")
    start, end = ws["ts"].min(), ws["ts"].max() + timedelta(minutes=1)
    rest = CoinbaseClient().fetch_bars(args.symbol, TIMEFRAME, start, end)
    print(compare(ws, rest))


if __name__ == "__main__":
    main()
