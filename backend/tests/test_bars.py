from datetime import timedelta

import polars as pl

from quantlab.data.bars import clean_bars
from tests.conftest import T0, make_bars


def test_clean_keeps_good_bars():
    df, report = clean_bars(make_bars(5), "1h")
    assert df.height == 5
    assert (report.duplicates, report.invalid, report.gaps) == (0, 0, 0)


def test_clean_sorts_and_drops_duplicates():
    bars = make_bars(4)
    shuffled = pl.concat([bars[2:], bars[:2], bars[1:2]])
    df, report = clean_bars(shuffled, "1h")
    assert report.duplicates == 1
    assert df["ts"].is_sorted()
    assert df.height == 4


def test_clean_drops_impossible_candles():
    bars = make_bars(3).with_columns(
        # bar 1: high below close
        pl.when(pl.int_range(pl.len()) == 1).then(50.0).otherwise(pl.col("high")).alias("high")
    )
    df, report = clean_bars(bars, "1h")
    assert report.invalid == 1
    assert df.height == 2


def test_clean_counts_gaps():
    bars = make_bars(3)
    # remove the middle hour and append one 3 hours later -> 1 + 2 missing bars
    gapped = pl.concat([bars[:1], bars[2:], make_bars(1, start=T0 + timedelta(hours=5))])
    _, report = clean_bars(gapped, "1h")
    assert report.gaps == 3
