import math

import polars as pl

from quantlab.stream.check import compare
from tests.conftest import make_bars


def test_identical_bars_match_exactly():
    bars = make_bars(4)
    c = compare(bars, bars)
    assert (c.ws_bars, c.rest_bars, c.matched, c.exact_ohlc) == (4, 4, 4, 4)
    assert c.close_diff_pct == 0 and c.volume_ratio == 1


def test_measures_differences_on_overlapping_minutes_only():
    rest = make_bars(4)
    # stream started late (missed the first bar) and saw a slightly different last price
    ws = rest.slice(1).with_columns(
        pl.when(pl.col("ts") == rest["ts"][3])
        .then(pl.col("close") * 1.001)
        .otherwise("close")
        .alias("close"),
        pl.col("volume") * 0.5,
    )
    c = compare(ws, rest)
    assert (c.matched, c.exact_ohlc) == (3, 2)
    assert math.isclose(c.close_diff_pct, 0.1 / 3, rel_tol=1e-6)
    assert c.volume_ratio == 0.5


def test_no_overlap():
    c = compare(make_bars(2), make_bars(2).with_columns(pl.col("ts").dt.offset_by("1d")))
    assert c.matched == 0 and math.isnan(c.close_diff_pct)
    assert "0 matched" in str(c)
