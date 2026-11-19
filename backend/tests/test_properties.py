"""Property-based tests (Hypothesis) for the pieces that take arbitrary user/market input.

Unit tests pin down a few hand-picked cases; these throw random data at the invariants:
request schemas, bar cleaning, bar storage and backtest metrics.
"""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from quantlab.api.schemas import BacktestRequest, PaperOrderRequest, TrainRequest
from quantlab.backtest import CostModel, compute_metrics, run_vectorized
from quantlab.backtest.metrics import max_drawdown
from quantlab.data.bars import BAR_SCHEMA, TIMEFRAMES, clean_bars
from quantlab.data.store import init_db, load_bars, make_engine, upsert_bars

T0 = datetime(2026, 1, 1, tzinfo=UTC)

# ---------------------------------------------------------------------------------------------
# strategies (Hypothesis "strategies", not trading ones)
# ---------------------------------------------------------------------------------------------

# symbols the API would plausibly see; no whitespace or control chars
symbols = st.text(alphabet="abcXYZ0123-_", min_size=1, max_size=32)
positive = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)

# nến thô: các field độc lập nên nhiều nến vô lý
# one raw candle: fields are independent, so many of them are impossible (high < low, ...)
raw_candle = st.tuples(
    st.floats(-10, 1_000, allow_nan=False),  # open
    st.floats(-10, 1_000, allow_nan=False),  # high
    st.floats(-10, 1_000, allow_nan=False),  # low
    st.floats(-10, 1_000, allow_nan=False),  # close
    st.floats(-5, 1_000, allow_nan=False),  # volume
    st.integers(0, 40),  # hour offset, small range on purpose so duplicates happen
)


@st.composite
def valid_candle(draw):
    """A candle that satisfies low <= open, close <= high and low > 0."""
    low = draw(st.floats(0.01, 1_000, allow_nan=False))
    high = low * draw(st.floats(1.0, 3.0))
    open_ = draw(st.floats(low, high))
    close = draw(st.floats(low, high))
    volume = draw(st.floats(0, 1_000, allow_nan=False))
    return open_, high, low, close, volume


def frame(rows: list[tuple]) -> pl.DataFrame:
    """Build a canonical bar frame from (open, high, low, close, volume, hour) tuples."""
    data = [(T0 + timedelta(hours=h), o, hi, lo, c, v) for o, hi, lo, c, v, h in rows]
    return pl.DataFrame(data, schema=list(BAR_SCHEMA), orient="row").cast(BAR_SCHEMA)


# ---------------------------------------------------------------------------------------------
# request schemas
# ---------------------------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    symbol=symbols,
    side=st.sampled_from(["buy", "sell"]),
    qty=positive,
    kind=st.sampled_from(["market", "limit"]),
    limit_price=st.one_of(st.none(), positive),
)
def test_paper_order_limit_price_matches_order_type(symbol, side, qty, kind, limit_price):
    """Limit orders need a price, market orders must not carry one; nothing else slips through."""
    payload = {"symbol": symbol, "side": side, "qty": qty, "type": kind, "limit_price": limit_price}
    consistent = (kind == "limit") == (limit_price is not None)
    if consistent:
        req = PaperOrderRequest(**payload)
        assert req.symbol == symbol.upper()  # always normalized
        assert req.qty > 0
    else:
        with pytest.raises(ValidationError):
            PaperOrderRequest(**payload)


@settings(max_examples=100, deadline=None)
@given(
    qty=st.floats(allow_nan=True, allow_infinity=True),
    price=st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
)
def test_paper_order_rejects_non_positive_numbers(qty, price):
    """Zero, negative and NaN quantities / prices are 422s; inf is the only odd one accepted."""
    kind = "market" if price is None else "limit"
    # TODO: thử thêm qty cực nhỏ (denormal) xem pydantic xử lý thế nào
    ok = qty > 0 and (price is None or price > 0)  # NaN > 0 is False, so NaN is rejected too
    if ok:
        PaperOrderRequest(symbol="btc-usd", side="buy", qty=qty, type=kind, limit_price=price)
    else:
        with pytest.raises(ValidationError):
            PaperOrderRequest(symbol="btc-usd", side="buy", qty=qty, type=kind, limit_price=price)


@settings(max_examples=80, deadline=None)
@given(
    fee=st.floats(-50, 800, allow_nan=False),
    slip=st.floats(-50, 800, allow_nan=False),
    band=st.floats(-1, 2, allow_nan=False),
    start_h=st.integers(0, 1_000),
    end_h=st.integers(0, 1_000),
)
def test_backtest_request_enforces_bounds_and_date_order(fee, slip, band, start_h, end_h):
    """Cost/band limits and start < end are rejected in the schema, before any job exists."""
    payload = {
        "symbol": "BTC-USD",
        "strategy": "buy_and_hold",
        "fee_bps": fee,
        "slippage_bps": slip,
        "band": band,
        "start": T0 + timedelta(hours=start_h),
        "end": T0 + timedelta(hours=end_h),
    }
    valid = 0 <= fee <= 500 and 0 <= slip <= 500 and 0 <= band <= 1 and start_h < end_h
    if valid:
        req = BacktestRequest(**payload)
        assert req.start < req.end
    else:
        with pytest.raises(ValidationError):
            BacktestRequest(**payload)


@settings(max_examples=80, deadline=None)
@given(
    horizon=st.integers(-5, 400),
    train_days=st.integers(-5, 4_000),
    trials=st.integers(-5, 100),
    symbol=symbols,
)
def test_train_request_caps_resource_usage(horizon, train_days, trials, symbol):
    """The caps on horizon / window / trials are what stop one request hogging the worker."""
    payload = {"symbol": symbol, "horizon": horizon, "train_days": train_days, "trials": trials}
    valid = 1 <= horizon <= 168 and 7 <= train_days <= 3650 and 1 <= trials <= 50
    if valid:
        assert TrainRequest(**payload).symbol == symbol.upper()
    else:
        with pytest.raises(ValidationError):
            TrainRequest(**payload)


# ---------------------------------------------------------------------------------------------
# bar cleaning: OHLC invariants
# ---------------------------------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(rows=st.lists(raw_candle, min_size=1, max_size=40), tf=st.sampled_from(sorted(TIMEFRAMES)))
def test_clean_bars_output_is_always_a_valid_series(rows, tf):
    """Whatever garbage goes in, what comes out is sorted, unique and has sane OHLCV."""
    df, report = clean_bars(frame(rows), tf)
    # report phải tính đủ mọi dòng đầu vào
    # the report must account for every input row
    assert report.rows_in == len(rows)
    assert report.rows_out == df.height
    assert report.duplicates >= 0 and report.invalid >= 0 and report.gaps >= 0
    if df.is_empty():
        return
    ts = df["ts"]
    assert ts.is_sorted() and ts.n_unique() == df.height
    assert (df["high"] >= df.select(pl.max_horizontal("open", "close", "low"))["open"]).all()
    assert (df["low"] <= df.select(pl.min_horizontal("open", "close", "high"))["open"]).all()
    assert (df["low"] > 0).all() and (df["volume"] >= 0).all()


@settings(max_examples=100, deadline=None)
@given(rows=st.lists(raw_candle, min_size=1, max_size=40))
def test_clean_bars_is_idempotent(rows):
    """Cleaning already-clean bars changes nothing, so re-ingesting is safe."""
    once, _ = clean_bars(frame(rows), "1h")
    twice, report = clean_bars(once, "1h")
    assert twice.equals(once)
    assert report.duplicates == 0 and report.invalid == 0


# ---------------------------------------------------------------------------------------------
# bar store round trip
# ---------------------------------------------------------------------------------------------


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(candles=st.lists(valid_candle(), min_size=1, max_size=30))
def test_store_roundtrip_keeps_ohlc_invariants(candles):
    """Valid bars survive upsert -> load unchanged, and re-upserting is a no-op."""
    rows = [(*c, i) for i, c in enumerate(candles)]
    df, _ = clean_bars(frame(rows), "1h")
    assume(df.height > 0)
    # mỗi example một db in-memory mới để không dính dữ liệu của lần trước
    engine = make_engine("sqlite://")  # fresh in-memory db per example
    try:
        init_db(engine)
        upsert_bars(engine, df, "coinbase", "BTC-USD", "1h")
        upsert_bars(engine, df, "coinbase", "BTC-USD", "1h")  # twice on purpose
        loaded = load_bars(engine, "coinbase", "BTC-USD", "1h")
    finally:
        engine.dispose()
    assert loaded.equals(df)
    assert (loaded["high"] >= loaded["low"]).all()
    assert loaded["ts"].is_sorted()


# ---------------------------------------------------------------------------------------------
# backtest metrics
# ---------------------------------------------------------------------------------------------

# per-bar price ratios bounded away from 0 so a short position can't push equity below zero
ratios = st.lists(st.floats(0.7, 1.4), min_size=2, max_size=200)
weight_levels = st.sampled_from([-1.0, -0.5, 0.0, 0.5, 1.0])


@st.composite
def price_and_weights(draw):
    r = draw(ratios)
    close = 100.0 * np.cumprod(np.array(r))
    weights = np.array(draw(st.lists(weight_levels, min_size=len(r), max_size=len(r))))
    return close, weights


@settings(max_examples=100, deadline=None)
@given(
    data=price_and_weights(),
    fee=st.floats(0, 100),
    band=st.floats(0, 0.5),
    tf=st.sampled_from(["1h", "1d"]),
)
def test_metrics_are_finite_and_drawdown_is_bounded(data, fee, band, tf):
    """No NaN/inf in any metric and 0 <= max drawdown <= 1, for random prices and positions."""
    close, weights = data
    res = run_vectorized(close, weights, CostModel(fee, 0), 10_000, band)
    m = compute_metrics(res.equity, res.returns, weights, res.turnover, res.trades, tf, 10_000)
    values = m.as_dict()
    assert all(math.isfinite(v) for v in values.values()), values
    assert 0.0 <= m.max_drawdown <= 1.0
    assert 0.0 <= m.exposure <= 1.0
    assert m.volatility >= 0 and m.turnover >= 0 and m.trades >= 0
    assert m.total_return > -1.0  # the account can lose a lot but never goes negative here


@settings(max_examples=100, deadline=None)
@given(
    eq=st.lists(st.floats(1.0, 1e6), min_size=1, max_size=100),
    scale=st.floats(1e-3, 1e3),
)
def test_max_drawdown_ignores_scale_and_matches_a_naive_loop(eq, scale):
    """Drawdown is a ratio, so multiplying equity by a constant must not change it."""
    arr = np.array(eq)
    dd = max_drawdown(arr)
    assert 0.0 <= dd < 1.0
    assert dd == pytest.approx(max_drawdown(arr * scale), abs=1e-9)
    # so với vòng lặp python thuần (reference)
    # plain-python reference
    peak, worst = arr[0], 0.0
    for x in arr:
        peak = max(peak, x)
        worst = max(worst, 1 - x / peak)
    assert dd == pytest.approx(worst, abs=1e-12)


@settings(max_examples=60, deadline=None)
@given(n=st.integers(2, 150), cash=st.floats(1.0, 1e7), tf=st.sampled_from(["1h", "1d"]))
def test_flat_account_has_zero_everything(n, cash, tf):
    """No position: equity never moves, so every risk / return metric is exactly zero."""
    close = np.linspace(100, 150, n)
    zeros = np.zeros(n)
    res = run_vectorized(close, zeros, CostModel(), cash)
    m = compute_metrics(res.equity, res.returns, zeros, res.turnover, res.trades, tf, cash)
    assert m.total_return == pytest.approx(0.0, abs=1e-12)
    assert m.max_drawdown == 0.0 and m.volatility == 0.0 and m.sharpe == 0.0
    assert m.trades == 0 and m.exposure == 0.0
