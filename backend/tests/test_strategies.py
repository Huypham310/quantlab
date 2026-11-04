import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.backtest import CostModel, run_event_driven, run_vectorized
from quantlab.backtest.strategies import (
    LIBRARY,
    donchian,
    momentum,
    rolling_std,
    vol_target,
    zscore_reversion,
)
from quantlab.backtest.vectorized import simulate_reference
from quantlab.backtest.walkforward import grid, make_weights, sharpe, walk_forward
from tests.test_backtest import gbm


@pytest.mark.parametrize("name", list(LIBRARY))
def test_no_strategy_looks_ahead(name):
    fn, params = LIBRARY[name]
    close = gbm(2_000, seed=3)
    shocked = close.copy()
    shocked[1_200:] *= 0.3  # a crash that hasn't happened yet at bar 1199
    for p in grid(params):
        a, b = fn(close, **p), fn(shocked, **p)
        assert np.array_equal(a[:1_200], b[:1_200]), p
        assert np.all(np.abs(a) <= 1.0)


def test_momentum_follows_the_trend():
    up = np.linspace(100, 200, 50)
    assert np.all(momentum(up, 10)[10:] == 1.0) and np.all(momentum(up, 10)[:10] == 0.0)
    down = up[::-1]
    assert np.all(momentum(down, 10)[10:] == 0.0)
    assert np.all(momentum(down, 10, short=True)[10:] == -1.0)


def test_donchian_enters_on_a_new_high_and_holds():
    close = np.array([10, 11, 10, 11, 10, 12, 11.5, 11.8, 9.0, 9.5])
    w = donchian(close, window=4)
    assert w[:5].tolist() == [0, 0, 0, 0, 0]
    assert w[5] == 1.0  # 12 breaks the 4-bar high of 11
    assert w[6] == w[7] == 1.0  # holds inside the channel
    assert w[8] == 0.0  # 9 breaks the low


def test_zscore_reversion_buys_the_dip_and_exits_at_the_mean():
    close = np.concatenate([np.full(20, 100.0) + np.tile([0.5, -0.5], 10), [95.0, 97.0, 100.5]])
    w = zscore_reversion(close, window=20, entry=2.0)
    assert w[20] == 1.0  # far below the mean
    assert w[21] == 1.0  # still below
    assert w[22] == 0.0  # back above the mean


def test_rolling_std_matches_numpy():
    x = np.random.default_rng(0).normal(size=50)
    assert np.isnan(rolling_std(x, 10)[:9]).all()
    assert rolling_std(x, 10)[20] == pytest.approx(np.std(x[11:21], ddof=1))


def test_vol_target_scales_down_volatile_assets_and_caps_leverage():
    calm, wild = gbm(1_000, seed=1, vol=0.002), gbm(1_000, seed=1, vol=0.02)
    kw = {"target": 0.3, "window": 48, "periods_per_year": 8760, "max_leverage": 1.0}
    w_calm = vol_target(np.ones(1_000), calm, **kw)
    w_wild = vol_target(np.ones(1_000), wild, **kw)
    assert np.all(w_calm[:48] == 0) and np.all(w_wild[:48] == 0)  # no estimate yet
    assert np.all(w_calm[100:] == 1.0)  # calm asset: capped at 1x
    assert np.all(w_wild[100:] < 0.5)  # ~1.9 annualized vol -> ~0.16x


def test_band_cuts_trades_and_engines_still_agree():
    close = gbm(3_000, seed=8)
    w = vol_target(np.ones(3_000), close, 0.3, 48, 8760, 1.0)
    costs = CostModel(10, 5)
    no_band = run_vectorized(close, w, costs)
    banded = run_vectorized(close, w, costs, band=0.1)
    assert banded.trades < no_band.trades / 10
    # the band lets the position drift, so final equity can go either way; costs can't
    assert banded.costs.sum() < no_band.costs.sum() / 2
    assert np.allclose(banded.equity, simulate_reference(close, w, costs, band=0.1), rtol=1e-12)
    ev, _ = run_event_driven(close, lambda h: w[len(h) - 1], costs, band=0.1)
    assert np.allclose(banded.equity, ev, rtol=1e-9)
    with pytest.raises(ValueError):
        run_vectorized(close, w, costs, band=-0.1)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 1000), band=st.floats(0, 0.5))
def test_band_numba_matches_reference(seed, band):
    rng = np.random.default_rng(seed)
    close = gbm(200, seed)
    w = rng.uniform(-1, 1, 200)
    costs = CostModel(10, 5)
    fast = run_vectorized(close, w, costs, band=band).equity
    assert np.allclose(fast, simulate_reference(close, w, costs, band=band), rtol=1e-12)


def test_walk_forward_tiles_out_of_sample_and_picks_on_train_only():
    close = gbm(1_000, seed=5)
    costs = CostModel(10, 5)
    params = {"lookback": [5, 20, 60]}
    wf = walk_forward(
        close, momentum, params, train_bars=300, test_bars=100, costs=costs, timeframe="1h"
    )
    assert np.all(wf.weights[:300] == 0)
    tests = [f.test for f in wf.folds]
    assert tests[0][0] == 300 and tests[-1][1] == 1_000
    assert all(a[1] == b[0] for a, b in zip(tests, tests[1:], strict=False))  # no gaps/overlap

    for f in wf.folds:
        train = close[f.train[0] : f.train[1]]
        scores = {
            p["lookback"]: sharpe(
                run_vectorized(train, momentum(train, **p), costs).returns[1:], 8760
            )
            for p in grid(params)
        }
        assert f.params["lookback"] == max(scores, key=scores.get)
        # weights in the test window are the chosen rule, computed with past data only
        expected = make_weights(momentum, close[: f.test[1]], f.params, None)
        assert np.array_equal(wf.weights[f.test[0] : f.test[1]], expected[f.test[0] :])


def test_walk_forward_needs_enough_data():
    with pytest.raises(ValueError, match="not enough"):
        walk_forward(gbm(100), momentum, {"lookback": [5]}, 80, 40, CostModel(), "1h")
