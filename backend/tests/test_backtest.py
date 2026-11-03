import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.backtest import CostModel, run_event_driven, run_vectorized
from quantlab.backtest.costs import FREE
from quantlab.backtest.signals import sma, sma_cross_strategy, sma_cross_weights
from quantlab.backtest.vectorized import simulate_reference


def gbm(n: int, seed: int = 0, vol: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))


def test_buy_and_hold_tracks_the_price_minus_one_entry_cost():
    close = np.array([100.0, 110.0, 99.0, 120.0])
    costs = CostModel(fee_bps=10, slippage_bps=0)
    res = run_vectorized(close, np.ones(4), costs, initial_cash=1_000)
    expected = 1_000 * (1 - 0.001) * close / close[0]
    assert np.allclose(res.equity, expected)
    assert res.trades == 1


def test_no_position_means_flat_equity():
    res = run_vectorized(gbm(100), np.zeros(100), CostModel(), initial_cash=500)
    assert np.all(res.equity == 500) and res.trades == 0


def test_weight_at_t_only_earns_the_next_bar():
    # going long at the close of bar 1 must not earn the move from bar 0 to bar 1
    close = np.array([100.0, 200.0, 200.0])
    res = run_vectorized(close, np.array([0.0, 1.0, 1.0]), FREE, initial_cash=1)
    assert res.equity.tolist() == [1.0, 1.0, 1.0]


def test_flipping_long_to_short_pays_twice():
    close = np.full(3, 100.0)
    costs = CostModel(fee_bps=10, slippage_bps=0)
    res = run_vectorized(close, np.array([1.0, -1.0, -1.0]), costs, initial_cash=1_000)
    assert res.turnover.tolist() == [1.0, 2.0, 0.0]
    assert res.costs[1] == pytest.approx(2 * 0.001 * res.equity[0])


def test_held_weight_drifts_with_the_price():
    # 50% in, price doubles: 2/3 of equity is now in the asset, getting back to 50% costs 1/6
    close = np.array([100.0, 200.0])
    costs = CostModel(fee_bps=100, slippage_bps=0)
    res = run_vectorized(close, np.array([0.5, 0.5]), costs, initial_cash=1_000)
    assert res.turnover[1] == pytest.approx(2 / 3 - 0.5)


@pytest.mark.parametrize(
    "close,weights,match",
    [
        (np.array([1.0, 2.0]), np.array([1.0]), "same length"),
        (np.array([1.0, -2.0]), np.array([1.0, 1.0]), "positive"),
        (np.array([1.0, 2.0]), np.array([np.nan, 1.0]), "NaN"),
        (np.array([]), np.array([]), "no bars"),
    ],
)
def test_input_validation(close, weights, match):
    with pytest.raises(ValueError, match=match):
        run_vectorized(close, weights, FREE)


def test_negative_costs_are_rejected():
    with pytest.raises(ValueError):
        CostModel(fee_bps=-1)


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(2, 300),
    fee=st.floats(0, 50),
    levels=st.lists(st.sampled_from([-1.0, -0.5, 0.0, 0.5, 1.0]), min_size=1, max_size=5),
)
def test_numba_engine_matches_the_python_reference(seed, n, fee, levels):
    rng = np.random.default_rng(seed)
    close = gbm(n, seed)
    weights = rng.choice(levels, size=n)
    costs = CostModel(fee, 0)
    fast = run_vectorized(close, weights, costs).equity
    slow = simulate_reference(close, weights, costs)
    assert np.allclose(fast, slow, rtol=1e-12, atol=0)


@pytest.mark.parametrize("short", [False, True])
def test_vectorized_and_event_driven_agree(short):
    close = gbm(3_000, seed=4)
    costs = CostModel(10, 5)
    vec = run_vectorized(close, sma_cross_weights(close, 10, 40, short=short), costs)
    ev, pf = run_event_driven(close, sma_cross_strategy(10, 40, short=short), costs)
    assert np.allclose(vec.equity, ev, rtol=1e-9)
    assert len(pf.fills) == vec.trades


def test_event_strategy_cannot_see_or_touch_the_future():
    close = gbm(50)
    seen = []

    def spy(history):
        seen.append(len(history))
        with pytest.raises(ValueError):
            history[-1] = 0.0  # read-only view
        return 0.0

    run_event_driven(close, spy, FREE)
    assert seen == list(range(1, 51))


def test_future_prices_dont_change_past_equity():
    close = gbm(500, seed=1)
    w = sma_cross_weights(close, 5, 20)
    base = run_vectorized(close, w, CostModel()).equity
    shocked = close.copy()
    shocked[300:] *= 3  # something wild happens later
    w2 = sma_cross_weights(shocked, 5, 20)
    after = run_vectorized(shocked, w2, CostModel()).equity
    assert np.array_equal(base[:300], after[:300])


def test_sma_and_signal_warmup():
    x = np.arange(1.0, 6.0)
    assert np.allclose(sma(x, 2)[1:], [1.5, 2.5, 3.5, 4.5])
    assert np.isnan(sma(x, 2)[0])
    w = sma_cross_weights(gbm(30), 3, 10)
    assert np.all(w[:9] == 0)
    with pytest.raises(ValueError):
        sma_cross_weights(x, 5, 3)
