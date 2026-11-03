import numpy as np
import pytest

from quantlab.backtest import CostModel, compute_metrics, run_vectorized
from quantlab.backtest.metrics import bars_per_year, max_drawdown


def test_max_drawdown():
    assert max_drawdown(np.array([100.0, 120, 90, 130, 65])) == pytest.approx(0.5)
    assert max_drawdown(np.array([1.0, 2, 3])) == 0.0


def test_bars_per_year():
    assert bars_per_year("1h") == 8760
    assert bars_per_year("1d") == 365


def test_doubling_over_one_year_of_daily_bars():
    close = np.geomspace(100, 200, 366)  # 365 daily returns, steady growth
    res = run_vectorized(close, np.ones_like(close), CostModel(0, 0), 1_000)
    m = compute_metrics(
        res.equity, res.returns, np.ones_like(close), res.turnover, res.trades, "1d", 1_000
    )
    assert m.total_return == pytest.approx(1.0)
    assert m.cagr == pytest.approx(1.0, rel=0.01)
    assert m.max_drawdown == 0.0
    assert m.exposure == 1.0 and m.trades == 1


def test_zero_volatility_gives_zero_sharpe_not_inf():
    eq = np.full(10, 1_000.0)
    m = compute_metrics(eq, np.zeros(10), np.zeros(10), np.zeros(10), 0, "1h", 1_000)
    assert m.sharpe == 0.0 and m.sortino == 0.0 and m.volatility == 0.0


def test_sharpe_sign_and_scale():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 10_000)
    eq = 1_000 * np.cumprod(1 + r)
    m = compute_metrics(eq, r, np.ones_like(r), np.zeros_like(r), 0, "1d", 1_000)
    # mean/std * sqrt(365) ~ 0.1 * 19.1
    assert m.sharpe == pytest.approx(0.1 * np.sqrt(365), rel=0.1)
    assert m.sortino > m.sharpe  # positive drift -> less downside than total deviation


def test_wiped_out_account():
    eq = np.array([500.0, 0.0])
    m = compute_metrics(eq, np.array([-0.5, -1.0]), np.ones(2), np.zeros(2), 1, "1d", 1_000)
    assert m.cagr == -1.0 and m.max_drawdown == 1.0
