"""Vectorized backtest, compiled with Numba.

Convention (the one that rules out look-ahead): ``weights[t]`` is decided at the close of bar t,
using data up to t only, and is held over the next bar, so it earns
``close[t + 1] / close[t] - 1``. Rebalancing at bar t costs |target - weight actually held|
times the cost rate, paid out of equity at bar t. The held weight drifts with the price between
bars (a fixed number of units, not a fixed fraction), same as a real account.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from quantlab.backtest.costs import CostModel


@dataclass
class BacktestResult:
    equity: np.ndarray  # equity after each bar, starts at initial_cash before bar 0
    returns: np.ndarray  # per-bar net return
    turnover: np.ndarray  # |change in weight| per bar
    costs: np.ndarray  # cost paid per bar, in cash
    trades: int  # bars where the position changed


@njit(cache=True)
def _simulate(close, weights, cost_rate, initial_cash):  # pragma: no cover - compiled
    n = close.shape[0]
    equity = np.empty(n)
    turnover = np.zeros(n)
    costs = np.zeros(n)
    trades = 0
    eq = initial_cash
    cur_w = 0.0  # weight actually held right now (drifts with the price)
    for t in range(n):
        # 1) mark to market: units held since bar t-1 earn this bar's move
        if t > 0:
            r = close[t] / close[t - 1] - 1.0
            grow = 1.0 + cur_w * r
            eq = eq * grow
            # a 50% position that rose 10% is no longer 50% of equity
            cur_w = cur_w * (1.0 + r) / grow if grow != 0.0 else 0.0
        # 2) rebalance to the target at the close of t
        dw = abs(weights[t] - cur_w)
        if dw > 1e-12:
            c = dw * cost_rate * eq
            costs[t] = c
            turnover[t] = dw
            trades += 1
            eq -= c
        cur_w = weights[t]
        equity[t] = eq
    return equity, turnover, costs, trades


def run_vectorized(
    close: np.ndarray,
    weights: np.ndarray,
    costs: CostModel,
    initial_cash: float = 10_000.0,
) -> BacktestResult:
    """Simulate holding ``weights`` (fraction of equity, negative = short) on ``close`` prices."""
    close = np.ascontiguousarray(close, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if close.shape != weights.shape or close.ndim != 1:
        raise ValueError("close and weights must be 1-D arrays of the same length")
    if close.size == 0:
        raise ValueError("no bars")
    if not np.all(np.isfinite(close)) or np.any(close <= 0):
        raise ValueError("close prices must be positive and finite")
    if not np.all(np.isfinite(weights)):
        # NaN weights usually mean an indicator still warming up; make the caller decide
        raise ValueError("weights contain NaN/inf (fill the warm-up period first)")
    equity, turnover, cost, trades = _simulate(close, weights, costs.rate, float(initial_cash))
    returns = np.diff(equity, prepend=float(initial_cash)) / np.concatenate(
        ([float(initial_cash)], equity[:-1])
    )
    return BacktestResult(equity, returns, turnover, cost, int(trades))


def simulate_reference(close, weights, costs: CostModel, initial_cash: float = 10_000.0):
    """Plain-Python twin of ``_simulate``; slow, only used in tests and the benchmark."""
    eq = initial_cash
    cur_w = 0.0
    equity = []
    for t in range(len(close)):
        if t > 0:
            r = close[t] / close[t - 1] - 1.0
            grow = 1.0 + cur_w * r
            eq *= grow
            cur_w = cur_w * (1.0 + r) / grow if grow != 0.0 else 0.0
        eq -= abs(weights[t] - cur_w) * costs.rate * eq
        cur_w = weights[t]
        equity.append(eq)
    return np.array(equity)
