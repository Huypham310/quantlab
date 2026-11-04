"""Walk-forward evaluation: pick parameters on a past window, trade them on the next one.

Picking the best of 15 parameter sets on the whole history and reporting its Sharpe is the
classic way to fool yourself. Here every out-of-sample bar is traded with parameters chosen
only from data that ended before it, and the out-of-sample pieces are stitched into one curve.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from quantlab.backtest.costs import CostModel
from quantlab.backtest.metrics import bars_per_year
from quantlab.backtest.strategies import Strategy, vol_target
from quantlab.backtest.vectorized import run_vectorized


@dataclass
class Fold:
    train: tuple[int, int]  # [start, end) bar indexes
    test: tuple[int, int]
    params: dict
    train_sharpe: float


@dataclass
class WalkForwardResult:
    weights: np.ndarray  # out-of-sample weights, 0 before the first test window
    first_test_bar: int
    folds: list[Fold] = field(default_factory=list)


def grid(params: dict[str, list]) -> list[dict]:
    keys = list(params)
    return [dict(zip(keys, combo, strict=True)) for combo in product(*params.values())]


def sharpe(returns: np.ndarray, periods: float) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(periods)) if std > 0 else 0.0


def make_weights(strategy: Strategy, close: np.ndarray, params: dict, vt: dict | None):
    w = strategy(close, **params)
    if vt:
        w = vol_target(w, close, **vt)
    return w


def walk_forward(
    close: np.ndarray,
    strategy: Strategy,
    param_grid: dict[str, list],
    train_bars: int,
    test_bars: int,
    costs: CostModel,
    timeframe: str,
    vt: dict | None = None,
    band: float = 0.0,
) -> WalkForwardResult:
    """Rolling (not expanding) train window of `train_bars`, then `test_bars` out of sample."""
    n = len(close)
    if train_bars + test_bars > n:
        raise ValueError("not enough bars for one train + test window")
    periods = bars_per_year(timeframe)
    out = np.zeros(n)
    folds = []
    candidates = grid(param_grid)

    for start in range(0, n - train_bars, test_bars):
        tr0, tr1 = start, start + train_bars
        te1 = min(tr1 + test_bars, n)
        if tr1 >= n:
            break
        train_close = close[tr0:tr1]
        best, best_sharpe = None, -np.inf
        for params in candidates:
            w = make_weights(strategy, train_close, params, vt)
            res = run_vectorized(train_close, w, costs, band=band)
            s = sharpe(res.returns[1:], periods)
            if s > best_sharpe:
                best, best_sharpe = params, s
        # compute the chosen rule on everything up to the end of the test window: the rules only
        # look backwards, so this gives the test bars a full warm-up without seeing their future
        w_full = make_weights(strategy, close[:te1], best, vt)
        out[tr1:te1] = w_full[tr1:te1]
        folds.append(Fold((tr0, tr1), (tr1, te1), best, best_sharpe))

    return WalkForwardResult(out, train_bars, folds)
