"""A couple of simple signals, written twice: vectorized for speed and bar-by-bar for the
event engine, so the two engines can be checked against each other on the same idea.
(The real strategy library comes in the strategies phase.)"""

import numpy as np


def sma(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing simple moving average; NaN until ``window`` values are available."""
    out = np.full(x.shape, np.nan)
    if window <= 0:
        raise ValueError("window must be positive")
    if len(x) >= window:
        c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
        out[window - 1 :] = (c[window:] - c[:-window]) / window
    return out


def sma_cross_weights(close: np.ndarray, fast: int, slow: int, short: bool = False) -> np.ndarray:
    """1 when the fast SMA is above the slow one, else 0 (or -1 with ``short``).

    Uses closes up to and including t for the weight at t, which the engines hold over t -> t+1.
    """
    if fast >= slow:
        raise ValueError("fast window must be shorter than the slow one")
    f, s = sma(close, fast), sma(close, slow)
    w = np.where(f > s, 1.0, -1.0 if short else 0.0)
    # warm-up: no position until both averages exist
    w[np.isnan(s)] = 0.0
    return w


def sma_cross_strategy(fast: int, slow: int, short: bool = False):
    """Same rule for the event engine: only ever sees the history up to now."""

    def strategy(history: np.ndarray) -> float:
        if len(history) < slow:
            return 0.0
        f = history[-fast:].mean()
        s = history[-slow:].mean()
        return 1.0 if f > s else (-1.0 if short else 0.0)

    return strategy
