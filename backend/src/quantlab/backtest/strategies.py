"""Strategy library: each function maps closes to target weights using data up to t only.

Weights are what the engines hold over t -> t+1 (see vectorized.py), in [-1, 1] before any
volatility targeting. Every rule has a warm-up period where it stays flat.
"""

from collections.abc import Callable

import numpy as np

from quantlab.backtest.signals import sma, sma_cross_weights

Strategy = Callable[..., np.ndarray]


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing sample std; NaN during warm-up."""
    out = np.full(x.shape, np.nan)
    if len(x) < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(x, window)
    out[window - 1 :] = view.std(axis=1, ddof=1)
    return out


def log_returns(close: np.ndarray) -> np.ndarray:
    """Bar-to-bar log returns; NaN for the first bar, which has no previous close.

    (A 0 there looks harmless but drags the first volatility estimate down.)
    """
    r = np.full(close.shape, np.nan)
    r[1:] = np.diff(np.log(close))
    return r


def momentum(close: np.ndarray, lookback: int, short: bool = False) -> np.ndarray:
    """Time-series momentum: long if the price is above where it was `lookback` bars ago."""
    w = np.zeros_like(close, dtype=np.float64)
    past = close[:-lookback]
    up = close[lookback:] > past
    w[lookback:] = np.where(up, 1.0, -1.0 if short else 0.0)
    return w


def donchian(close: np.ndarray, window: int, short: bool = False) -> np.ndarray:
    """Breakout: go long on a new `window`-bar high, exit / reverse on a new low, else hold."""
    n = len(close)
    w = np.zeros(n)
    if n <= window:
        return w
    view = np.lib.stride_tricks.sliding_window_view(close, window)
    # channel from the PREVIOUS window bars, so today's close can break it
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi[window:] = view[:-1].max(axis=1)
    lo[window:] = view[:-1].min(axis=1)
    pos = 0.0
    for t in range(window, n):
        if close[t] > hi[t]:
            pos = 1.0
        elif close[t] < lo[t]:
            pos = -1.0 if short else 0.0
        w[t] = pos
    return w


def zscore_reversion(close: np.ndarray, window: int, entry: float = 2.0) -> np.ndarray:
    """Mean reversion: long when price is `entry` std below its rolling mean, short when above,
    flat again once it crosses back through the mean."""
    n = len(close)
    mean = sma(close, window)
    std = rolling_std(close, window)
    z = (close - mean) / std
    w = np.zeros(n)
    pos = 0.0
    for t in range(n):
        if np.isnan(z[t]):
            continue
        if pos == 0.0:
            if z[t] < -entry:
                pos = 1.0
            elif z[t] > entry:
                pos = -1.0
        elif (pos > 0 and z[t] >= 0) or (pos < 0 and z[t] <= 0):
            pos = 0.0
        w[t] = pos
    return w


def vol_target(
    weights: np.ndarray,
    close: np.ndarray,
    target: float,
    window: int,
    periods_per_year: float,
    max_leverage: float = 1.0,
) -> np.ndarray:
    """Scale positions so the asset's recent realized volatility maps to `target` (annualized).

    Uses volatility up to t only. Capped at `max_leverage` so a quiet period can't blow the
    position up; flat (0) until the volatility estimate exists.
    """
    vol = rolling_std(log_returns(close), window) * np.sqrt(periods_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(vol > 0, target / vol, 0.0)
    scale = np.nan_to_num(np.minimum(scale, max_leverage), nan=0.0)
    return weights * scale


# name -> (function, parameter grid for walk-forward selection)
LIBRARY: dict[str, tuple[Strategy, dict[str, list]]] = {
    "sma_cross": (sma_cross_weights, {"fast": [12, 24, 48], "slow": [96, 168, 336]}),
    "momentum": (momentum, {"lookback": [24, 72, 168, 336, 720]}),
    "donchian": (donchian, {"window": [48, 168, 336, 720]}),
    "zscore_reversion": (zscore_reversion, {"window": [24, 72, 168], "entry": [1.5, 2.0, 2.5]}),
}
