"""Features and labels for the ML signals.

Every feature at row t uses bars up to and including t only (same rule as the strategies:
the weight decided at the close of t is held over t -> t+1). The label at t is the log return
from close t to close t + horizon, i.e. purely in the future of the features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from quantlab.backtest.strategies import log_returns, rolling_std

RETURN_LAGS = (1, 3, 6, 12, 24, 72, 168)


@dataclass
class FeatureSet:
    x: np.ndarray  # (bars, features), NaN during warm-up
    names: list[str]


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if len(x) < window:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1 :] = (c[window:] - c[:-window]) / window
    return out


def _rolling_extreme(x: np.ndarray, window: int, fn) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if len(x) < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(x, window)
    out[window - 1 :] = fn(view, axis=1)
    return out


def make_features(bars: pl.DataFrame) -> FeatureSet:
    """Price, volatility, range, volume and calendar features from OHLCV bars."""
    close = bars["close"].to_numpy().astype(np.float64)
    high = bars["high"].to_numpy().astype(np.float64)
    low = bars["low"].to_numpy().astype(np.float64)
    volume = bars["volume"].to_numpy().astype(np.float64)
    logc = np.log(close)
    r1 = log_returns(close)

    cols: dict[str, np.ndarray] = {}
    for k in RETURN_LAGS:
        lagged = np.full_like(logc, np.nan)
        lagged[k:] = logc[k:] - logc[:-k]
        cols[f"ret_{k}"] = lagged

    vol24, vol168 = rolling_std(r1, 24), rolling_std(r1, 168)
    cols["vol_24"] = vol24
    cols["vol_168"] = vol168
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["vol_ratio"] = vol24 / vol168
        for w in (24, 168):
            # distance from the moving average in units of the price's own rolling std
            cols[f"z_{w}"] = (close - _rolling_mean(close, w)) / rolling_std(close, w)
        hi, lo = _rolling_extreme(high, 24, np.max), _rolling_extreme(low, 24, np.min)
        cols["range_pos_24"] = (close - lo) / (hi - lo)
    cols["bar_range_24"] = _rolling_mean(np.log(high / low), 24)

    logv = np.log1p(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["volume_z_168"] = (logv - _rolling_mean(logv, 168)) / rolling_std(logv, 168)

    # chu kỳ trong ngày / tuần: sin-cos để 23h và 0h nằm cạnh nhau
    ts = bars["ts"]
    hour = ts.dt.hour().to_numpy().astype(np.float64)
    dow = ts.dt.weekday().to_numpy().astype(np.float64)
    cols["hour_sin"], cols["hour_cos"] = (
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    )
    cols["dow_sin"], cols["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    x = np.column_stack(list(cols.values()))
    x[~np.isfinite(x)] = np.nan
    return FeatureSet(x=x, names=list(cols))


def forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """log(close[t + horizon] / close[t]); NaN for the last `horizon` bars."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = np.full(close.shape, np.nan)
    logc = np.log(close)
    out[:-horizon] = logc[horizon:] - logc[:-horizon]
    return out
