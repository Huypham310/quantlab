"""The work behind the API's background jobs, as plain functions (Celery just calls them)."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import polars as pl

from quantlab.backtest.costs import CostModel
from quantlab.backtest.metrics import bars_per_year, compute_metrics
from quantlab.backtest.strategies import LIBRARY, vol_target
from quantlab.backtest.vectorized import run_vectorized
from quantlab.model.run import train_signal

MAX_CURVE_POINTS = 500


def strategy_params(name: str) -> list[str]:
    """Keyword parameters a library strategy accepts (everything after `close`)."""
    if name == "buy_and_hold":
        return []
    fn, _ = LIBRARY[name]
    return [p for p in inspect.signature(fn).parameters if p != "close"]


def validate_strategy(name: str, params: dict) -> None:
    """Raise ValueError for an unknown strategy, unknown params or missing required ones."""
    if name != "buy_and_hold" and name not in LIBRARY:
        raise ValueError(f"unknown strategy {name!r}")
    allowed = strategy_params(name)
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(f"unknown parameter(s) for {name}: {', '.join(sorted(unknown))}")
    if name != "buy_and_hold":
        sig = inspect.signature(LIBRARY[name][0])
        missing = [
            p
            for p in allowed
            if sig.parameters[p].default is inspect.Parameter.empty and p not in params
        ]
        if missing:
            raise ValueError(f"missing parameter(s) for {name}: {', '.join(missing)}")


def downsample(ts: list, values: np.ndarray, n: int = MAX_CURVE_POINTS) -> list[dict]:
    """At most `n` evenly spaced points of a curve, always keeping the last one."""
    idx = np.unique(np.linspace(0, len(values) - 1, min(n, len(values))).astype(int))
    return [{"ts": str(ts[i]), "value": round(float(values[i]), 4)} for i in idx]


def backtest_job(bars: pl.DataFrame, req: dict) -> dict:
    """Run one strategy (or buy and hold) on `bars` with the request's costs / sizing."""
    validate_strategy(req["strategy"], req.get("params", {}))
    if bars.height < 2:
        raise ValueError("not enough bars for a backtest")
    close = bars["close"].to_numpy()
    timeframe = req.get("timeframe", "1h")
    if req["strategy"] == "buy_and_hold":
        w = np.ones_like(close)
    else:
        w = LIBRARY[req["strategy"]][0](close, **req.get("params", {}))
    vt = req.get("vol_target")
    if vt:
        per_day = 24 if timeframe == "1h" else 1
        w = vol_target(w, close, vt, 7 * per_day, bars_per_year(timeframe))
    costs = CostModel(req.get("fee_bps", 10.0), req.get("slippage_bps", 5.0))
    cash = float(req.get("cash", 10_000.0))
    res = run_vectorized(close, w, costs, cash, band=req.get("band", 0.0))
    m = compute_metrics(res.equity, res.returns, w, res.turnover, res.trades, timeframe, cash)
    ts = bars["ts"].to_list()
    return {
        "metrics": m.as_dict(),
        "bars": bars.height,
        "start": str(ts[0]),
        "end": str(ts[-1]),
        "equity": downsample(ts, res.equity),
    }


def train_job(bars: pl.DataFrame, req: dict, progress: Callable[[dict], None] | None) -> dict:
    """Walk-forward ML training; `progress` gets each finished fold."""
    report = train_signal(
        bars,
        model=req["model"],
        timeframe=req.get("timeframe", "1h"),
        horizon=req.get("horizon", 24),
        train_days=req.get("train_days", 365),
        test_days=req.get("test_days", 90),
        trials=req.get("trials", 10),
        seed=req.get("seed", 0),
        on_fold=progress,
    )
    # json can't hold NaN (e.g. AUC of a fold with one class); the frontend wants null
    return _nan_to_none(report)


def _nan_to_none(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    return x
