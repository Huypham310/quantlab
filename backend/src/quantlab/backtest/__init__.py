"""Backtesting: a Numba-compiled vectorized engine, cross-checked by an event-driven one."""

from quantlab.backtest.costs import CostModel
from quantlab.backtest.event import run_event_driven
from quantlab.backtest.metrics import Metrics, compute_metrics
from quantlab.backtest.vectorized import BacktestResult, run_vectorized

__all__ = [
    "BacktestResult",
    "CostModel",
    "Metrics",
    "compute_metrics",
    "run_event_driven",
    "run_vectorized",
]
