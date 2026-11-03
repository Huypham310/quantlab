"""Throughput of the three ways to run the same backtest.

python -m quantlab.backtest.bench
"""

import argparse
import time

import numpy as np

from quantlab.backtest.costs import CostModel
from quantlab.backtest.event import run_event_driven
from quantlab.backtest.signals import sma_cross_strategy, sma_cross_weights
from quantlab.backtest.vectorized import run_vectorized, simulate_reference


def _best_of(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bars", type=int, default=1_000_000)
    p.add_argument("--event-bars", type=int, default=50_000, help="the event engine is slow")
    args = p.parse_args(argv)

    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, args.bars)))
    weights = sma_cross_weights(close, 24, 168, short=True)
    costs = CostModel(10, 5)

    run_vectorized(close[:1000], weights[:1000], costs)  # first call compiles, keep it out
    t_numba = _best_of(lambda: run_vectorized(close, weights, costs), 5)
    t_python = _best_of(lambda: simulate_reference(close, weights, costs), 1)
    ev = close[: args.event_bars]
    strat = sma_cross_strategy(24, 168, short=True)
    t_event = _best_of(lambda: run_event_driven(ev, strat, costs), 1)

    rows = [
        ("numba vectorized", args.bars, t_numba),
        ("pure python loop", args.bars, t_python),
        ("event-driven", args.event_bars, t_event),
    ]
    print(f"{'engine':18s} {'bars':>10s} {'seconds':>9s} {'bars/s':>14s}")
    for name, n, t in rows:
        print(f"{name:18s} {n:10d} {t:9.4f} {n / t:14,.0f}")
    print(f"numba vs python loop: {t_python / t_numba:.0f}x")


if __name__ == "__main__":
    main()
