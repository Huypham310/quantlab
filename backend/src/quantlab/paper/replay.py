"""Stress the paper broker with a long replayed tick stream and a random order flow.

    python -m quantlab.paper.replay --ticks 500000

Prices follow a geometric random walk per symbol (one print every 100 ms of simulated time,
~60% annualized vol, plus rare 3-8% jumps so the fill clamp and kill switch get exercised).
At every tick a random "trader" may send a market or limit order, or cancel an open one.
Reports how long submit() and on_trade() take, what got rejected and why, and the number of
limit breaches (which must be 0).
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter

import numpy as np

from quantlab.paper.broker import PaperBroker
from quantlab.paper.risk import RiskLimits

START = {"BTC-USD": 60_000.0, "ETH-USD": 3_000.0}


def run(ticks: int, seed: int = 0, order_prob: float = 0.01, cancel_prob: float = 0.02) -> dict:
    rng = random.Random(seed)
    broker = PaperBroker(cash=100_000.0, limits=RiskLimits(max_orders_per_minute=10_000))
    prices = dict(START)
    t_ms = 1_700_000_000_000
    for sym, px in prices.items():
        broker.on_trade(sym, px, t_ms)

    step_vol = 0.6 / math.sqrt(365 * 24 * 3600 * 10)  # per 100 ms
    submit_ns: list[int] = []
    tick_ns: list[int] = []
    outcomes: Counter[str] = Counter()
    # loss limit resets every simulated hour (a random trader burns through 5% of equity in
    # fees within hours; with daily resets almost every later order is a kill-switch reject)
    session_ms = 3600 * 1000
    kills = 0
    for i in range(ticks):
        t_ms += 100
        if i and t_ms % session_ms < 100:
            kills += broker.killed
            broker.new_day()
        sym = "BTC-USD" if rng.random() < 0.5 else "ETH-USD"
        shock = rng.gauss(0, step_vol)
        if rng.random() < 1e-4:
            shock += rng.choice((-1, 1)) * rng.uniform(0.03, 0.08)  # rare jump
        prices[sym] *= math.exp(shock)

        t0 = time.perf_counter_ns()
        broker.on_trade(sym, prices[sym], t_ms)
        tick_ns.append(time.perf_counter_ns() - t0)

        if rng.random() < order_prob:
            side = rng.choice(("buy", "sell"))
            qty = rng.uniform(0.05, 1.0) * 5_000 / prices[sym] / 1.02
            limit = None
            if rng.random() < 0.5:
                limit = prices[sym] * rng.uniform(0.98, 1.02)
            t0 = time.perf_counter_ns()
            order = broker.submit(sym, side, qty, "limit" if limit else "market", limit, t_ms)
            submit_ns.append(time.perf_counter_ns() - t0)
            outcomes[order.reason or "accepted"] += 1
        if broker.open and rng.random() < cancel_prob:
            broker.cancel(rng.choice(list(broker.open)))

    s, k = np.array(submit_ns) / 1000, np.array(tick_ns) / 1000
    return {
        "ticks": ticks,
        "orders": len(submit_ns),
        "fills": len(broker.fills),
        "outcomes": dict(outcomes.most_common()),
        "submit_us_p50": float(np.percentile(s, 50)),
        "submit_us_p99": float(np.percentile(s, 99)),
        "tick_us_p50": float(np.percentile(k, 50)),
        "tick_us_p99": float(np.percentile(k, 99)),
        "breaches": broker.breaches,
        "sessions_killed": kills + broker.killed,
        "final_equity": broker.equity(),
        "fees_paid": broker.fees_paid,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticks", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--order-prob", type=float, default=0.01, help="chance of an order per tick")
    args = p.parse_args(argv)
    start = time.perf_counter()
    r = run(args.ticks, args.seed, args.order_prob)
    wall = time.perf_counter() - start
    print(
        f"{r['ticks']:,} ticks, {r['orders']:,} orders, {r['fills']:,} fills in {wall:.1f} s; "
        f"breaches: {r['breaches']}, sessions stopped by the kill switch: {r['sessions_killed']}"
    )
    print(
        f"submit: p50 {r['submit_us_p50']:.1f} us, p99 {r['submit_us_p99']:.1f} us | "
        f"on_trade: p50 {r['tick_us_p50']:.1f} us, p99 {r['tick_us_p99']:.1f} us"
    )
    for reason, n in r["outcomes"].items():
        print(f"  {n:8,}  {reason}")


if __name__ == "__main__":
    main()
