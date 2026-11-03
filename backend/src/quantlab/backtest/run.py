"""Backtest a signal on stored bars and print the metrics.

python -m quantlab.backtest.run --symbol BTC-USD --timeframe 1h --fast 24 --slow 168
"""

import argparse
import json

import numpy as np

from quantlab.backtest.costs import CostModel
from quantlab.backtest.metrics import compute_metrics
from quantlab.backtest.signals import sma_cross_weights
from quantlab.backtest.vectorized import run_vectorized
from quantlab.config import get_settings
from quantlab.data.store import load_bars, make_engine


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SMA crossover backtest on stored bars")
    p.add_argument("--db", default=get_settings().database_url)
    p.add_argument("--source", default="coinbase")
    p.add_argument("--symbol", default="BTC-USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--fast", type=int, default=24)
    p.add_argument("--slow", type=int, default=168)
    p.add_argument(
        "--short", action="store_true", help="go short instead of flat below the slow MA"
    )
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--cash", type=float, default=10_000.0)
    args = p.parse_args(argv)

    bars = load_bars(make_engine(args.db), args.source, args.symbol, args.timeframe)
    if bars.is_empty():
        raise SystemExit(
            f"no {args.timeframe} bars for {args.symbol}, run quantlab.data.ingest first"
        )
    close = bars["close"].to_numpy()
    costs = CostModel(args.fee_bps, args.slippage_bps)

    weights = sma_cross_weights(close, args.fast, args.slow, short=args.short)
    res = run_vectorized(close, weights, costs, args.cash)
    strat = compute_metrics(
        res.equity, res.returns, weights, res.turnover, res.trades, args.timeframe, args.cash
    )

    # buy and hold as the baseline every strategy has to beat
    hold_w = np.ones_like(close)
    hold = run_vectorized(close, hold_w, costs, args.cash)
    base = compute_metrics(
        hold.equity, hold.returns, hold_w, hold.turnover, hold.trades, args.timeframe, args.cash
    )

    print(f"{args.symbol} {args.timeframe}: {len(close)} bars, {bars['ts'][0]} -> {bars['ts'][-1]}")
    print(json.dumps({"strategy": strat.as_dict(), "buy_and_hold": base.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
