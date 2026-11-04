"""Compare every strategy in the library, honestly.

For each one: the best parameters picked on the whole history (in-sample, flattering), the
walk-forward result (out-of-sample), both with and without volatility targeting, against buy and
hold over the same out-of-sample bars.

    python -m quantlab.backtest.research --symbol BTC-USD
"""

import argparse
import json

import numpy as np

from quantlab.backtest.costs import CostModel
from quantlab.backtest.metrics import bars_per_year, compute_metrics
from quantlab.backtest.strategies import LIBRARY
from quantlab.backtest.vectorized import run_vectorized
from quantlab.backtest.walkforward import grid, make_weights, sharpe, walk_forward
from quantlab.config import get_settings
from quantlab.data.store import load_bars, make_engine


def evaluate(close, weights, costs, timeframe, start, band=0.0):
    """Metrics over bars [start:], trading `weights` (flat before start)."""
    c, w = close[start:], weights[start:]
    res = run_vectorized(c, w, costs, band=band)
    m = compute_metrics(res.equity, res.returns, w, res.turnover, res.trades, timeframe, 10_000)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.as_dict().items()}


def research(close, timeframe, costs, train_bars, test_bars, vt, band=0.0):
    periods = bars_per_year(timeframe)
    rows = {}
    oos_start = train_bars
    for name, (fn, params) in LIBRARY.items():
        for label, v in (("", None), ("+vt", vt)):
            # in-sample: best parameters on the full history, scored on the same bars (optimistic)
            best_s, best_w = -np.inf, None
            for p in grid(params):
                w = make_weights(fn, close, p, v)
                s = sharpe(run_vectorized(close, w, costs, band=band).returns[1:], periods)
                if s > best_s:
                    best_s, best_w = s, w
            wf = walk_forward(close, fn, params, train_bars, test_bars, costs, timeframe, v, band)
            rows[name + label] = {
                "in_sample": evaluate(close, best_w, costs, timeframe, oos_start, band),
                "walk_forward": evaluate(close, wf.weights, costs, timeframe, oos_start, band),
                "params_chosen": [f.params for f in wf.folds],
            }
    hold = np.ones_like(close)
    rows["buy_and_hold"] = {
        "walk_forward": evaluate(close, hold, costs, timeframe, oos_start, band)
    }
    if vt:
        hold_vt = make_weights(lambda c: np.ones_like(c), close, {}, vt)
        rows["buy_and_hold+vt"] = {
            "walk_forward": evaluate(close, hold_vt, costs, timeframe, oos_start, band)
        }
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=get_settings().database_url)
    p.add_argument("--source", default="coinbase")
    p.add_argument("--symbol", default="BTC-USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--vol-target", type=float, default=0.30)
    p.add_argument("--band", type=float, default=0.10, help="no-trade band on the weight")
    p.add_argument("--json", action="store_true", help="dump everything as json")
    args = p.parse_args(argv)

    bars = load_bars(make_engine(args.db), args.source, args.symbol, args.timeframe)
    close = bars["close"].to_numpy()
    per_day = 86_400 / (3600 if args.timeframe == "1h" else 86_400)
    train, test = int(args.train_days * per_day), int(args.test_days * per_day)
    vt = {
        "target": args.vol_target,
        "window": int(7 * per_day),
        "periods_per_year": bars_per_year(args.timeframe),
        "max_leverage": 1.0,
    }
    rows = research(
        close,
        args.timeframe,
        CostModel(args.fee_bps, args.slippage_bps),
        train,
        test,
        vt,
        args.band,
    )

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return
    oos = bars["ts"][train]
    print(
        f"{args.symbol} {args.timeframe}: out-of-sample {oos} -> {bars['ts'][-1]} "
        f"({len(close) - train} bars), {args.fee_bps + args.slippage_bps:g} bps costs"
    )
    print(
        f"{'strategy':20s} {'IS sharpe':>10s} {'WF sharpe':>10s} {'WF return':>10s} "
        f"{'WF maxDD':>9s} {'trades':>7s}"
    )
    for name, r in rows.items():
        wf = r["walk_forward"]
        is_s = f"{r['in_sample']['sharpe']:10.2f}" if "in_sample" in r else f"{'':10s}"
        print(
            f"{name:20s} {is_s} {wf['sharpe']:10.2f} {wf['total_return']:10.1%} "
            f"{wf['max_drawdown']:9.1%} {wf['trades']:7d}"
        )


if __name__ == "__main__":
    main()
