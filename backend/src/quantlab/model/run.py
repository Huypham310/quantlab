"""Train the ML signal walk-forward and backtest it out of sample.

python -m quantlab.model.run --symbol ETH-USD --model lgbm
python -m quantlab.model.run --symbol BTC-USD --model lstm --trials 6
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from quantlab.backtest.costs import CostModel
from quantlab.backtest.metrics import bars_per_year
from quantlab.backtest.research import evaluate
from quantlab.config import get_settings
from quantlab.data.store import load_bars, make_engine
from quantlab.model.evaluate import prediction_metrics, proba_to_weights, run_walk_forward
from quantlab.model.features import forward_returns, make_features
from quantlab.model.splits import walk_forward_splits

WARMUP = 24 * 7 + 1  # longest feature window (168 bars) + 1


def train_signal(
    bars: pl.DataFrame,
    model: str = "lgbm",
    timeframe: str = "1h",
    horizon: int = 24,
    train_days: int = 365,
    test_days: int = 90,
    trials: int = 20,
    mode: str = "long_only",
    threshold: float = 0.0,
    smooth: int = 12,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    seed: int = 0,
    verbose: bool = False,
    on_fold: Callable[[dict], None] | None = None,
) -> dict:
    """Walk-forward train + out-of-sample evaluation; returns the report (json-safe dict).

    Shared by the CLI and the API's background job, so both give the same numbers.
    """
    close = bars["close"].to_numpy()
    feats = make_features(bars)
    fwd = forward_returns(close, horizon)
    per_day = 24 if timeframe == "1h" else 1
    splits = walk_forward_splits(
        len(close), train_days * per_day, test_days * per_day, horizon, start=WARMUP
    )
    if verbose:
        print(
            f"{len(close)} bars, {len(feats.names)} features, {len(splits)} folds, "
            f"model {model}, horizon {horizon}",
            flush=True,
        )
    run = run_walk_forward(
        feats.x, fwd, splits, model, trials, seed, feats.names, verbose=verbose, on_fold=on_fold
    )

    oos = splits[0].test_range[0]
    costs = CostModel(fee_bps, slippage_bps)
    w = proba_to_weights(run.proba, mode, threshold, smooth)
    return {
        "model": model,
        "horizon": horizon,
        "oos_start": str(bars["ts"][oos]),
        "oos_end": str(bars["ts"][-1]),
        "bars_per_year": bars_per_year(timeframe),
        "prediction": prediction_metrics(run.proba[oos:], fwd[oos:]),
        "backtest": {
            "model": evaluate(close, w, costs, timeframe, oos),
            # same trades with zero costs: separates "no edge" from "edge eaten by fees"
            "model_gross": evaluate(close, w, CostModel(0.0, 0.0), timeframe, oos),
            "buy_and_hold": evaluate(close, np.ones_like(close), costs, timeframe, oos),
        },
        "folds": run.folds,
        "importance": run.importance,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=get_settings().database_url)
    p.add_argument("--source", default="coinbase")
    p.add_argument("--symbol", default="ETH-USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--model", choices=["lgbm", "lstm"], default="lgbm")
    p.add_argument("--horizon", type=int, default=24, help="label = sign of the next N-bar return")
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--trials", type=int, default=20, help="Optuna trials per fold")
    p.add_argument("--mode", choices=["long_only", "long_short"], default="long_only")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--smooth", type=int, default=12, help="average the last N predictions")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports", help="directory for the json report")
    args = p.parse_args(argv)

    engine = make_engine(args.db)
    bars = load_bars(engine, args.source, args.symbol, args.timeframe)
    engine.dispose()  # training takes minutes, no reason to hold the connection
    print(f"{args.symbol} {args.timeframe}:", end=" ")
    report = train_signal(
        bars,
        args.model,
        args.timeframe,
        args.horizon,
        args.train_days,
        args.test_days,
        args.trials,
        args.mode,
        args.threshold,
        args.smooth,
        args.fee_bps,
        args.slippage_bps,
        args.seed,
        verbose=True,
    )

    pred, bt = report["prediction"], report["backtest"]
    print(
        f"\nout of sample {report['oos_start']} -> {report['oos_end']}: auc {pred['auc']:.3f}, "
        f"accuracy {pred['accuracy']:.3f} (base rate {pred['base_rate']:.3f}), ic {pred['ic']:+.3f}"
    )
    print(f"{'':16s} {'sharpe':>7s} {'return':>8s} {'maxDD':>7s} {'trades':>7s} {'exposure':>9s}")
    rows = (
        (f"{args.model} {args.mode}", bt["model"]),
        ("  before costs", bt["model_gross"]),
        ("buy and hold", bt["buy_and_hold"]),
    )
    for name, m in rows:
        print(
            f"{name:16s} {m['sharpe']:7.2f} {m['total_return']:8.1%} {m['max_drawdown']:7.1%} "
            f"{m['trades']:7d} {m.get('exposure', float('nan')):9.2f}"
        )
    if report["importance"]:
        top = ", ".join(f"{k} {v:.2f}" for k, v in list(report["importance"].items())[:6])
        print(f"top features (gain share): {top}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"symbol": args.symbol, "args": vars(args) | {"db": None}} | report
    path = out / f"model_{args.symbol}_{args.model}_h{args.horizon}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
