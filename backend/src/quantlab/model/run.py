"""Train the ML signal walk-forward and backtest it out of sample.

python -m quantlab.model.run --symbol ETH-USD --model lgbm
python -m quantlab.model.run --symbol BTC-USD --model lstm --trials 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantlab.backtest.costs import CostModel
from quantlab.backtest.metrics import bars_per_year
from quantlab.backtest.research import evaluate
from quantlab.config import get_settings
from quantlab.data.store import load_bars, make_engine
from quantlab.model.evaluate import prediction_metrics, proba_to_weights, run_walk_forward
from quantlab.model.features import forward_returns, make_features
from quantlab.model.splits import walk_forward_splits

WARMUP = 24 * 7 + 1  # longest feature window (168 bars) + 1


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
    close = bars["close"].to_numpy()
    feats = make_features(bars)
    fwd = forward_returns(close, args.horizon)
    per_day = 24 if args.timeframe == "1h" else 1
    splits = walk_forward_splits(
        len(close),
        args.train_days * per_day,
        args.test_days * per_day,
        args.horizon,
        start=WARMUP,
    )
    print(
        f"{args.symbol} {args.timeframe}: {len(close)} bars, {len(feats.names)} features, "
        f"{len(splits)} folds, model {args.model}, horizon {args.horizon}",
        flush=True,
    )
    run = run_walk_forward(
        feats.x, fwd, splits, args.model, args.trials, args.seed, feats.names, verbose=True
    )

    oos = splits[0].test_range[0]
    pred = prediction_metrics(run.proba[oos:], fwd[oos:])
    costs = CostModel(args.fee_bps, args.slippage_bps)
    w = proba_to_weights(run.proba, args.mode, args.threshold, args.smooth)
    model_bt = evaluate(close, w, costs, args.timeframe, oos)
    # same trades with zero costs: separates "no edge" from "edge eaten by fees"
    gross_bt = evaluate(close, w, CostModel(0.0, 0.0), args.timeframe, oos)
    hold_bt = evaluate(close, np.ones_like(close), costs, args.timeframe, oos)

    print(
        f"\nout of sample {bars['ts'][oos]} -> {bars['ts'][-1]}: auc {pred['auc']:.3f}, "
        f"accuracy {pred['accuracy']:.3f} (base rate {pred['base_rate']:.3f}), ic {pred['ic']:+.3f}"
    )
    print(f"{'':16s} {'sharpe':>7s} {'return':>8s} {'maxDD':>7s} {'trades':>7s} {'exposure':>9s}")
    rows = (
        (f"{args.model} {args.mode}", model_bt),
        ("  before costs", gross_bt),
        ("buy and hold", hold_bt),
    )
    for name, m in rows:
        print(
            f"{name:16s} {m['sharpe']:7.2f} {m['total_return']:8.1%} {m['max_drawdown']:7.1%} "
            f"{m['trades']:7d} {m.get('exposure', float('nan')):9.2f}"
        )
    if run.importance:
        top = ", ".join(f"{k} {v:.2f}" for k, v in list(run.importance.items())[:6])
        print(f"top features (gain share): {top}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "symbol": args.symbol,
        "model": args.model,
        "args": vars(args) | {"db": None},
        "bars_per_year": bars_per_year(args.timeframe),
        "prediction": pred,
        "backtest": {"model": model_bt, "model_gross": gross_bt, "buy_and_hold": hold_bt},
        "folds": run.folds,
        "importance": run.importance,
    }
    path = out / f"model_{args.symbol}_{args.model}_h{args.horizon}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
