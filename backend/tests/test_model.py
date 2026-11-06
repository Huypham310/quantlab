import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quantlab.data.bars import BAR_SCHEMA
from quantlab.model.evaluate import prediction_metrics, proba_to_weights, run_walk_forward
from quantlab.model.features import forward_returns, make_features
from quantlab.model.lstm import windows
from quantlab.model.splits import walk_forward_splits


def random_bars(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.abs(rng.normal(0, 0.005, n))
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [
        (t0 + timedelta(hours=i), c, c * (1 + s), c * (1 - s), c, float(rng.integers(1, 100)))
        for i, (c, s) in enumerate(zip(close, spread, strict=True))
    ]
    return pl.DataFrame(rows, schema=list(BAR_SCHEMA), orient="row").cast(BAR_SCHEMA)


def test_features_never_look_ahead():
    bars = random_bars(600)
    base = make_features(bars).x
    # rewrite everything after bar 400: features up to 400 must not move
    cut = 400
    noisy = bars.with_columns(
        pl.when(pl.int_range(pl.len()) > cut).then(pl.col(c) * 1.5).otherwise(pl.col(c)).alias(c)
        for c in ("close", "high", "low", "volume")
    )
    changed = make_features(noisy).x
    np.testing.assert_array_equal(base[: cut + 1], changed[: cut + 1])
    assert not np.allclose(np.nan_to_num(base[cut + 1 :]), np.nan_to_num(changed[cut + 1 :]))


def test_feature_values_and_warmup():
    bars = random_bars(300)
    fs = make_features(bars)
    assert fs.x.shape == (300, len(fs.names))
    close = bars["close"].to_numpy()
    i = fs.names.index("ret_24")
    assert fs.x[250, i] == pytest.approx(np.log(close[250] / close[226]))
    assert np.isnan(fs.x[23, i]) and np.isfinite(fs.x[24, i])
    # after the longest window everything is defined
    assert np.isfinite(fs.x[200:]).all()
    pos = fs.x[:, fs.names.index("range_pos_24")]
    assert np.nanmin(pos) >= 0 and np.nanmax(pos) <= 1


def test_forward_returns():
    close = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(forward_returns(close, 2)[:2], np.log([4.0, 4.0]))
    assert np.isnan(forward_returns(close, 2)[2:]).all()
    with pytest.raises(ValueError):
        forward_returns(close, 0)


def test_splits_are_purged_and_tile_the_out_of_sample_period():
    h = 24
    splits = walk_forward_splits(n=3000, train_size=1000, test_size=400, horizon=h, start=100)
    assert splits[0].test_range == (1100, 1500)
    for s in splits:
        # no label of a fitted row may reach into the next block
        assert s.train.max() + h < s.val.min()
        assert s.val.max() + h < s.test.min()
        assert s.train.min() >= 100
    # test windows are back to back and end at n
    for a, b in zip(splits, splits[1:], strict=False):
        assert a.test_range[1] == b.test_range[0]
    assert splits[-1].test_range[1] == 3000
    with pytest.raises(ValueError):
        walk_forward_splits(n=3000, train_size=50, test_size=10, horizon=40)


def test_lstm_windows_line_up_with_rows():
    x = np.arange(20, dtype=float).reshape(10, 2)
    w = windows(x, np.array([3, 9]), seq_len=3)
    assert w.shape == (2, 3, 2)
    np.testing.assert_array_equal(w[0, :, 0], [2, 4, 6])  # rows 1, 2, 3
    np.testing.assert_array_equal(w[1, -1], x[9])
    with pytest.raises(ValueError):
        windows(x, np.array([1]), seq_len=3)


def test_proba_to_weights():
    p = np.array([np.nan, 0.6, 0.4, 0.52, 0.49])
    np.testing.assert_array_equal(proba_to_weights(p), [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(
        proba_to_weights(p, "long_short", threshold=0.05), [0, 1, -1, 0, 0]
    )
    # smoothing is trailing: the first value only sees itself
    sm = proba_to_weights(np.array([0.9, 0.1, 0.1, 0.9]), smooth=2)
    np.testing.assert_array_equal(sm, [1, 0, 0, 0])
    with pytest.raises(ValueError):
        proba_to_weights(p, "sideways")


def test_prediction_metrics_on_a_perfect_and_a_useless_signal():
    rng = np.random.default_rng(0)
    r = rng.normal(size=2000)
    perfect = prediction_metrics(1 / (1 + np.exp(-r)), r)
    assert perfect["auc"] == pytest.approx(1.0) and perfect["ic"] == pytest.approx(1.0)
    noise = prediction_metrics(rng.uniform(size=2000), r)
    assert abs(noise["auc"] - 0.5) < 0.05
    flat = prediction_metrics(np.full(2000, 0.51), r)
    assert flat["ic"] == 0.0 and flat["auc"] == pytest.approx(0.5)


def _planted(n=4000, seed=0):
    """Feature 0 at t decides the sign of the forward return; the rest is noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    fwd = 0.5 * np.sign(x[:, 0]) + rng.normal(0, 0.5, n)
    fwd[-5:] = np.nan
    return x, fwd


@pytest.mark.parametrize("kind", ["lgbm", "lstm"])
def test_walk_forward_finds_a_planted_signal(kind):
    x, fwd = _planted()
    splits = walk_forward_splits(len(x), 1500, 800, horizon=5, start=50)
    run = run_walk_forward(x, fwd, splits, kind, trials=2, names=[f"f{i}" for i in range(5)])
    oos = splits[0].test_range[0]
    assert np.isnan(run.proba[:oos]).all() and np.isfinite(run.proba[oos:]).all()
    m = prediction_metrics(run.proba[oos:], fwd[oos:])
    assert m["auc"] > 0.75
    assert len(run.folds) == len(splits)
    if kind == "lgbm":
        assert next(iter(run.importance)) == "f0"  # the planted feature dominates the gain


@pytest.mark.parametrize("model", ["lgbm", "lstm"])
def test_cli_writes_a_report(engine, tmp_path, capsys, model):
    from quantlab.data.store import upsert_bars
    from quantlab.model import run

    upsert_bars(engine, random_bars(1_200, seed=3), "coinbase", "ETH-USD", "1h")
    run.main(
        [
            "--db", str(engine.url),
            "--model", model,
            "--train-days", "20",
            "--test-days", "10",
            "--trials", "1",
            "--out", str(tmp_path),
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert "buy and hold" in out and "before costs" in out
    report = json.loads((tmp_path / f"model_ETH-USD_{model}_h24.json").read_text())
    assert report["prediction"]["rows"] > 0
    assert report["args"]["db"] is None  # no connection string in the report
    # 1200 bars - 169 warm-up - 480 train, in 240-bar test windows
    assert len(report["folds"]) == 3


def test_unknown_model_kind():
    x, fwd = _planted(400)
    with pytest.raises(ValueError):
        run_walk_forward(x, fwd, walk_forward_splits(400, 200, 100, 5), "xgboost")
