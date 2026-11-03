import json

import pytest

from quantlab.backtest import bench, run
from quantlab.data.store import upsert_bars
from tests.conftest import make_bars


def test_run_prints_strategy_and_buy_and_hold(engine, capsys):
    upsert_bars(engine, make_bars(60), "coinbase", "BTC-USD", "1h")
    run.main(["--db", str(engine.url), "--fast", "3", "--slow", "8"])
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{") :])
    assert set(report) == {"strategy", "buy_and_hold"}
    # make_bars rises steadily, so holding it must make money
    assert report["buy_and_hold"]["total_return"] > 0
    assert "60 bars" in out


def test_run_without_data_says_what_to_do(engine):
    with pytest.raises(SystemExit, match="ingest"):
        run.main(["--db", str(engine.url)])


def test_bench_smoke(capsys):
    bench.main(["--bars", "5000", "--event-bars", "500"])
    out = capsys.readouterr().out
    assert "numba vectorized" in out and "numba vs python loop" in out
