from datetime import timedelta

import httpx
import pytest

from quantlab.data.sources.coinbase import CoinbaseClient, parse_candles
from quantlab.data.sources.kraken import KrakenClient, parse_ohlc, to_kraken_pair
from tests.conftest import T0

EPOCH = int(T0.timestamp())


def test_parse_coinbase_reorders_columns():
    # coinbase rows are [time, low, high, open, close, volume], newest first
    df = parse_candles([[EPOCH + 3600, 9, 12, 10, 11, 5.0], [EPOCH, 8, 11, 9, 10, 4.0]])
    assert df["ts"].to_list()[0] == T0
    assert df.row(0) == (T0, 9.0, 11.0, 8.0, 10.0, 4.0)


def test_parse_coinbase_empty():
    assert parse_candles([]).is_empty()


def test_coinbase_paginates_in_300_candle_windows():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["start"])
        return httpx.Response(200, json=[[EPOCH, 1, 2, 1.5, 1.5, 1.0]])

    client = CoinbaseClient(httpx.Client(transport=httpx.MockTransport(handler)), pause=0)
    client.client.base_url = "https://example.test"
    # 700 hourly bars -> windows of 300, 300, 100
    client.fetch_bars("BTC-USD", "1h", T0, T0 + timedelta(hours=700))
    assert len(calls) == 3


def test_coinbase_empty_range_makes_no_requests():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not call the API")

    client = CoinbaseClient(httpx.Client(transport=httpx.MockTransport(handler)), pause=0)
    assert client.fetch_bars("BTC-USD", "1h", T0, T0).is_empty()


def test_kraken_pair_aliases():
    assert to_kraken_pair("BTC-USD") == "XBTUSD"
    assert to_kraken_pair("ETH-USD") == "ETHUSD"


def test_parse_kraken_uses_whatever_pair_key():
    payload = {
        "error": [],
        "result": {
            "XXBTZUSD": [[EPOCH, "10", "12", "9", "11", "10.5", "3.5", 7]],
            "last": EPOCH,
        },
    }
    df = parse_ohlc(payload)
    assert df.row(0) == (T0, 10.0, 12.0, 9.0, 11.0, 3.5)


def test_parse_kraken_raises_on_api_error():
    with pytest.raises(RuntimeError, match="Unknown asset pair"):
        parse_ohlc({"error": ["EQuery:Unknown asset pair"], "result": {}})


def test_kraken_drops_forming_candle():
    rows = [[EPOCH + i * 3600, "1", "2", "1", "1.5", "1.5", "1", 1] for i in range(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": [], "result": {"XXBTZUSD": rows, "last": 0}})

    client = KrakenClient(
        httpx.Client(transport=httpx.MockTransport(handler), base_url="https://k")
    )
    df = client.fetch_bars("BTC-USD", "1h", T0, T0 + timedelta(days=1))
    # last row is the candle that's still open
    assert df.height == 2
