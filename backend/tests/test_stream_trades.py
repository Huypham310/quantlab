import pytest

from quantlab.stream.trades import Trade, parse_entry

ENTRY = {
    b"exchange": b"coinbase",
    b"symbol": b"BTC-USD",
    b"trade_id": b"42",
    b"price": b"64321.5",
    b"size": b"0.001",
    b"side": b"buy",
    b"ts_ms": b"1793556245123",
}


def test_parses_gateway_entry():
    assert parse_entry(ENTRY) == Trade(
        "coinbase", "BTC-USD", 42, 64321.5, 0.001, "buy", 1793556245123
    )


def test_accepts_str_keys_too():
    assert parse_entry({k.decode(): v.decode() for k, v in ENTRY.items()}).trade_id == 42


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("price", b"nan", "bad price"),
        ("price", b"0", "bad price"),
        ("price", b"inf", "bad price"),
        ("size", b"-1", "bad size"),
        ("side", b"hold", "bad side"),
        ("price", b"abc", "could not convert"),
        ("trade_id", b"1.5", "invalid literal"),
    ],
)
def test_rejects_bad_values(field, value, match):
    with pytest.raises(ValueError, match=match):
        parse_entry({**ENTRY, field.encode(): value})


def test_reports_missing_fields():
    entry = {k: v for k, v in ENTRY.items() if k not in (b"price", b"side")}
    with pytest.raises(ValueError, match="missing fields: price, side"):
        parse_entry(entry)
