"""Trade entries as the Go gateway writes them to the Redis Stream (see gateway/internal/sink)."""

import math
from dataclasses import dataclass

FIELDS = ("exchange", "symbol", "trade_id", "price", "size", "side", "ts_ms")


@dataclass(frozen=True, slots=True)
class Trade:
    exchange: str
    symbol: str
    trade_id: int
    price: float
    size: float
    side: str  # taker side, "buy" or "sell"
    ts_ms: int  # trade time, ms since epoch (UTC)


def _text(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else v


def parse_entry(fields: dict[bytes, bytes] | dict[str, str]) -> Trade:
    """Validate one stream entry. Raises ValueError on anything the bars can't use."""
    data = {_text(k): _text(v) for k, v in fields.items()}
    missing = [f for f in FIELDS if f not in data]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")

    price = float(data["price"])
    size = float(data["size"])
    # float("nan") parses fine, but one nan would poison a whole bar's high/low
    if not (math.isfinite(price) and price > 0):
        raise ValueError(f"bad price {data['price']!r}")
    if not (math.isfinite(size) and size > 0):
        raise ValueError(f"bad size {data['size']!r}")
    if data["side"] not in ("buy", "sell"):
        raise ValueError(f"bad side {data['side']!r}")

    return Trade(
        exchange=data["exchange"],
        symbol=data["symbol"],
        trade_id=int(data["trade_id"]),
        price=price,
        size=size,
        side=data["side"],
        ts_ms=int(data["ts_ms"]),
    )
