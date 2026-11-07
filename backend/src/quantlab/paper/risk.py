"""Pre-trade risk checks for the paper broker.

Every check runs *before* an order is accepted, against the worst case: the current position
plus every open order on the same side as if they all filled. Fills that would still overshoot
(a market order filling at a worse price than the one it was checked at) are clamped by the
broker, so a limit can't be breached by construction; the tests check that invariant.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskLimits:
    max_order_notional: float = 5_000.0
    max_position_notional: float = 20_000.0  # per symbol, |qty| x price
    max_gross_leverage: float = 2.0  # sum of |position notional| / equity
    max_orders_per_minute: int = 120
    price_band: float = 0.05  # limit price must be within 5% of the last trade
    max_daily_loss: float = 0.05  # kill switch at -5% from the day's starting equity
    market_buffer: float = 0.01  # market orders are checked at last x (1 +/- 1%)
    symbols: frozenset[str] = frozenset({"BTC-USD", "ETH-USD"})


@dataclass(frozen=True)
class RiskDecision:
    ok: bool
    reason: str = ""


ACCEPT = RiskDecision(True)


@dataclass
class OrderRateLimiter:
    """Sliding 60-second window of accepted order timestamps (ms)."""

    limit: int
    window_ms: int = 60_000
    _times: deque[int] = field(default_factory=deque)

    def allow(self, now_ms: int) -> bool:
        while self._times and self._times[0] <= now_ms - self.window_ms:
            self._times.popleft()
        return len(self._times) < self.limit

    def record(self, now_ms: int) -> None:
        self._times.append(now_ms)


def check_order(
    limits: RiskLimits,
    *,
    symbol: str,
    side: str,
    qty: float,
    limit_price: float | None,
    last_price: float | None,
    position_qty: float,
    open_same_side_qty: float,
    gross_other: float,
    equity: float,
    killed: bool,
    rate_ok: bool,
) -> RiskDecision:
    """All pre-trade checks for one order. `gross_other` is the worst-case gross notional of
    every *other* symbol; `open_same_side_qty` is signed like the order (+ buy, - sell)."""
    signed = qty if side == "buy" else -qty
    reducing = (
        position_qty != 0
        and (signed > 0) != (position_qty > 0)
        and abs(signed) <= abs(position_qty)
    )
    if symbol not in limits.symbols:
        return RiskDecision(False, f"symbol {symbol} not allowed")
    if not qty > 0:
        return RiskDecision(False, "quantity must be positive")
    if last_price is None:
        return RiskDecision(False, "no price yet for this symbol")
    if killed and not reducing:
        return RiskDecision(False, "daily loss limit hit: only reducing orders allowed")
    if not rate_ok:
        return RiskDecision(False, "order rate limit")
    if limit_price is not None:
        if abs(limit_price / last_price - 1) > limits.price_band:
            return RiskDecision(False, f"limit price outside the {limits.price_band:.0%} band")
        ref = limit_price
    else:
        # market orders: assume a fill a bit worse than the last trade
        ref = last_price * (1 + limits.market_buffer)
    if qty * ref > limits.max_order_notional:
        return RiskDecision(False, "order notional above limit")
    worst = abs(position_qty + open_same_side_qty + signed) * ref
    if not reducing and worst > limits.max_position_notional:
        return RiskDecision(False, "position limit")
    if not reducing and gross_other + worst > limits.max_gross_leverage * max(equity, 0.0):
        return RiskDecision(False, "gross leverage limit")
    return ACCEPT
