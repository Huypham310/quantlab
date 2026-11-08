"""In-memory paper broker: orders, fills on live (or replayed) trades, positions, P&L.

Fill model (deliberately simple and on the conservative side):
  - market orders fill on the *next* trade for their symbol, at that price plus slippage;
  - a buy limit fills at its limit price once a trade prints at or below it (sell: at or above),
    i.e. as if it had been resting passively;
  - fees are charged on every fill's notional.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field

from quantlab.paper.risk import OrderRateLimiter, RiskLimits, check_order


@dataclass
class Order:
    id: int
    symbol: str
    side: str  # "buy" / "sell"
    qty: float
    type: str  # "market" / "limit"
    limit_price: float | None
    created_ms: int
    status: str = "open"  # open / filled / cancelled / rejected
    filled_qty: float = 0.0
    avg_price: float = 0.0
    reason: str = ""
    # a limit that already crossed the last price when it arrived takes liquidity like a market
    # order (filling at the print, capped by its limit) instead of resting at its limit
    marketable: bool = False

    @property
    def signed_open(self) -> float:
        left = self.qty - self.filled_qty
        return left if self.side == "buy" else -left


@dataclass
class Fill:
    order_id: int
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    ts_ms: int


@dataclass
class PaperBroker:
    cash: float = 100_000.0
    limits: RiskLimits = field(default_factory=RiskLimits)
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        self.positions: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.orders: dict[int, Order] = {}
        self.open: dict[int, Order] = {}
        self.fills: list[Fill] = []
        self.fees_paid = 0.0
        self.day_start_equity = self.cash
        self.killed = False
        # orders that would have pushed a position past its limit at fill time; must stay 0
        self.breaches = 0
        self._ids = itertools.count(1)
        self._rate = OrderRateLimiter(self.limits.max_orders_per_minute)

    # ---- state ---------------------------------------------------------------------------------

    def equity(self) -> float:
        return self.cash + sum(q * self.last.get(s, 0.0) for s, q in self.positions.items())

    def _open_signed(self, symbol: str, side: str) -> float:
        return sum(
            o.signed_open for o in self.open.values() if o.symbol == symbol and o.side == side
        )

    def _worst_gross_excluding(self, symbol: str) -> float:
        total = 0.0
        for s, px in self.last.items():
            if s == symbol:
                continue
            pos = self.positions.get(s, 0.0)
            worst = max(
                abs(pos + self._open_signed(s, "buy")), abs(pos + self._open_signed(s, "sell"))
            )
            total += worst * px
        return total

    def snapshot(self) -> dict:
        return {
            "cash": self.cash,
            "equity": self.equity(),
            "positions": {s: q for s, q in self.positions.items() if q != 0},
            "last": dict(self.last),
            "open_orders": [asdict(o) for o in self.open.values()],
            "fees_paid": self.fees_paid,
            "killed": self.killed,
            "breaches": self.breaches,
        }

    # ---- orders --------------------------------------------------------------------------------

    def submit(
        self,
        symbol: str,
        side: str,
        qty: float,
        type: str = "market",
        limit_price: float | None = None,
        now_ms: int | None = None,
    ) -> Order:
        """Risk-check and accept (status "open") or reject (status "rejected", with reason)."""
        if side not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        if type not in ("market", "limit") or (type == "limit") != (limit_price is not None):
            raise ValueError("limit orders need a limit price, market orders must not have one")
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        order = Order(next(self._ids), symbol, side, qty, type, limit_price, now_ms)
        decision = check_order(
            self.limits,
            symbol=symbol,
            side=side,
            qty=qty,
            limit_price=limit_price,
            last_price=self.last.get(symbol),
            position_qty=self.positions.get(symbol, 0.0),
            open_same_side_qty=self._open_signed(symbol, side),
            gross_other=self._worst_gross_excluding(symbol),
            equity=self.equity(),
            killed=self.killed,
            rate_ok=self._rate.allow(now_ms),
        )
        self.orders[order.id] = order
        if not decision.ok:
            order.status, order.reason = "rejected", decision.reason
            return order
        self._rate.record(now_ms)
        if limit_price is not None:
            last = self.last[symbol]
            order.marketable = limit_price >= last if side == "buy" else limit_price <= last
        self.open[order.id] = order
        return order

    def cancel(self, order_id: int, reason: str = "cancelled by user") -> bool:
        order = self.open.pop(order_id, None)
        if order is None:
            return False
        order.status, order.reason = "cancelled", reason
        return True

    # ---- market data ---------------------------------------------------------------------------

    def on_trade(self, symbol: str, price: float, ts_ms: int) -> list[Fill]:
        """Feed one trade print: fill what crosses, mark to market, check the kill switch."""
        fills = []
        if symbol in self.last:
            # market orders accepted before this print fill on it; limits fill if it crosses
            for order in [o for o in self.open.values() if o.symbol == symbol]:
                fill = self._try_fill(order, price, ts_ms)
                if fill is not None:
                    fills.append(fill)
        self.last[symbol] = price
        if not self.killed and self.equity() < self.day_start_equity * (
            1 - self.limits.max_daily_loss
        ):
            self.killed = True
            for oid in list(self.open):
                self.cancel(oid, "kill switch: daily loss limit")
        return fills

    def _try_fill(self, order: Order, price: float, ts_ms: int) -> Fill | None:
        slip = self.slippage_bps / 10_000
        taker = price * (1 + slip) if order.side == "buy" else price * (1 - slip)
        if order.type == "market":
            px = taker
        elif order.marketable:
            # takes liquidity: market-like fill, but never worse than the limit
            if order.side == "buy" and price <= order.limit_price:
                px = min(taker, order.limit_price)
            elif order.side == "sell" and price >= order.limit_price:
                px = max(taker, order.limit_price)
            else:
                return None  # the market moved past the limit before it could trade
        elif order.side == "buy" and price <= order.limit_price:
            px = order.limit_price  # resting: filled at its own price when the market comes to it
        elif order.side == "sell" and price >= order.limit_price:
            px = order.limit_price
        else:
            return None

        qty = order.qty - order.filled_qty
        pos = self.positions.get(order.symbol, 0.0)
        signed = qty if order.side == "buy" else -qty
        if abs(pos + signed) > abs(pos):
            # increasing: never let the fill itself push the position past its limit (a market
            # order can fill worse than the price it was checked at)
            room = max(self.limits.max_position_notional / px - abs(pos), 0.0)
            if qty > room:
                qty = room
                signed = qty if order.side == "buy" else -qty
        if qty <= 1e-12:
            self.cancel(order.id, "no room left under the position limit")
            return None

        fee = qty * px * self.fee_bps / 10_000
        self.cash -= signed * px + fee
        self.fees_paid += fee
        new_pos = pos + signed
        self.positions[order.symbol] = new_pos
        # only fills that *grow* the position can breach: a short that's above the limit because
        # the price ran up must still be allowed to buy back (Hypothesis found that case)
        if abs(new_pos) > abs(pos) and abs(new_pos) * px > self.limits.max_position_notional * (
            1 + 1e-9
        ):
            self.breaches += 1
        order.avg_price = (order.avg_price * order.filled_qty + px * qty) / (order.filled_qty + qty)
        order.filled_qty += qty
        if order.filled_qty >= order.qty - 1e-12:
            order.status = "filled"
            self.open.pop(order.id, None)
        else:
            # clamped: the rest can't be filled safely. Status ends "cancelled" with
            # filled_qty > 0, i.e. partially filled
            self.cancel(order.id, "partially filled up to the position limit")
        fill = Fill(order.id, order.symbol, order.side, qty, px, fee, ts_ms)
        self.fills.append(fill)
        return fill

    def new_day(self) -> None:
        """Start-of-day reset: the daily loss limit is measured from here."""
        self.day_start_equity = self.equity()
        self.killed = False
