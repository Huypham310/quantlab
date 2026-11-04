"""Event-driven backtest: the strategy sees one bar at a time and trades real units with cash.

Much slower than the vectorized engine, but structurally unable to look ahead (the strategy only
ever receives the history up to the current bar), so it's the reference the fast engine is
checked against.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from quantlab.backtest.costs import CostModel

# strategy(history_of_closes_up_to_now) -> target weight for the next bar
Strategy = Callable[[np.ndarray], float]


@dataclass
class Fill:
    bar: int
    units: float  # + bought, - sold
    price: float
    cost: float


@dataclass
class Portfolio:
    cash: float
    units: float = 0.0
    fills: list[Fill] = field(default_factory=list)

    def value(self, price: float) -> float:
        return self.cash + self.units * price


def run_event_driven(
    close: np.ndarray,
    strategy: Strategy,
    costs: CostModel,
    initial_cash: float = 10_000.0,
    band: float = 0.0,
) -> tuple[np.ndarray, Portfolio]:
    """Returns the equity after each bar and the final portfolio (with every fill)."""
    pf = Portfolio(cash=initial_cash)
    equity = np.empty(len(close))
    for t in range(len(close)):
        price = float(close[t])
        # read-only view: the strategy physically can't see bars after t
        history = close[: t + 1]
        history.flags.writeable = False
        target_w = float(strategy(history))

        eq = pf.value(price)
        # fraction of equity actually held right now, after the market moved
        current_w = pf.units * price / eq if eq > 0 else 0.0
        dw = target_w - current_w
        # same rule as the vectorized engine: no trade inside the band, float noise isn't a trade
        if abs(dw) > band and abs(dw) > 1e-12 and eq > 0:
            # costs come out of equity first, then the remaining equity is split by weight
            cost = abs(dw) * costs.rate * eq
            eq_after = eq - cost
            new_units = target_w * eq_after / price
            traded = new_units - pf.units
            pf.cash -= traded * price + cost
            pf.units = new_units
            pf.fills.append(Fill(t, traded, price, cost))
        equity[t] = pf.value(price)
    return equity, pf
