"""Trading costs, in basis points of traded notional."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Fee + slippage charged on every change of position.

    ``fee_bps`` is the exchange fee (Coinbase Advanced taker is 60 bps at the lowest tier, ~5-10
    bps for active traders); ``slippage_bps`` models crossing the spread and moving the book.
    Both are charged on |change in weight| x equity, so a full flip long -> short pays twice.
    """

    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs can't be negative")

    @property
    def rate(self) -> float:
        """Total cost as a fraction of traded notional."""
        return (self.fee_bps + self.slippage_bps) / 10_000


FREE = CostModel(0.0, 0.0)
