"""Performance metrics from an equity curve."""

from dataclasses import asdict, dataclass

import numpy as np

from quantlab.data.bars import TIMEFRAMES

SECONDS_PER_YEAR = 365 * 24 * 3600  # crypto trades every day of the year


@dataclass
class Metrics:
    total_return: float
    cagr: float
    volatility: float  # annualized std of per-bar returns
    sharpe: float  # annualized, risk-free rate 0
    sortino: float
    max_drawdown: float  # as a positive fraction, 0.25 = -25% from the peak
    turnover: float  # sum of |weight changes|
    trades: int
    exposure: float  # fraction of bars with a position

    def as_dict(self) -> dict:
        return asdict(self)


def bars_per_year(timeframe: str) -> float:
    return SECONDS_PER_YEAR / TIMEFRAMES[timeframe]


def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peaks)) if equity.size else 0.0


def compute_metrics(
    equity: np.ndarray,
    returns: np.ndarray,
    weights: np.ndarray,
    turnover: np.ndarray,
    trades: int,
    timeframe: str,
    initial_cash: float,
) -> Metrics:
    periods = bars_per_year(timeframe)
    n = len(equity)
    total = float(equity[-1] / initial_cash - 1.0)
    years = n / periods
    # a wiped-out account has no meaningful CAGR, report -100%
    cagr = float((equity[-1] / initial_cash) ** (1 / years) - 1.0) if equity[-1] > 0 else -1.0

    std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(returns))
    sharpe = mean / std * np.sqrt(periods) if std > 0 else 0.0
    # downside deviation counts every bar, positive ones as zero
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
    sortino = mean / downside * np.sqrt(periods) if downside > 0 else 0.0

    return Metrics(
        total_return=total,
        cagr=cagr,
        volatility=std * float(np.sqrt(periods)),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_drawdown(equity),
        turnover=float(np.sum(turnover)),
        trades=int(trades),
        exposure=float(np.mean(weights != 0)),
    )
