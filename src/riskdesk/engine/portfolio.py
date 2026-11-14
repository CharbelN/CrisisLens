"""Portfolio value and the loss under one set of market moves (a stress scenario)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from riskdesk.engine.drivers import validate_moves
from riskdesk.engine.holdings import HOLDINGS_BY_ID, holding_return

# A portfolio is {holding_id: amount in USD}, e.g. {"us_stocks": 10_000, "gold": 2_000}.
Portfolio = Mapping[str, float]


class InvalidPortfolioError(ValueError):
    """Raised when a portfolio cannot be used."""


def validate_portfolio(portfolio: Portfolio) -> dict[str, float]:
    unknown = set(portfolio) - set(HOLDINGS_BY_ID)
    if unknown:
        raise InvalidPortfolioError(f"Unknown holding(s): {sorted(unknown)}")
    clean: dict[str, float] = {}
    for hid, amount in portfolio.items():
        try:
            value = float(amount)
        except (TypeError, ValueError):
            raise InvalidPortfolioError(f"{hid}: amount must be a number, got {amount!r}") from None
        if not math.isfinite(value) or value < 0:
            raise InvalidPortfolioError(f"{hid}: amount must be zero or positive, got {amount!r}")
        if value > 0:
            clean[hid] = value
    if not clean:
        raise InvalidPortfolioError("Portfolio is empty: enter at least one positive amount.")
    return clean


def portfolio_value(portfolio: Portfolio) -> float:
    return sum(validate_portfolio(portfolio).values())


@dataclass(frozen=True)
class HoldingResult:
    holding_id: str
    amount: float
    return_pct: float  # e.g. -10.0 for -10%
    pnl: float  # profit and loss in USD; negative = loss


@dataclass(frozen=True)
class StressResult:
    start_value: float
    pnl: float  # negative = loss
    pnl_pct: float  # relative to start value, e.g. -7.83
    end_value: float
    holdings: tuple[HoldingResult, ...]
    moves: dict[str, float]

    @property
    def loss(self) -> float:
        """Loss as a positive number (0 if the scenario is a gain)."""
        return max(0.0, -self.pnl)


def apply_moves(portfolio: Portfolio, moves: Mapping[str, float]) -> StressResult:
    """Apply one set of market moves to the portfolio and compute the profit or loss."""
    clean_portfolio = validate_portfolio(portfolio)
    clean_moves = validate_moves(moves)

    rows = []
    for hid, amount in clean_portfolio.items():
        r = float(holding_return(HOLDINGS_BY_ID[hid], clean_moves))
        rows.append(HoldingResult(hid, amount, r * 100, amount * r))

    start = sum(clean_portfolio.values())
    pnl = sum(row.pnl for row in rows)
    return StressResult(
        start_value=start,
        pnl=pnl,
        pnl_pct=pnl / start * 100,
        end_value=start + pnl,
        holdings=tuple(rows),
        moves=clean_moves,
    )
