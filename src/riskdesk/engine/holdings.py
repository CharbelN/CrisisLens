"""The menu of holdings a user can own, and how each one reacts to the risk drivers.

The links are kept deliberately simple so each one can be explained in a sentence:
- Stocks, oil, gold and bitcoin follow their driver 1:1 (driver +10% -> holding +10%).
- The US bond fund uses duration: return ~ -duration x change in yield.
  With duration 7, a +50 bp (+0.50 point) yield rise gives about -3.5%.
  This is the standard first-order (linear) approximation.
- Cash does not move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

BOND_FUND_DURATION = 7.0


@dataclass(frozen=True)
class Holding:
    id: str
    label: str
    driver: str | None  # risk driver it follows 1:1, if any
    duration: float | None = None  # for bond holdings: sensitivity to us10y

    @property
    def drivers_used(self) -> tuple[str, ...]:
        if self.duration is not None:
            return ("us10y",)
        return (self.driver,) if self.driver else ()


HOLDINGS: tuple[Holding, ...] = (
    Holding("us_stocks", "US stocks (S&P 500 fund)", driver="equity"),
    Holding("us_bonds", "US bond fund", driver=None, duration=BOND_FUND_DURATION),
    Holding("gold", "Gold", driver="gold"),
    Holding("oil", "Oil", driver="oil"),
    Holding("bitcoin", "Bitcoin", driver="btc"),
    Holding("cash", "Cash (USD)", driver=None),
)

HOLDINGS_BY_ID: dict[str, Holding] = {h.id: h for h in HOLDINGS}


def _move(moves: Mapping[str, Any], driver_id: str) -> Any:
    return moves[driver_id] if driver_id in moves else 0.0


def holding_return(holding: Holding, moves: Mapping[str, Any]) -> Any:
    """Return of one holding as a decimal (-0.10 = -10%).

    `moves` can be a dict of numbers (one scenario) or a DataFrame whose columns
    are drivers (many scenarios at once, used for VaR). Same formula either way.
    """
    if holding.duration is not None:
        return -holding.duration * _move(moves, "us10y") / 10_000  # bp -> decimal
    if holding.driver is not None:
        return _move(moves, holding.driver) / 100  # percent -> decimal
    return 0.0 * _move(moves, "equity")  # cash: zero, same shape as the input
