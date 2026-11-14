"""The fixed list of risk drivers and the rules for a valid set of market moves.

A "move" is how much one driver changes in a scenario:
- price drivers (stocks, oil, gold, dollar, bitcoin) move in PERCENT: -10.0 means -10%.
- the US 10-year yield moves in BASIS POINTS: +50.0 means +0.50 percentage points.

A set of moves is a dict such as {"equity": -8.0, "oil": 40.0, "us10y": 50.0}.
Drivers left out are treated as "no change" (0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

Unit = Literal["pct", "bp"]


@dataclass(frozen=True)
class Driver:
    id: str
    label: str
    unit: Unit
    source: str


DRIVERS: tuple[Driver, ...] = (
    Driver("equity", "US stocks (S&P 500)", "pct", "yfinance ^GSPC"),
    Driver("oil", "Oil (Brent)", "pct", "FRED DCOILBRENTEU"),
    Driver("gold", "Gold", "pct", "yfinance GC=F"),
    Driver("usd", "US dollar index", "pct", "yfinance DX-Y.NYB"),
    Driver("btc", "Bitcoin", "pct", "yfinance BTC-USD"),
    Driver("us10y", "US 10-year yield", "bp", "FRED DGS10"),
)

DRIVER_IDS: tuple[str, ...] = tuple(d.id for d in DRIVERS)
DRIVERS_BY_ID: dict[str, Driver] = {d.id: d for d in DRIVERS}

# Hard limits: moves outside these are impossible or absurd, not just extreme.
# (Stage 3 adds tighter "plausible" limits for the language model's guesses.)
MIN_PRICE_MOVE_PCT = -100.0  # a price cannot fall more than 100%
MAX_PRICE_MOVE_PCT = 1000.0
MAX_ABS_RATE_MOVE_BP = 2000.0  # 20 percentage points


class InvalidMovesError(ValueError):
    """Raised when a set of market moves cannot be used."""


def validate_moves(moves: Mapping[str, float]) -> dict[str, float]:
    """Check a set of moves and return a complete copy (missing drivers = 0)."""
    unknown = set(moves) - set(DRIVER_IDS)
    if unknown:
        raise InvalidMovesError(f"Unknown risk driver(s): {sorted(unknown)}")

    clean: dict[str, float] = {}
    for driver in DRIVERS:
        raw = moves.get(driver.id, 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise InvalidMovesError(f"{driver.id}: move must be a number, got {raw!r}") from None
        if not math.isfinite(value):
            raise InvalidMovesError(f"{driver.id}: move must be a finite number, got {raw!r}")
        if driver.unit == "pct":
            if value <= MIN_PRICE_MOVE_PCT:
                raise InvalidMovesError(f"{driver.id}: a price cannot fall {value}% (limit is above -100%)")
            if value > MAX_PRICE_MOVE_PCT:
                raise InvalidMovesError(f"{driver.id}: move {value}% is above the {MAX_PRICE_MOVE_PCT}% limit")
        elif abs(value) > MAX_ABS_RATE_MOVE_BP:
            raise InvalidMovesError(f"{driver.id}: move {value} bp is beyond ±{MAX_ABS_RATE_MOVE_BP} bp")
        clean[driver.id] = value
    return clean
