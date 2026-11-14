"""Historical Value-at-Risk (VaR) and Expected Shortfall (ES).

Method (historical simulation, as in the FRM curriculum):
1. Take every past window of `horizon` trading days from the market data.
2. Measure how each risk driver moved in that window.
3. Apply each past window's moves to TODAY's portfolio -> one profit/loss per window.
4. Sort the results. With n results and confidence level c, look at the
   k = ceil(n x (1 - c)) worst ones:
     VaR = the k-th worst loss   ("we lose more than this only k times in n")
     ES  = the average of those k worst losses ("how bad it is when it is bad")

Multi-day windows overlap (day 1-10, day 2-11, ...). This uses all the data but
makes neighbouring windows similar, which is a known limitation to mention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskdesk.engine.holdings import HOLDINGS_BY_ID, holding_return
from riskdesk.engine.portfolio import Portfolio, validate_portfolio


@dataclass(frozen=True)
class VarResult:
    horizon_days: int
    level: float  # e.g. 0.99
    var: float  # USD, positive number = loss
    es: float  # USD, positive number = loss
    var_pct: float
    es_pct: float
    n_scenarios: int
    first_date: str
    last_date: str


def scenario_pnl(portfolio: Portfolio, changes: pd.DataFrame) -> pd.Series:
    """Profit/loss of today's portfolio under each historical row of driver moves.

    Rows with missing data for a driver the portfolio needs are dropped
    (e.g. bitcoin before 2014 when the portfolio holds bitcoin).
    """
    clean = validate_portfolio(portfolio)
    needed = sorted({d for hid in clean for d in HOLDINGS_BY_ID[hid].drivers_used})
    usable = changes.dropna(subset=needed) if needed else changes
    pnl = pd.Series(0.0, index=usable.index)
    for hid, amount in clean.items():
        pnl = pnl + amount * holding_return(HOLDINGS_BY_ID[hid], usable)
    return pnl


def var_es_from_pnl(pnl: pd.Series | np.ndarray, level: float) -> tuple[float, float]:
    """VaR and ES (as positive loss numbers) from a list of profit/loss outcomes."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1, e.g. 0.99")
    values = np.sort(np.asarray(pnl, dtype=float))  # worst (most negative) first
    n = len(values)
    if n == 0:
        raise ValueError("No scenarios to compute VaR from.")
    k = max(1, math.ceil(round(n * (1 - level), 9)))  # round() avoids 5.000000001 -> 6
    tail = values[:k]
    return float(-tail[-1]), float(-tail.mean())


def historical_var_es(
    portfolio: Portfolio, changes: pd.DataFrame, horizon_days: int, level: float
) -> VarResult:
    """`changes` must hold `horizon_days`-day driver moves (see data.market.driver_changes)."""
    pnl = scenario_pnl(portfolio, changes)
    var, es = var_es_from_pnl(pnl, level)
    start = sum(validate_portfolio(portfolio).values())
    return VarResult(
        horizon_days=horizon_days,
        level=level,
        var=var,
        es=es,
        var_pct=var / start * 100,
        es_pct=es / start * 100,
        n_scenarios=len(pnl),
        first_date=str(pnl.index.min().date()) if len(pnl) else "",
        last_date=str(pnl.index.max().date()) if len(pnl) else "",
    )
