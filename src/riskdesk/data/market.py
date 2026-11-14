"""Load the saved market data file and turn price levels into driver moves."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskdesk.config import PROJECT_ROOT
from riskdesk.engine.drivers import DRIVERS

MARKET_DATA_PATH = PROJECT_ROOT / "data" / "market_data.csv"


def load_market_data(path: Path = MARKET_DATA_PATH) -> pd.DataFrame:
    """Daily levels, one column per driver (yield in percent, e.g. 4.25)."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    missing = [d.id for d in DRIVERS if d.id not in df.columns]
    if missing:
        raise ValueError(f"market data is missing driver column(s): {missing}")
    return df[[d.id for d in DRIVERS]].sort_index()


def driver_changes(levels: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Moves of each driver over every `horizon_days` window, in engine units.

    Price drivers: percent change, (P[t+h] / P[t] - 1) x 100.
    Yield:         change in basis points, (y[t+h] - y[t]) x 100.
    Each row is labelled with the window's END date.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    out = pd.DataFrame(index=levels.index)
    for d in DRIVERS:
        col = levels[d.id]
        if d.unit == "pct":
            out[d.id] = (col / col.shift(horizon_days) - 1) * 100
        else:
            out[d.id] = (col - col.shift(horizon_days)) * 100
    return out.iloc[horizon_days:]
