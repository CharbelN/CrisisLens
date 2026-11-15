"""Crisis library: each past event in data/events.csv plus what markets actually did.

Window rule (the same for every event):
- start = the last trading day BEFORE the event date (so the event day's own
  reaction is included),
- end   = 20 trading days after the start.
Moves are in engine units: percent for prices, basis points for the yield.
A driver with no data at the start (bitcoin before Sept 2014) is left empty (n/a).

The library also stores the FIRST-DAY reaction (start -> next trading day) in
columns "d1_<driver>". It is the cleanest read of the shock itself: the 20-day
window also holds unrelated later news (data releases, policy moves, reversals).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from riskdesk.checks.moves import HORIZON_DAYS
from riskdesk.config import PROJECT_ROOT
from riskdesk.engine.drivers import DRIVERS

EVENTS_PATH = PROJECT_ROOT / "data" / "events.csv"
LIBRARY_PATH = PROJECT_ROOT / "data" / "crisis_library.csv"
RISK_TYPES = ("market", "credit", "operational", "liquidity")
DAY1_PREFIX = "d1_"
EVENT_COLS = ["id", "name", "date", "description", "risk_type", "region", "checked"]


def load_events(path: Path = EVENTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in EVENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"events file is missing column(s): {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def event_window(levels: pd.DataFrame, date: pd.Timestamp, horizon: int = HORIZON_DAYS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) trading days for an event, by the window rule above."""
    before = levels.index[levels.index < date]
    if len(before) == 0:
        raise ValueError(f"no trading day before {date.date()} in the market data")
    start_pos = levels.index.get_loc(before[-1])
    end_pos = start_pos + horizon
    if end_pos >= len(levels.index):
        raise ValueError(f"fewer than {horizon} trading days of data after {date.date()}")
    return levels.index[start_pos], levels.index[end_pos]


def event_moves(levels: pd.DataFrame, date: pd.Timestamp, horizon: int = HORIZON_DAYS) -> dict[str, float | None]:
    start, end = event_window(levels, date, horizon)
    # fill short gaps (e.g. a FRED holiday for oil), never across the series start
    filled = levels.ffill(limit=5)
    out: dict[str, float | None] = {}
    for d in DRIVERS:
        a, b = filled.at[start, d.id], filled.at[end, d.id]
        if pd.isna(a) or pd.isna(b):
            out[d.id] = None
        elif d.unit == "pct":
            out[d.id] = round((b / a - 1) * 100, 2)
        else:
            out[d.id] = round((b - a) * 100, 1)
    return out


def build_library(events: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ev in events.iterrows():
        start, end = event_window(levels, ev["date"])
        rows.append(
            {
                "id": ev["id"], "name": ev["name"], "date": ev["date"].date().isoformat(),
                "description": ev["description"], "risk_type": ev["risk_type"], "region": ev["region"],
                "checked": ev["checked"], "window_start": start.date().isoformat(),
                "window_end": end.date().isoformat(), **event_moves(levels, ev["date"]),
                **{DAY1_PREFIX + k: v for k, v in event_moves(levels, ev["date"], horizon=1).items()},
            }
        )
    return pd.DataFrame(rows)


def load_library(path: Path = LIBRARY_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"id": str}, keep_default_na=True)


def check_events(events: pd.DataFrame, levels: pd.DataFrame) -> list[str]:
    """Mechanical problems in the event list (dates, ids, types, lengths)."""
    problems: list[str] = []
    dup = events["id"][events["id"].duplicated()].tolist()
    if dup:
        problems.append(f"duplicate ids: {dup}")
    for _, ev in events.iterrows():
        tag = f"{ev['id']} ({ev['name']})"
        if ev["risk_type"] not in RISK_TYPES:
            problems.append(f"{tag}: risk_type '{ev['risk_type']}' is not one of {RISK_TYPES}")
        words = len(ev["description"].split())
        if not 15 <= words <= 60:
            problems.append(f"{tag}: description has {words} words (want 15-60)")
        if ev["checked"] not in ("yes", "no"):
            problems.append(f"{tag}: checked must be yes or no")
        try:
            event_window(levels, ev["date"])
        except ValueError as exc:
            problems.append(f"{tag}: {exc}")
    return problems
