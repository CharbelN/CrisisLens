"""Sanity check for the market moves a language model proposes.

Two layers:
1. Format: the answer must be JSON with a "moves" object holding a number for
   every one of the 6 risk drivers (no missing, no extra).
2. Plausibility: each move must lie inside a band set from history. The band is
   the worst and best 20-trading-day move seen in market_data.csv, widened by 1.5x
   to leave room for a crisis bigger than any in the data. A price can never fall
   by 100% or more, so price moves are also kept above -95%.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pandas as pd

from riskdesk.data.market import driver_changes, load_market_data
from riskdesk.engine.drivers import DRIVER_IDS, DRIVERS, InvalidMovesError, validate_moves

HORIZON_DAYS = 20
WIDEN_FACTOR = 1.5
PRICE_FLOOR_PCT = -95.0


@dataclass(frozen=True)
class Limits:
    low: float
    high: float


def plausible_limits(
    levels: pd.DataFrame, horizon_days: int = HORIZON_DAYS, widen: float = WIDEN_FACTOR
) -> dict[str, Limits]:
    """Allowed band per driver from the worst/best historical move over `horizon_days`."""
    changes = driver_changes(levels, horizon_days)
    limits: dict[str, Limits] = {}
    for d in DRIVERS:
        col = changes[d.id].dropna()
        low, high = float(col.min()) * widen, float(col.max()) * widen
        if d.unit == "pct":
            low = max(low, PRICE_FLOOR_PCT)
        limits[d.id] = Limits(round(low, 1), round(high, 1))
    return limits


def default_limits() -> dict[str, Limits]:
    return plausible_limits(load_market_data())


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    moves: dict[str, float] | None
    reasoning: str
    events_used: list[str]
    errors: list[str]


def check_translation(
    raw_text: str, limits: dict[str, Limits], allowed_events: set[str] | None = None
) -> CheckResult:
    """Read the model's answer and list every problem in plain words.

    `allowed_events`: ids of the retrieved past events. If given, every id in
    "events_used" must be one of them (the model may only cite evidence it was shown).
    """
    errors: list[str] = []
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return CheckResult(False, None, "", [], [f"Answer is not valid JSON ({exc.msg})."])
    if not isinstance(data, dict) or not isinstance(data.get("moves"), dict):
        return CheckResult(False, None, "", [], ['Answer must be a JSON object with a "moves" object.'])

    raw_moves = data["moves"]
    missing = [d for d in DRIVER_IDS if d not in raw_moves]
    extra = [k for k in raw_moves if k not in DRIVER_IDS]
    if missing:
        errors.append(f"Missing driver(s): {missing}. Give a number for every driver, 0 if no change.")
    if extra:
        errors.append(f"Unknown driver(s): {extra}. Use only {list(DRIVER_IDS)}.")

    moves: dict[str, float] = {}
    for d in DRIVERS:
        if d.id not in raw_moves:
            continue
        value = raw_moves[d.id]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{d.id}: must be a plain number, got {value!r}.")
            continue
        lim = limits[d.id]
        unit = d.suffix
        if not lim.low <= value <= lim.high:
            errors.append(
                f"{d.id}: {value}{unit} is outside the plausible range "
                f"{lim.low}{unit} to {lim.high}{unit} for {HORIZON_DAYS} trading days."
            )
            continue
        moves[d.id] = float(value)

    if not errors:
        try:
            moves = validate_moves(moves)  # engine's hard limits, as a final guard
        except InvalidMovesError as exc:
            errors.append(str(exc))

    reasoning = data.get("reasoning", "")
    events = data.get("events_used", [])
    reasoning = reasoning if isinstance(reasoning, str) else str(reasoning)
    if not isinstance(events, list):
        errors.append('"events_used" must be a list of event ids.')
        events = []
    events = [str(e) for e in events]
    if allowed_events is not None:
        bad = [e for e in events if e not in allowed_events]
        if bad:
            errors.append(f"events_used contains ids that were not given as evidence: {bad}. Use only {sorted(allowed_events)}.")

    if errors:
        return CheckResult(False, None, reasoning, events, errors)
    return CheckResult(True, moves, reasoning, events, [])
