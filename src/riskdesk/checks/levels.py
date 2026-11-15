"""Stated levels: when a scenario names a level ("oil jumps to $150"), code computes the move.

The model is told today's starting levels and reports, in "stated_levels", only the
levels the scenario text itself states (e.g. {"oil": 150}). Code then:
1. checks each stated level really appears as a number in the scenario text
   (so the model cannot invent one),
2. computes the exact move from the starting level:
   prices: (level / start - 1) x 100 percent;  the yield (in %): (level - start) x 100 bp,
3. compares it with the model's move. A move further off than LEVEL_TOLERANCE is sent
   back to the model; if it is still off after the retries, code puts in the computed move.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from riskdesk.engine.drivers import DRIVER_IDS, DRIVERS, DRIVERS_BY_ID

LEVEL_TOLERANCE = {"pct": 2.0, "bp": 10.0}  # percentage points for prices, bp for the yield

# "$150", "150", "30,000", "6.5%", "30k" (k = thousand)
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([kK]\b)?")


@dataclass(frozen=True)
class StartLevels:
    date: str  # latest date in the market data, e.g. "2026-11-13"
    levels: dict[str, float]  # price per driver; the yield in percent (5.10 = 5.10%)


def latest_levels(levels: pd.DataFrame) -> StartLevels:
    """Last known level of each driver (each series filled forward to the last date)."""
    last = levels.ffill().iloc[-1]
    return StartLevels(levels.index[-1].date().isoformat(), {d: float(last[d]) for d in DRIVER_IDS})


def numbers_in_text(text: str) -> list[float]:
    out = []
    for num, k in _NUMBER.findall(text):
        value = float(num.replace(",", ""))
        out.append(value * 1000 if k else value)
    return out


def implied_move(driver: str, start: float, level: float) -> float:
    if DRIVERS_BY_ID[driver].unit == "pct":
        return round((level / start - 1) * 100, 2)
    return round((level - start) * 100, 1)


@dataclass(frozen=True)
class LevelCheck:
    errors: list[str]  # problems to send back to the model
    fixes: dict[str, float]  # driver -> move computed by code, where the model's move is off
    used: dict[str, float]  # valid stated levels (driver -> level)


def check_levels(raw, moves: dict[str, float], start: StartLevels, text: str) -> LevelCheck:
    """Compare the model's moves with the moves implied by the levels the scenario states."""
    if raw in (None, {}):
        return LevelCheck([], {}, {})
    if not isinstance(raw, dict):
        return LevelCheck(['"stated_levels" must be an object such as {"oil": 150}, or {} if none.'], {}, {})

    errors: list[str] = []
    fixes: dict[str, float] = {}
    used: dict[str, float] = {}
    in_text = numbers_in_text(text)
    for driver, level in raw.items():
        if driver not in DRIVERS_BY_ID:
            errors.append(f"stated_levels: unknown driver '{driver}'. Use only {list(DRIVER_IDS)}.")
            continue
        if isinstance(level, bool) or not isinstance(level, (int, float)) or level <= 0:
            errors.append(f"stated_levels: {driver} must be a positive number, got {level!r}.")
            continue
        if not any(abs(level - n) <= 1e-6 * max(1.0, abs(n)) for n in in_text):
            errors.append(f"stated_levels: {driver} {level:g} is not a number stated in the scenario; "
                          "list only levels the scenario itself states.")
            continue
        used[driver] = float(level)
        d = DRIVERS_BY_ID[driver]
        target = implied_move(driver, start.levels[driver], float(level))
        if abs(moves[driver] - target) > LEVEL_TOLERANCE[d.unit]:
            fixes[driver] = target
            errors.append(f"{driver}: the scenario states {level:g}, and today's level is {start.levels[driver]:g}, "
                          f"so the move is {target:+g}{d.suffix}, not {moves[driver]:+g}{d.suffix}.")
    return LevelCheck(errors, fixes, used)


def start_levels_text(start: StartLevels) -> str:
    def fmt(d) -> str:
        v = start.levels[d.id]
        return f"{d.id} {v:.2f}%" if d.unit == "bp" else f"{d.id} {v:,.2f}"

    return ", ".join(fmt(d) for d in DRIVERS)
