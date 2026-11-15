"""Number checker for the risk memo: every number written in the memo must be a calculated one.

How it works:
1. Code builds the list of allowed numbers from the calculation (loss, per-holding results,
   moves, VaR / ES, the past events' real moves, the horizon, the scenario's own numbers).
2. Code pulls every number out of the memo text ("$3,214", "-6.43%", "1.4x", "50 bp", "2.5k").
3. A written number passes if it equals an allowed number after rounding to the number of
   decimals it is written with: "$3,214" matches 3,213.6; "6.4%" matches -6.43; "6%" matches -6.43.
   Signs are ignored ("a loss of $3,214" and "-$3,214" are the same number).
Dates are checked separately: an ISO date (2008-09-15) or a year must be one given in the facts.

What it does NOT check: that a number is attached to the right label. "Gold -12%" passes if
-12% is any calculated number (for example the stock move). The check catches invented and
mistyped numbers, which is the failure that matters most for a memo.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

# Names that contain digits but are not figures.
_NAME_PHRASES = re.compile(r"S&P\s*500|10-year|\b10y\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR = re.compile(r"(?<![\w.,$])(19\d{2}|20\d{2})(?![\w.,%])")
# A number not glued to a letter on the left (so the 013 in "E013" is skipped):
# optional $, digits with optional thousands commas, optional decimals, optional k / thousand / million.
_NUMBER = re.compile(
    r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s*(k|thousand|million)\b)?", re.IGNORECASE
)
_SCALE = {"k": 1e3, "thousand": 1e3, "million": 1e6}


@dataclass(frozen=True)
class Written:
    """A number as written in the text."""

    text: str  # e.g. "$3,214" or "6.4"
    value: float  # e.g. 3214.0
    step: float  # the rounding step it is written to, e.g. 1 for "3,214", 0.1 for "6.4", 100 for "3.2k"


# Models often write typographic dashes and special spaces (non-breaking hyphen in "10-year", minus sign).
_DASHES = (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212)  # hyphens, en/em dash, minus sign
_SPACES = (0x00A0, 0x202F, 0x2009)  # non-breaking, narrow and thin spaces
_PLAIN = {c: "-" for c in _DASHES} | {c: " " for c in _SPACES}


def extract_numbers(text: str) -> tuple[list[Written], list[str]]:
    """(numbers, dates) found in the text. Dates are ISO dates and stand-alone years."""
    text = _NAME_PHRASES.sub(" ", text.translate(_PLAIN))
    dates = _ISO_DATE.findall(text)
    text = _ISO_DATE.sub(" ", text)
    dates += _YEAR.findall(text)
    text = _YEAR.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(text):
        whole, decimals, word = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        scale = _SCALE.get(word, 1.0)
        step = 10.0 ** -(len(decimals) - 1) if decimals else 1.0
        out.append(Written(m.group(0).strip(), float(whole.replace(",", "") + decimals) * scale, step * scale))
    return out, dates


def matches(written: Written, allowed: float) -> bool:
    """True if `allowed` rounds to the written number at the precision it is written with."""
    return abs(written.value - abs(allowed)) <= written.step / 2 + 1e-9


@dataclass(frozen=True)
class NumberCheck:
    wrong_numbers: list[str]  # written numbers that match no calculated number
    wrong_dates: list[str]

    @property
    def ok(self) -> bool:
        return not self.wrong_numbers and not self.wrong_dates


def check_numbers(text: str, allowed: Iterable[float], allowed_dates: Iterable[str]) -> NumberCheck:
    values = [a for a in allowed if a is not None and math.isfinite(a)]
    dates = set(allowed_dates) | {d[:4] for d in allowed_dates}  # a date's year is allowed on its own
    numbers, found_dates = extract_numbers(text)
    wrong = [w.text for w in numbers if not any(matches(w, a) for a in values)]
    return NumberCheck(wrong, [d for d in found_dates if d not in dates])
