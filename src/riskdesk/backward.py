"""Backward mode (reverse stress test): "what would have to happen for me to lose X%?"

Pipeline (no user-interface code here, so it can be tested):
1. Search (plain code): sets of market moves that make the portfolio lose exactly X%,
   preferring the most plausible by history (engine/reverse.py).
2. Match (plain code): the past event whose real moves were closest to each set.
3. Narrate (AI): each set written as a short news story (llm/stories.py).
4. Check (plain code): every number in a story is one of its moves; the main moves are stated.
   Wrong stories are rewritten once, then dropped.
Every number shown comes from step 1 (the risk engine), never from the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from riskdesk.checks.moves import Limits
from riskdesk.engine.portfolio import Portfolio, portfolio_value
from riskdesk.engine.reverse import ReverseError, ReverseResult, reverse_stress
from riskdesk.llm.stories import StoriesResult, write_stories
from riskdesk.llm.translator import ChatClient


@dataclass
class BackwardResult:
    status: str  # "ok", "no_stories", "no_scenarios", "error"
    message: str
    reverse: ReverseResult | None
    stories: StoriesResult | None
    start_value: float = 0.0


def run_backward(
    portfolio: Portfolio, target_pct: float, changes: pd.DataFrame, limits: dict[str, Limits],
    library: pd.DataFrame | None, client: ChatClient | None,
) -> BackwardResult:
    """Never raises for an AI problem; a bad portfolio or target gives status "error"."""
    try:
        start = portfolio_value(portfolio)
        rev = reverse_stress(portfolio, target_pct, changes, limits, library)
    except ReverseError as exc:
        return BackwardResult("error", str(exc), None, None)
    if not rev.scenarios:
        return BackwardResult("no_scenarios", " ".join(rev.notes), rev, None, start)
    if client is None:
        return BackwardResult("no_stories", "Scenarios found by code (stories are switched off).", rev, None, start)
    stories = write_stories(rev.scenarios, client)
    status = "ok" if stories.stories else "no_stories"
    msg = (f"{len(rev.scenarios)} scenario(s) that lose {target_pct:g}% found by code; "
           "stories written by the AI and checked by code.")
    return BackwardResult(status, msg, rev, stories, start)
