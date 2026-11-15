"""Forward mode: "what does this news do to my portfolio?"

Pipeline (no user-interface code here, so it can be tested):
0. The retriever finds the most similar past events ("historical twins").
1. The translator turns the text into market moves (AI step, checked by code),
   using the twins and their real moves as evidence.
2. The risk engine applies the moves to the portfolio (plain code).
3. Historical VaR / ES over the same 20-day horizon gives context:
   how does this scenario compare with a bad month in normal history?

Every number shown to the user comes from step 2 or 3, never from the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import pandas as pd

from riskdesk.checks.levels import StartLevels
from riskdesk.checks.moves import HORIZON_DAYS, Limits
from riskdesk.engine.portfolio import Portfolio, StressResult, apply_moves
from riskdesk.engine.var import VarResult, historical_var_es
from riskdesk.llm.client import LLMUnavailableError
from riskdesk.llm.translator import (
    FALLBACK_NOTE, STRONG_MATCH, ChatClient, TranslationResult, fallback_moves, translate,
)
from riskdesk.retrieval.index import Twin

VAR_LEVELS = (0.95, 0.99)


class Searcher(Protocol):
    def search(self, text: str) -> list[Twin]: ...


@dataclass
class ForwardResult:
    status: str  # "ok", "fallback", "translation_failed", "llm_unavailable", "manual", "error"
    message: str
    translation: TranslationResult | None
    stress: StressResult | None
    var: dict[float, VarResult]
    twins: list[Twin] = field(default_factory=list)


def var_context(portfolio: Portfolio, changes: pd.DataFrame) -> dict[float, VarResult]:
    return {lvl: historical_var_es(portfolio, changes, HORIZON_DAYS, lvl) for lvl in VAR_LEVELS}


def compute(portfolio: Portfolio, moves: Mapping[str, float], changes: pd.DataFrame) -> tuple[StressResult, dict[float, VarResult]]:
    return apply_moves(portfolio, moves), var_context(portfolio, changes)


def run_forward(
    portfolio: Portfolio,
    text: str,
    client: ChatClient,
    limits: dict[str, Limits],
    changes: pd.DataFrame,
    retriever: Searcher | None = None,
    start: StartLevels | None = None,
) -> ForwardResult:
    """Retrieve twins, translate the text, then calculate. Never raises for an AI problem."""
    twins = retriever.search(text) if retriever is not None else []
    try:
        tr = translate(text, client, limits, twins, start)
    except LLMUnavailableError as exc:
        fb = fallback_moves(twins, limits)
        if fb is not None:
            stress, var = compute(portfolio, fb, changes)
            tr = TranslationResult("fallback", fb, "The language model is unavailable. " + FALLBACK_NOTE,
                                   [t.id for t in twins if t.similarity >= STRONG_MATCH], 0)
            return ForwardResult("fallback", f"The language model is not available ({exc}), so the moves come from "
                                 "the closest past events where they agree (fallback).", tr, stress, var, twins)
        return ForwardResult(
            "llm_unavailable",
            f"The language model is not available ({exc}), and there are not enough close past events that agree "
            "to fall back on. You can type moves by hand and press Recalculate.",
            None, None, {}, twins,
        )
    if not tr.has_moves:
        last = "; ".join(tr.errors[-1]) if tr.errors else "unknown problem"
        return ForwardResult(
            "translation_failed",
            f"The model could not produce valid moves after {tr.attempts} attempts (last problem: {last}). "
            "No close past event was found to fall back on. Try rephrasing, or type moves by hand and press Recalculate.",
            tr, None, {}, twins,
        )
    stress, var = compute(portfolio, tr.moves, changes)
    if tr.status == "fallback":
        msg = (f"The model's answers failed the checks {tr.attempts} times, so the moves are the average real moves "
               "of the closest past events (fallback).")
        return ForwardResult("fallback", msg, tr, stress, var, twins)
    if twins:
        msg = "Moves estimated by the AI, grounded in past events, and checked by code."
    else:
        msg = "Moves estimated by the AI and checked by code."
    return ForwardResult("ok", msg, tr, stress, var, twins)


def run_manual(portfolio: Portfolio, moves: Mapping[str, float], changes: pd.DataFrame) -> ForwardResult:
    """Recalculate from moves typed or edited by the user (no AI involved)."""
    stress, var = compute(portfolio, moves, changes)
    return ForwardResult("manual", "Calculated from the moves in the table (edited by you, no AI).", None, stress, var)
