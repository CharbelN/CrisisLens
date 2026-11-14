"""Forward mode: "what does this news do to my portfolio?"

Pipeline (no user-interface code here, so it can be tested):
1. The translator turns the text into market moves (AI step, checked by code).
2. The risk engine applies the moves to the portfolio (plain code).
3. Historical VaR / ES over the same 20-day horizon gives context:
   how does this scenario compare with a bad month in normal history?

Every number shown to the user comes from step 2 or 3, never from the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from riskdesk.checks.moves import HORIZON_DAYS, Limits
from riskdesk.engine.portfolio import Portfolio, StressResult, apply_moves
from riskdesk.engine.var import VarResult, historical_var_es
from riskdesk.llm.client import LLMUnavailableError
from riskdesk.llm.translator import ChatClient, TranslationResult, translate

VAR_LEVELS = (0.95, 0.99)


@dataclass
class ForwardResult:
    status: str  # "ok", "translation_failed", "llm_unavailable", "manual"
    message: str
    translation: TranslationResult | None
    stress: StressResult | None
    var: dict[float, VarResult]


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
) -> ForwardResult:
    """Translate the text, then calculate. Never raises for an AI problem."""
    try:
        tr = translate(text, client, limits)
    except LLMUnavailableError as exc:
        return ForwardResult(
            "llm_unavailable",
            f"The language model is not available ({exc}). You can type moves by hand and press Recalculate.",
            None, None, {},
        )
    if not tr.ok:
        last = "; ".join(tr.errors[-1]) if tr.errors else "unknown problem"
        return ForwardResult(
            "translation_failed",
            f"The model could not produce valid moves after {tr.attempts} attempts (last problem: {last}). "
            "Try rephrasing, or type moves by hand and press Recalculate.",
            tr, None, {},
        )
    stress, var = compute(portfolio, tr.moves, changes)
    return ForwardResult("ok", "Moves estimated by the AI and checked by code.", tr, stress, var)


def run_manual(portfolio: Portfolio, moves: Mapping[str, float], changes: pd.DataFrame) -> ForwardResult:
    """Recalculate from moves typed or edited by the user (no AI involved)."""
    stress, var = compute(portfolio, moves, changes)
    return ForwardResult("manual", "Calculated from the moves in the table (edited by you, no AI).", None, stress, var)
