"""The demo flows: exactly the calls the app makes for the demo, so their answers can be cached.

Cache keys are a hash of the full prompt, so a cached answer is only found if the app asks
exactly the same question. Translation prompts depend on the headline only; the memo and the
backward-mode stories also depend on the portfolio. The demo therefore uses the sample portfolio
as it appears in the app's table.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from riskdesk.backward import run_backward
from riskdesk.checks.levels import latest_levels
from riskdesk.checks.moves import HORIZON_DAYS, plausible_limits
from riskdesk.config import PROJECT_ROOT
from riskdesk.data.library import load_library
from riskdesk.data.market import driver_changes, load_market_data
from riskdesk.forward import run_forward
from riskdesk.llm.memo import write_memo
from riskdesk.llm.translator import ChatClient
from riskdesk.retrieval.index import Retriever
from riskdesk.ui import parse_portfolio, portfolio_table

SAMPLE_IDS = ["S01", "S02", "S03", "S06", "S10"]  # the sample buttons in the app
DEMO_TARGETS = [20.0]  # backward mode: the default target in the app


def sample_headlines() -> list[str]:
    scenarios = pd.read_csv(PROJECT_ROOT / "eval" / "scenarios.csv", dtype=str, keep_default_na=False)
    return scenarios.set_index("id").loc[SAMPLE_IDS, "scenario"].tolist()


@dataclass
class FlowResult:
    name: str
    ok: bool  # True if the AI parts worked (no fallback, no template, no dropped story)
    detail: str


def run_demo_flows(client: ChatClient) -> list[FlowResult]:
    levels = load_market_data()
    changes = driver_changes(levels, HORIZON_DAYS)
    limits, start = plausible_limits(levels), latest_levels(levels)
    retriever, library = Retriever.load(), load_library()
    portfolio = parse_portfolio(portfolio_table())  # the sample portfolio, as the app reads it

    out = []
    for text in sample_headlines():
        res = run_forward(portfolio, text, client, limits, changes, retriever, start)
        out.append(FlowResult(f"Run: {text}", res.status == "ok", res.status))
        if res.stress is None:
            continue
        for plant in (False, True):
            memo = write_memo(res, text, client, plant)
            out.append(FlowResult(f"  memo{' (planted number)' if plant else ''}", memo.source == "ai", memo.note))
    for target in DEMO_TARGETS:
        back = run_backward(portfolio, target, changes, limits, library, client)
        st = back.stories
        ok = back.status == "ok" and st is not None and not st.dropped
        out.append(FlowResult(f"Backward: lose {target:g}%", ok, st.note if st else back.message))
    return out
