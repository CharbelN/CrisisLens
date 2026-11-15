"""Risk memo: the large model writes a short plain-language memo from the calculated figures only.

Steps:
1. Code writes a facts sheet from the calculation (risk engine + VaR) and the past events relied on.
   It also keeps the list of allowed numbers, dates and event ids.
2. The model fills fixed sections (JSON). It must copy numbers from the facts and not calculate.
3. Code checks the answer: valid JSON, every section present, the briefing short enough to read
   aloud in about 30 seconds, every number and date in the facts, every cited event id in the facts.
4. If a check fails, the model is asked once more with the problems listed.
   If it fails again (or the model is unavailable), code fills a fixed template from the same facts.

The caveat ("AI estimates, not advice") is added by code, so it is always there.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from riskdesk.checks.levels import numbers_in_text
from riskdesk.checks.moves import HORIZON_DAYS
from riskdesk.checks.numbers import check_numbers, extract_numbers
from riskdesk.engine.drivers import DRIVERS
from riskdesk.engine.holdings import HOLDINGS_BY_ID
from riskdesk.forward import VAR_LEVELS, ForwardResult
from riskdesk.llm.client import LLMUnavailableError
from riskdesk.llm.translator import ChatClient, format_moves

MEMO_RETRIES = 1  # the brief: regenerate once, then use the template
BRIEFING_MAX_WORDS = 80  # about 30 seconds read aloud

SECTIONS = (
    ("briefing", "Briefing"),
    ("what_happens", "What happens"),
    ("biggest_hits", "Biggest hits"),
    ("history", "Compared with history"),
    ("past_events", "Past events relied on"),
)
SECTION_KEYS = tuple(k for k, _ in SECTIONS)
CAVEAT = ("The market moves are AI estimates, not forecasts, and this is not investment advice. "
          "All money figures were calculated by plain code.")


def money(x: float) -> str:
    return f"${abs(x):,.0f}"


def signed_money(x: float) -> str:
    if round(x) == 0:
        return "$0"
    return f"-${abs(x):,.0f}" if x < 0 else f"+${x:,.0f}"


def pct(x: float) -> str:
    return f"{x + 0.0:+.2f}%" if round(x, 2) != 0 else "0.00%"  # never "-0.00%"


# --- 1. facts ----------------------------------------------------------------
@dataclass
class Facts:
    lines: list[str]  # the facts sheet given to the model
    numbers: list[float]  # every number the memo may contain
    dates: set[str]
    event_ids: set[str]
    template: dict[str, str]  # the code-written memo, same sections


def _source(result: ForwardResult) -> str:
    if result.status == "manual":
        return "typed or edited by the user"
    if result.status == "fallback":
        return "average real moves of the closest past events (the AI's answer was not used)"
    return "estimated by the AI from the scenario and past events, checked by code"


def build_facts(result: ForwardResult, scenario: str) -> Facts:
    s = result.stress
    if s is None:
        raise ValueError("No calculation to write a memo about.")
    nums: list[float] = [HORIZON_DAYS, s.start_value, s.pnl, s.pnl_pct, s.end_value]
    nums += numbers_in_text(scenario)
    dates: set[str] = set()
    word = "loss" if s.pnl < 0 else "gain"

    lines = [
        f'Scenario: "{scenario}"' if scenario else "Scenario: market moves typed by hand",
        f"Where the market moves come from: {_source(result)}",
        f"Horizon: {HORIZON_DAYS} trading days (about one month)",
        f"Portfolio value at the start: {money(s.start_value)}",
        f"Result: {word} of {money(s.pnl)} ({pct(s.pnl_pct)} of the portfolio); value after: {money(s.end_value)}",
        "Result per holding (amount, return, profit or loss), largest loss first:",
    ]
    ranked = sorted(s.holdings, key=lambda h: h.pnl)
    for h in ranked:
        nums += [h.amount, h.return_pct, h.pnl]
        lines.append(f"- {HOLDINGS_BY_ID[h.holding_id].label}: {money(h.amount)}, {pct(h.return_pct)}, "
                     f"{signed_money(h.pnl)}")
    lines.append("Market moves used:")
    for d in DRIVERS:
        v = s.moves[d.id]
        nums.append(v)
        lines.append(f"- {d.label}: {v:+g}{d.suffix}")

    hist_line = "No historical comparison is available."
    v99 = result.var.get(0.99)
    if result.var and v99 is not None:
        v95 = result.var.get(0.95)
        nums += [v99.n_scenarios, 1] + [lvl * 100 for lvl in VAR_LEVELS]
        dates |= {v99.first_date, v99.last_date}
        lines.append(f"History: today's portfolio replayed over {v99.n_scenarios:,} past windows of "
                     f"{HORIZON_DAYS} trading days ({v99.first_date} to {v99.last_date}):")
        for v, what in ((v95, "95% VaR (a bad month)"), (v99, "99% VaR (a very bad month)")):
            if v is not None:
                nums += [v.var, v.var_pct]
                lines.append(f"- {what}: loss of {money(v.var)} ({v.var_pct:.2f}%)")
        nums += [v99.es, v99.es_pct]
        lines.append(f"- 99% Expected Shortfall (average of the worst 1%): loss of {money(v99.es)} ({v99.es_pct:.2f}%)")
        if s.pnl < 0 and v99.var > 0:
            ratio = round(s.loss / v99.var, 1)
            nums.append(ratio)
            lines.append(f"- This scenario's loss is {ratio:.1f}x the 99% VaR")
            hist_line = (f"This loss is {ratio:.1f}x the 99% VaR of {money(v99.var)}: a very bad month in the past "
                         f"{v99.n_scenarios:,} windows lost {money(v99.var)} or more only 1% of the time.")
        else:
            hist_line = (f"In history, a bad month (95% VaR) would lose {money(v95.var if v95 else v99.var)} and a very "
                         f"bad month (99% VaR) {money(v99.var)}; this scenario is a {word} instead.")

    used = set(result.translation.events_used) if result.translation else set()
    twins = [t for t in result.twins if t.id in used]
    if twins:
        lines.append("Past events relied on (real market moves):")
        for t in twins:
            dates.add(t.date)
            nums += [v for v in t.moves.values() if v is not None] + numbers_in_text(t.name)
            line = f"- [{t.id}] {t.name} ({t.date})"
            if t.day1 is not None:
                nums += [v for v in t.day1.values() if v is not None]
                line += f": first day {format_moves(t.day1)};"
            lines.append(line + f" over {HORIZON_DAYS} trading days {format_moves(t.moves)}")
    else:
        lines.append("Past events relied on: none")

    worst = ranked[0]
    template = {
        "briefing": (f"In this scenario the portfolio would {'lose' if s.pnl < 0 else 'gain'} {money(s.pnl)}, "
                     f"or {abs(s.pnl_pct):.2f}% of {money(s.start_value)}, over about a month. "
                     f"The biggest effect is on {HOLDINGS_BY_ID[worst.holding_id].label} "
                     f"({signed_money(worst.pnl)})."),
        "what_happens": (f"Scenario: {scenario or 'moves typed by hand'}. The market moves are "
                         f"{_source(result)}: " + ", ".join(
                             f"{d.label} {s.moves[d.id]:+g}{d.suffix}" for d in DRIVERS) + "."),
        "biggest_hits": "; ".join(f"{HOLDINGS_BY_ID[h.holding_id].label}: {pct(h.return_pct)}, {signed_money(h.pnl)}"
                                  for h in ranked) + ".",
        "history": hist_line,
        "past_events": ("; ".join(f"{t.name} [{t.id}] ({t.date})" for t in twins) + "."
                        if twins else "No past event was relied on."),
    }
    return Facts(lines, nums, dates, {t.id for t in twins}, template)


# --- 2. prompt -----------------------------------------------------------------
def build_messages(facts: Facts) -> list[dict[str, str]]:
    keys = ", ".join(f'"{k}"' for k in SECTION_KEYS)
    system = f"""You write a short risk memo for an ordinary investor, in plain language.

Use ONLY the facts below. Rules:
- Every number you write must be copied from the facts, rounded the same way or less precisely
  (you may write 6.4% for 6.43%). Do not calculate, add, subtract or convert anything.
- No other numbers: no forecasts, no probabilities, no counts that are not in the facts.
- Write dates exactly as in the facts (YYYY-MM-DD). Refer to past events by name with their id, like [E013].
- The scenario has not happened: write what WOULD happen ("the portfolio would lose").
  Say "loss" or "gain" correctly; the moves are estimates, not forecasts.

Reply with JSON only: one object with exactly these keys: {keys}.
Each value is ONE plain text string of full sentences (not a list, not an object).
- "briefing": 2 or 3 sentences, at most {BRIEFING_MAX_WORDS} words, to be read aloud: the result and the main reason.
- "what_happens": the scenario and the market moves that matter most, in 2 or 3 sentences.
- "biggest_hits": the holdings that lose or gain most, with their amounts.
- "history": how this result compares with the historical VaR and Expected Shortfall.
- "past_events": which past events the moves rely on, and for each one the one or two real moves
  that matter most for this scenario (or say none were used). Do not list every move.

FACTS
""" + "\n".join(facts.lines)
    return [{"role": "system", "content": system}, {"role": "user", "content": "Write the memo."}]


# --- 3. checks -----------------------------------------------------------------
_EVENT_ID = re.compile(r"\bE\d{3}\b")


def check_memo(text: str, facts: Facts) -> tuple[dict[str, str] | None, list[str]]:
    """(sections, problems). Sections is None if the JSON itself is unusable."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, ["The answer is not valid JSON."]
    if not isinstance(data, dict):
        return None, ["The answer must be a JSON object."]
    problems = []
    sections: dict[str, str] = {}
    for k in SECTION_KEYS:
        v = data.get(k)
        if isinstance(v, (list, dict)):
            problems.append(f'Section "{k}" must be one text string of sentences, not a list or object.')
        elif not isinstance(v, str) or not v.strip():
            problems.append(f'Section "{k}" is missing or empty.')
        else:
            sections[k] = v.strip()
    if problems:
        return None, problems
    n_words = len(sections["briefing"].split())
    if n_words > BRIEFING_MAX_WORDS:
        problems.append(f"The briefing has {n_words} words; keep it to {BRIEFING_MAX_WORDS} or fewer.")
    for k in SECTION_KEYS:
        nc = check_numbers(sections[k], facts.numbers, facts.dates)
        if nc.wrong_numbers:
            problems.append(f'Section "{k}": these numbers are not in the facts: {", ".join(nc.wrong_numbers)}.')
        if nc.wrong_dates:
            problems.append(f'Section "{k}": these dates are not in the facts: {", ".join(nc.wrong_dates)}.')
        unknown = sorted(set(_EVENT_ID.findall(sections[k])) - facts.event_ids)
        if unknown:
            problems.append(f'Section "{k}": these event ids are not among the past events relied on: {", ".join(unknown)}.')
    return sections, problems


# --- 4. write, check, retry, template -----------------------------------------
@dataclass
class Memo:
    sections: dict[str, str]
    source: str  # "ai" (passed the checks) or "template" (filled by code)
    note: str  # one line for the screen: how the memo was produced and checked
    attempts: int = 0
    problems: list[list[str]] = field(default_factory=list)  # problems found per rejected draft
    planted: str | None = None  # demo: the wrong number planted in the first draft
    model: str = ""
    latency_s: float = 0.0
    n_numbers: int = 0  # numbers checked in the final memo

    @property
    def caveat(self) -> str:
        return CAVEAT


def plant_wrong_number(text: str) -> tuple[str, str | None]:
    """Demo: change the first money amount in the "biggest_hits" section, to show the checker at work."""
    try:
        data = json.loads(text)
        section = data["biggest_hits"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return text, None
    if not isinstance(section, str):
        return text, None
    numbers, _ = extract_numbers(section)
    target = next((w for w in numbers if w.text.startswith("$")), numbers[0] if numbers else None)
    if target is None:
        return text, None
    wrong = f"${target.value * 1.37 + 113:,.0f}"  # far from any rounding of the real number
    data["biggest_hits"] = section.replace(target.text, wrong, 1)
    return json.dumps(data), f"{target.text} changed to {wrong}"


def _count_numbers(sections: dict[str, str]) -> int:
    return sum(sum(len(x) for x in extract_numbers(v)) for v in sections.values())


def write_memo(result: ForwardResult, scenario: str, client: ChatClient, plant: bool = False) -> Memo:
    """Ask the model for the memo, check every number; never raises for an AI problem."""
    facts = build_facts(result, scenario)
    messages = build_messages(facts)
    memo = Memo(facts.template, "template", "")
    for attempt in range(1 + MEMO_RETRIES):
        try:
            reply = client.chat_json(messages)
        except LLMUnavailableError as exc:
            memo.note = f"The language model is not available ({exc}), so code filled the memo template."
            break
        memo.attempts += 1
        memo.model, memo.latency_s = reply.model, memo.latency_s + reply.latency_s
        text = reply.text
        if plant and attempt == 0:
            text, memo.planted = plant_wrong_number(text)
        sections, problems = check_memo(text, facts)
        if sections is not None and not problems:
            memo.sections, memo.source = sections, "ai"
            fixed = f"after {attempt} correction(s)" if attempt else "on the first try"
            memo.note = f"Written by the AI and passed the number check {fixed}."
            break
        memo.problems.append(problems)
        problem_list = "\n".join(f"- {p}" for p in problems)
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"Your memo has these problems:\n{problem_list}\n"
                                        "Rewrite it using only numbers from the facts. Reply with JSON only."},
        ]
    else:
        memo.note = (f"The AI's memo failed the number check {memo.attempts} times, "
                     "so code filled the memo template from the calculated figures.")
    memo.n_numbers = _count_numbers(memo.sections)
    return memo
