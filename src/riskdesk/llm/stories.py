"""Backward mode, steps 4-5: the large model writes each loss scenario as a short news story; code checks it.

The model gets each scenario's moves (found by plain code) and the closest past event as a reference.
Checks, per story:
- every number in the headline and story is one of that scenario's moves (after rounding),
  a number in the reference event's name, or the reference event's date / year;
- the moves that cause most of the loss are stated with their numbers (the reader must see the cause);
- the story is short enough.
Stories that fail are sent back once with their problems; a story that fails again is dropped
(the scenario itself, with its moves and loss, is still shown).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from riskdesk.checks.levels import numbers_in_text
from riskdesk.checks.moves import HORIZON_DAYS
from riskdesk.checks.numbers import check_numbers, extract_numbers, matches
from riskdesk.engine.drivers import DRIVERS, DRIVERS_BY_ID
from riskdesk.engine.holdings import HOLDINGS_BY_ID
from riskdesk.engine.reverse import ReverseScenario
from riskdesk.llm.client import LLMUnavailableError
from riskdesk.llm.translator import ChatClient

STORY_RETRIES = 1
N_MAIN_MOVES = 2  # the story must state the moves causing most of the loss
STORY_MAX_WORDS = 90
LETTERS = "ABCDEFGH"


def move_text(d_id: str, v: float | None) -> str:
    d = DRIVERS_BY_ID[d_id]
    if v is None:
        return f"{d.label}: no data"
    return f"{d.label} {v:+g}{'%' if d.unit == 'pct' else ' bp'}"


def contributions(sc: ReverseScenario) -> dict[str, float]:
    """Each driver's part of the profit or loss, in USD (from the engine's per-holding results)."""
    out = {d.id: 0.0 for d in DRIVERS}
    for h in sc.stress.holdings:
        drivers = HOLDINGS_BY_ID[h.holding_id].drivers_used
        if drivers:
            out[drivers[0]] += h.pnl
    return out


def main_moves(sc: ReverseScenario, n: int = N_MAIN_MOVES) -> list[str]:
    """The drivers whose moves cause the largest part of the loss (largest first)."""
    loss = {d: -c for d, c in contributions(sc).items() if c < -0.5}
    return sorted(loss, key=loss.get, reverse=True)[:n]


@dataclass
class StoryFacts:
    letter: str
    lines: list[str]  # what the model is told about this scenario
    numbers: list[float]  # numbers the story may contain
    dates: set[str]
    required: dict[str, float]  # driver -> move that must be stated


def story_facts(letter: str, sc: ReverseScenario) -> StoryFacts:
    required = {d: sc.moves[d] for d in main_moves(sc)}
    lines = [f"Scenario {letter}: market moves over the next {HORIZON_DAYS} trading days: "
             + "; ".join(move_text(d.id, sc.moves[d.id]) for d in DRIVERS if sc.moves[d.id] is not None)]
    lines.append("Moves that cause most of the loss (must be stated with their numbers): "
                 + "; ".join(move_text(d, v) for d, v in required.items()))
    numbers = [v for v in sc.moves.values() if v is not None]
    dates: set[str] = set()
    if sc.match is not None:
        lines.append(f"Closest past event (a reference for the kind of story): {sc.match.name} ({sc.match.date})")
        numbers += numbers_in_text(sc.match.name)
        dates.add(sc.match.date)
    return StoryFacts(letter, lines, numbers, dates, required)


def build_messages(facts: Sequence[StoryFacts]) -> list[dict[str, str]]:
    letters = ", ".join(f'"{f.letter}"' for f in facts)
    system = f"""You write short, believable financial news stories for a risk-management tool.
Each scenario below is a set of market moves over the next {HORIZON_DAYS} trading days. For each one, invent a
plausible cause (a crisis, a policy shock, a war...) and write it as a news story. Use the closest past
event as a reference for the kind of story, not as the same event. The story is hypothetical.

Rules:
- State the moves that cause most of the loss, with their numbers. You may state other moves too.
- Every number you write must be one of that scenario's moves, copied as given (you may round to fewer
  decimals). Write the yield move in bp. No other numbers: no prices, levels, dates, counts or
  percentages that are not moves. You may name the reference event and its year.
- Do not mention any portfolio or money amount.
- Each value is plain text. "headline": one line. "story": 2 to 4 sentences, at most {STORY_MAX_WORDS} words.

Reply with JSON only: {{"stories": [{{"id": "A", "headline": "...", "story": "..."}}]}}
with one story for each scenario id: {letters}.

""" + "\n\n".join("\n".join(f.lines) for f in facts)
    return [{"role": "system", "content": system}, {"role": "user", "content": "Write the stories."}]


def check_story(item: object, f: StoryFacts) -> list[str]:
    """Problems with one story (empty list = it passes)."""
    if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip()
                                             for k in ("headline", "story")):
        return [f'Story {f.letter}: needs a text "headline" and a text "story".']
    text = f"{item['headline']}\n{item['story']}"
    problems = []
    n_words = len(item["story"].split())
    if n_words > STORY_MAX_WORDS:
        problems.append(f"Story {f.letter}: {n_words} words; keep it to {STORY_MAX_WORDS} or fewer.")
    nc = check_numbers(text, f.numbers, f.dates)
    if nc.wrong_numbers:
        problems.append(f"Story {f.letter}: these numbers are not moves of this scenario: {', '.join(nc.wrong_numbers)}.")
    if nc.wrong_dates:
        problems.append(f"Story {f.letter}: these dates or years are not allowed: {', '.join(nc.wrong_dates)}.")
    written, _ = extract_numbers(text)
    missing = [move_text(d, v) for d, v in f.required.items() if not any(matches(w, v) for w in written)]
    if missing:
        problems.append(f"Story {f.letter}: must state these moves with their numbers: {'; '.join(missing)}.")
    return problems


@dataclass
class Story:
    letter: str
    headline: str
    text: str


@dataclass
class StoriesResult:
    stories: dict[str, Story]  # letter -> story that passed the checks
    note: str
    attempts: int = 0
    problems: list[list[str]] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # letters whose story failed twice
    model: str = ""


def _parse(text: str) -> dict[str, object] | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    items = data.get("stories") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    return {str(i.get("id", "")).strip(): i for i in items if isinstance(i, dict)}


def write_stories(scenarios: Sequence[ReverseScenario], client: ChatClient) -> StoriesResult:
    """One request for all stories, one more for any that fail; never raises for an AI problem."""
    facts = [story_facts(LETTERS[i], sc) for i, sc in enumerate(scenarios)]
    out = StoriesResult({}, "")
    todo = list(facts)
    messages = build_messages(todo)
    for _ in range(1 + STORY_RETRIES):
        try:
            reply = client.chat_json(messages)
        except LLMUnavailableError as exc:
            out.note = f"The language model is not available ({exc}), so the scenarios are shown without stories."
            return out
        out.attempts += 1
        out.model = reply.model
        items = _parse(reply.text)
        problems: list[str] = []
        failed: list[StoryFacts] = []
        if items is None:
            problems, failed = ['The answer is not valid JSON with a "stories" list.'], list(todo)
        for f in todo if items is not None else []:
            item = items.get(f.letter)
            p = check_story(item, f) if item is not None else [f"Story {f.letter} is missing."]
            if p:
                problems += p
                failed.append(f)
            else:
                out.stories[f.letter] = Story(f.letter, item["headline"].strip(), item["story"].strip())
        if not failed:
            break
        out.problems.append(problems)
        todo = failed
        problem_list = "\n".join(f"- {p}" for p in dict.fromkeys(problems))
        messages = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": f"These stories have problems:\n{problem_list}\nRewrite ONLY stories "
                                        f"{', '.join(f.letter for f in failed)}. Reply with JSON only, same format."},
        ]
    out.dropped = [f.letter for f in todo if f.letter not in out.stories]
    ok = len(out.stories)
    out.note = f"{ok} of {len(facts)} stories written by the AI passed the check"
    out.note += (f" ({out.attempts - 1} correction round)" if out.attempts > 1 else " on the first try")
    if out.dropped:
        out.note += f"; story {', '.join(out.dropped)} failed twice and was dropped"
    out.note += "."
    return out
