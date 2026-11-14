"""Scenario translator: a headline or what-if sentence -> a fixed set of market moves.

Flow:
1. Build a prompt that lists the 6 risk drivers, their units and the 20-day horizon.
   With RAG, the prompt also holds the retrieved past events ("historical twins")
   and what markets ACTUALLY did after each one, as evidence.
2. Ask the language model for JSON.
3. Check the answer (checks/moves.py). If it fails, send the problems back to the
   model and ask again, at most MAX_RETRIES times.
4. If it still fails:
   - with twins that are a real match (similarity >= STRONG_MATCH): use the
     similarity-weighted average of their real moves, clipped to the limits,
     and say so (status "fallback");
   - otherwise: status "failed" with the reasons. Moves are never invented.

The moves are the model's educated guesses, not forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from riskdesk.checks.moves import HORIZON_DAYS, Limits, check_translation
from riskdesk.engine.drivers import DRIVERS
from riskdesk.llm.client import LLMReply, Messages
from riskdesk.retrieval.index import Twin, twin_average

MAX_RETRIES = 2
STRONG_MATCH = 0.6  # bge-small cosine similarity; below this a twin is a weak match


class ChatClient(Protocol):
    def chat_json(self, messages: Messages) -> LLMReply: ...


@dataclass
class TranslationResult:
    status: str  # "ok", "fallback" or "failed"
    moves: dict[str, float] | None
    reasoning: str
    events_used: list[str]
    attempts: int
    errors: list[list[str]] = field(default_factory=list)  # problems found per failed attempt
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def has_moves(self) -> bool:
        return self.moves is not None


def _unit_word(d) -> str:
    return "%" if d.unit == "pct" else " bp"


def _driver_lines(limits: dict[str, Limits]) -> str:
    lines = []
    for d in DRIVERS:
        unit = "percent change (-10 means -10%)" if d.unit == "pct" else "change in basis points (+50 means +0.50 percentage points)"
        lim = limits[d.id]
        lines.append(f'- "{d.id}": {d.label}, {unit}. Allowed range {lim.low} to {lim.high}.')
    return "\n".join(lines)


def format_twin(t: Twin) -> str:
    moves = ", ".join(
        f"{d.id} {'n/a' if t.moves[d.id] is None else f'{t.moves[d.id]:+g}{_unit_word(d)}'}" for d in DRIVERS
    )
    strength = "strong match" if t.similarity >= STRONG_MATCH else "weak match"
    return f"[{t.id}] {t.name} ({t.date}; similarity {t.similarity:.2f}, {strength})\n  {t.description}\n  Real moves over the next {HORIZON_DAYS} trading days: {moves}"


def build_messages(text: str, limits: dict[str, Limits], twins: Sequence[Twin] | None = None) -> Messages:
    if twins:
        evidence = (
            "\n\nEvidence: the most similar past events found in our crisis library, with what markets "
            "actually did afterwards. Use them to size your moves, adjusting for how the new scenario "
            "differs (scale, cause, today's starting conditions). Ignore events that are not really comparable.\n"
            "Their moves are measured the same way (from the close before the event, over the same window), so "
            "they include later reversals and policy responses. An event that was a turning point (a peak or a "
            "bottom) shows the reversal, not the shock itself; never let evidence contradict what the scenario states.\n\n"
            + "\n\n".join(format_twin(t) for t in twins)
            + '\n\n"events_used" must list the ids (e.g. "E013") of the events you relied on, '
            "only from the list above; it may be empty if none are comparable."
        )
    else:
        evidence = '\n\n"events_used" lists the names of the past events you relied on; it may be empty.'

    system = f"""You are a market risk analyst. You turn a news headline or what-if sentence into
your best estimate of how six market risk drivers would move over the next {HORIZON_DAYS} trading days
(about one month) if it happened. These are educated guesses, not forecasts.

Risk drivers (use exactly these keys):
{_driver_lines(limits)}

Rules:
- The {HORIZON_DAYS}-day window starts at the market close just BEFORE the news. So any move the scenario
  itself describes (for example "oil jumps to $150") is part of your answer, plus what follows.
- Give a number for every driver. Use 0 if you expect no meaningful change.
- Think about direct effects and the usual knock-on effects (flight to safety, inflation, central bank reaction).
- Size the moves by comparing with real past episodes.
- Plain numbers only: no units, no ranges, no strings.

Answer with JSON only, in exactly this shape:
{{"moves": {{"equity": 0, "oil": 0, "gold": 0, "usd": 0, "btc": 0, "us10y": 0}},
 "reasoning": "two or three sentences explaining the main moves",
 "events_used": []}}{evidence}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Scenario: {text.strip()}"},
    ]


def fallback_moves(twins: Sequence[Twin], limits: dict[str, Limits]) -> dict[str, float] | None:
    strong = [t for t in twins if t.similarity >= STRONG_MATCH]
    if not strong:
        return None
    avg = twin_average(strong)
    return {d: round(min(max(v, limits[d].low), limits[d].high), 2) for d, v in avg.items()}


def translate(
    text: str, client: ChatClient, limits: dict[str, Limits], twins: Sequence[Twin] | None = None
) -> TranslationResult:
    if not text or not text.strip():
        raise ValueError("Scenario text is empty.")

    twins = list(twins or [])
    allowed = {t.id for t in twins} if twins else None
    messages = build_messages(text, limits, twins)
    result = TranslationResult(status="failed", moves=None, reasoning="", events_used=[], attempts=0)

    for _ in range(1 + MAX_RETRIES):
        reply = client.chat_json(messages)
        result.attempts += 1
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        result.latency_s += reply.latency_s
        result.model = reply.model

        check = check_translation(reply.text, limits, allowed)
        if check.ok:
            result.status = "ok"
            result.moves = check.moves
            result.reasoning = check.reasoning
            result.events_used = check.events_used
            return result

        result.errors.append(check.errors)
        problems = "\n".join(f"- {e}" for e in check.errors)
        messages = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": f"Your answer has these problems:\n{problems}\nReply again with corrected JSON only."},
        ]

    fb = fallback_moves(twins, limits)
    if fb is not None:
        strong = [t.id for t in twins if t.similarity >= STRONG_MATCH]
        result.status = "fallback"
        result.moves = fb
        result.events_used = strong
        result.reasoning = "The model's answers failed the checks, so these are the average real moves of the closest past events."
    return result
