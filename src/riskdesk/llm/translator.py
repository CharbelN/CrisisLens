"""Scenario translator: a headline or what-if sentence -> a fixed set of market moves.

Flow:
1. Build a prompt that lists the 6 risk drivers, their units and the 20-day horizon.
2. Ask the language model for JSON.
3. Check the answer (checks/moves.py). If it fails, send the problems back to the
   model and ask again, at most MAX_RETRIES times.
4. If it still fails, return status "failed" with the reasons. The app must then
   tell the user; it never invents moves on its own. (Step 5 adds a fallback:
   the average moves of the retrieved past events.)

The moves are the model's educated guesses, not forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from riskdesk.checks.moves import HORIZON_DAYS, Limits, check_translation
from riskdesk.engine.drivers import DRIVERS
from riskdesk.llm.client import LLMReply, Messages

MAX_RETRIES = 2


class ChatClient(Protocol):
    def chat_json(self, messages: Messages) -> LLMReply: ...


@dataclass
class TranslationResult:
    status: str  # "ok" or "failed"
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


def _driver_lines(limits: dict[str, Limits]) -> str:
    lines = []
    for d in DRIVERS:
        unit = "percent change (-10 means -10%)" if d.unit == "pct" else "change in basis points (+50 means +0.50 percentage points)"
        lim = limits[d.id]
        lines.append(f'- "{d.id}": {d.label}, {unit}. Allowed range {lim.low} to {lim.high}.')
    return "\n".join(lines)


def build_messages(text: str, limits: dict[str, Limits]) -> Messages:
    system = f"""You are a market risk analyst. You turn a news headline or what-if sentence into
your best estimate of how six market risk drivers would move over the next {HORIZON_DAYS} trading days
(about one month) if it happened. These are educated guesses, not forecasts.

Risk drivers (use exactly these keys):
{_driver_lines(limits)}

Rules:
- Give a number for every driver. Use 0 if you expect no meaningful change.
- Think about direct effects and the usual knock-on effects (flight to safety, inflation, central bank reaction).
- Size the moves by comparing with real past episodes you know.
- Plain numbers only: no units, no ranges, no strings.

Answer with JSON only, in exactly this shape:
{{"moves": {{"equity": 0, "oil": 0, "gold": 0, "usd": 0, "btc": 0, "us10y": 0}},
 "reasoning": "two or three sentences explaining the main moves",
 "events_used": []}}
"events_used" lists the names of the past events you relied on; it may be empty."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Scenario: {text.strip()}"},
    ]


def translate(text: str, client: ChatClient, limits: dict[str, Limits]) -> TranslationResult:
    if not text or not text.strip():
        raise ValueError("Scenario text is empty.")

    messages = build_messages(text, limits)
    result = TranslationResult(status="failed", moves=None, reasoning="", events_used=[], attempts=0)

    for _ in range(1 + MAX_RETRIES):
        reply = client.chat_json(messages)
        result.attempts += 1
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        result.latency_s += reply.latency_s
        result.model = reply.model

        check = check_translation(reply.text, limits)
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
    return result
