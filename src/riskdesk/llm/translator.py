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
# A move within this band counts as "flat" (no clear direction): prices in %, the yield in bp.
FLAT_BAND = {"pct": 2.0, "bp": 15.0}


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


def _fmt_moves(moves: dict[str, float | None]) -> str:
    return ", ".join(
        f"{d.id} {'n/a' if moves[d.id] is None else f'{moves[d.id]:+g}{_unit_word(d)}'}" for d in DRIVERS
    )


def format_twin(t: Twin) -> str:
    """One past event for the prompt: its first-day reaction and its 20-day move."""
    strength = "strong match" if t.similarity >= STRONG_MATCH else "weak match"
    lines = [f"[{t.id}] {t.name} ({t.date}; similarity {t.similarity:.2f}, {strength})", f"  {t.description}"]
    if t.day1 is not None:
        lines.append(f"  First-day reaction: {_fmt_moves(t.day1)}")
    lines.append(f"  Over the next {HORIZON_DAYS} trading days: {_fmt_moves(t.moves)}")
    return "\n".join(lines)


def typical_moves(twins: Sequence[Twin]) -> str:
    """Median 20-day move across the strong matches, with how many went up / down."""
    strong = [t for t in twins if t.similarity >= STRONG_MATCH]
    if not strong:
        return "No strong match was found, so size the moves from your own knowledge of past episodes."
    parts = []
    for d in DRIVERS:
        vals = sorted(t.moves[d.id] for t in strong if t.moves[d.id] is not None)
        if not vals:
            parts.append(f"{d.id} n/a")
            continue
        mid = len(vals) // 2
        med = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        up, down = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
        parts.append(f"{d.id} {med:+.1f}{_unit_word(d)} ({up} up, {down} down)")
    return (f"Typical {HORIZON_DAYS}-trading-day moves after the {len(strong)} strong match(es) "
            f"(median): " + ", ".join(parts))


# A 20-day window often holds unrelated later news (the FTX window holds a soft inflation report
# that cut the 10-year yield by 71 bp). The first-day reaction isolates the shock itself.
EVIDENCE_RULES = f"""How to use the evidence:
- Use the evidence to SIZE the moves, adjusting for scale and cause. Never copy one event's numbers.
- The first-day reaction is the cleanest read of each shock. The {HORIZON_DAYS}-day moves also hold later,
  unrelated news (data releases, policy moves, reversals at a peak or bottom).
- For a driver the scenario does not touch, prefer a small move or 0 over a move borrowed from the evidence."""


def build_messages(text: str, limits: dict[str, Limits], twins: Sequence[Twin] | None = None) -> Messages:
    if twins:
        evidence = (
            "\n\nEvidence: the most similar past events found in our crisis library, with what markets "
            "actually did. Moves are measured from the close before the event. Ignore events that are not "
            "really comparable.\n\n"
            + "\n\n".join(format_twin(t) for t in twins)
            + "\n\n" + typical_moves(twins)
            + "\n\n" + EVIDENCE_RULES
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


def direction_lock(moves: dict[str, float]) -> dict[str, str]:
    """"+" / "-" / "0" per driver from the answer WITHOUT evidence. "0" = within the flat band."""
    out = {}
    for d in DRIVERS:
        band = FLAT_BAND[d.unit]
        v = moves[d.id]
        out[d.id] = "+" if v > band else "-" if v < -band else "0"
    return out


def _lock_text(lock: dict[str, str]) -> str:
    words = []
    for d in DRIVERS:
        band = f"{FLAT_BAND[d.unit]:g}{_unit_word(d)}"
        words.append({"+": f"{d.id} up (above +{band})", "-": f"{d.id} down (below -{band})",
                      "0": f"{d.id} flat (within +/-{band})"}[lock[d.id]])
    return ("\n\nDirections already decided from the scenario alone. Keep each move in its range and use the "
            "evidence only to size it: " + ", ".join(words) + ".")


def check_directions(moves: dict[str, float], lock: dict[str, str]) -> list[str]:
    """Problems where an evidence-based move breaks the locked direction."""
    errors = []
    for d in DRIVERS:
        v, want, band = moves[d.id], lock[d.id], FLAT_BAND[d.unit]
        unit = _unit_word(d)
        if want == "+" and v <= band or want == "-" and v >= -band:
            word, bound = ("up", f"above +{band:g}{unit}") if want == "+" else ("down", f"below -{band:g}{unit}")
            errors.append(f"{d.id}: {v:+g}{unit} does not keep the clear {word} move that follows from the scenario; "
                          f"keep it {bound} and use the evidence only for its size.")
        elif want == "0" and abs(v) > band:
            errors.append(f"{d.id}: {v:+g}{unit} is a big move for a driver the scenario barely touches; "
                          f"keep it within +/-{band:g}{unit}.")
    return errors


def _ask(
    client: ChatClient, messages: Messages, limits: dict[str, Limits], result: TranslationResult,
    allowed: set[str] | None = None, lock: dict[str, str] | None = None,
) -> bool:
    """Ask, check, re-ask with the problems (at most MAX_RETRIES times). Fills `result`; True if valid."""
    for _ in range(1 + MAX_RETRIES):
        reply = client.chat_json(messages)
        result.attempts += 1
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        result.latency_s += reply.latency_s
        result.model = reply.model

        check = check_translation(reply.text, limits, allowed)
        errors = check.errors
        if check.ok and lock is not None:
            errors = check_directions(check.moves, lock)
        if not errors:
            result.status = "ok"
            result.moves = check.moves
            result.reasoning = check.reasoning
            result.events_used = check.events_used
            return True

        result.errors.append(errors)
        problems = "\n".join(f"- {e}" for e in errors)
        messages = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": f"Your answer has these problems:\n{problems}\nReply again with corrected JSON only."},
        ]
    return False


def translate(
    text: str, client: ChatClient, limits: dict[str, Limits], twins: Sequence[Twin] | None = None
) -> TranslationResult:
    """Without twins: one checked answer. With twins: two steps.

    1. Answer WITHOUT evidence: this fixes the direction of each driver (from the scenario's logic).
    2. Answer WITH evidence, sizing the moves; code rejects any move that flips a locked direction.
    Why: given the evidence in one step, the model let single past events (whose 20-day windows
    hold unrelated news) flip directions; on eval/scenarios.csv direction agreement fell from 80% to 67-71%.
    """
    if not text or not text.strip():
        raise ValueError("Scenario text is empty.")

    twins = list(twins or [])
    result = TranslationResult(status="failed", moves=None, reasoning="", events_used=[], attempts=0)
    if not twins:
        _ask(client, build_messages(text, limits), limits, result)
        return result

    _ask(client, build_messages(text, limits), limits, result)  # step 1: directions
    base = (result.moves, result.reasoning) if result.ok else None
    lock = direction_lock(result.moves) if base else None
    result.status, result.moves, result.reasoning, result.events_used = "failed", None, "", []

    messages = build_messages(text, limits, twins)
    if lock is not None:
        messages[0] = {"role": "system", "content": messages[0]["content"] + _lock_text(lock)}
    if _ask(client, messages, limits, result, {t.id for t in twins}, lock):  # step 2: sizes
        return result

    if base is not None:
        result.status, result.moves = "ok", base[0]
        result.reasoning = "Answer without past-event evidence (the evidence-based answers failed the checks). " + base[1]
        return result
    fb = fallback_moves(twins, limits)
    if fb is not None:
        strong = [t.id for t in twins if t.similarity >= STRONG_MATCH]
        result.status = "fallback"
        result.moves = fb
        result.events_used = strong
        result.reasoning = "The model's answers failed the checks, so these are the average real moves of the closest past events."
    return result
