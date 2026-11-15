"""Scenario translator: a headline or what-if sentence -> a fixed set of market moves.

Flow:
1. Build a prompt that lists the 6 risk drivers, their units and the 20-day horizon.
   With RAG, the prompt also holds the retrieved past events ("historical twins")
   and what markets ACTUALLY did after each one, as evidence.
2. Ask the language model for JSON.
3. Check the answer (checks/moves.py). If it fails, send the problems back to the
   model and ask again, at most MAX_RETRIES times.
   With starting levels: the model is told today's level of each driver, and when the
   scenario states a level ("oil jumps to $150") code computes that move and checks the
   model's (checks/levels.py). If it is still off after the retries, code puts it in.
4. If it still fails:
   - with at least FALLBACK_MIN_TWINS strong twins (similarity >= STRONG_MATCH): for each
     driver where they all moved the same way, the similarity-weighted average of their
     real moves, clipped to the limits; 0 where they disagree (status "fallback");
   - otherwise: status "failed" with the reasons. Moves are never invented.

The moves are the model's educated guesses, not forecasts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from riskdesk.checks.levels import StartLevels, check_levels, numbers_in_text, start_levels_text
from riskdesk.checks.moves import HORIZON_DAYS, Limits, check_translation
from riskdesk.engine.drivers import DRIVERS
from riskdesk.llm.client import LLMReply, Messages
from riskdesk.retrieval.index import Twin, twin_average

MAX_RETRIES = 2
# bge-small cosine similarity; below this a twin is a weak match. Measured on eval/scenarios.csv
# (top 4 per headline): 0.65 keeps 24/30 expected matches and 19/42 others (0.60: 28/30 and 34/42).
# Right and wrong matches overlap, so "strong" means related, not proven comparable.
STRONG_MATCH = 0.65
FALLBACK_MIN_TWINS = 2  # the no-model fallback needs at least this many strong twins
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
    stated_levels: dict[str, float] = field(default_factory=dict)  # levels the scenario states (driver -> level)
    level_fixes: list[str] = field(default_factory=list)  # drivers whose move code set from a stated level
    steps: int = 1  # 2 with past events: directions first, then sizes (each step is one or more attempts)

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


def _levels_rule(start: StartLevels) -> str:
    return f"""
- Today's starting levels (latest close in our data, {start.date}): {start_levels_text(start)}.
  If the scenario states a level for a driver (for example "oil jumps to $150"), that driver's move is
  exactly the change from today's level to the stated level: (level / start - 1) x 100 for prices,
  (level - start) x 100 bp for the yield.
- "stated_levels" lists only levels the scenario itself states, e.g. {{"oil": 150}} (the yield in percent,
  e.g. {{"us10y": 6}}); it is {{}} if the scenario states none."""


LEVELS_KEY = ',\n "stated_levels": {}'


def build_messages(
    text: str, limits: dict[str, Limits], twins: Sequence[Twin] | None = None, start: StartLevels | None = None
) -> Messages:
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

    # Starting levels only when the text holds a number: a level cannot be stated without one. Measured on
    # eval/scenarios.csv, showing them for every headline made the model inflate moves on untouched drivers.
    show_levels = start is not None and bool(numbers_in_text(text))
    levels_rule = _levels_rule(start) if show_levels else ""
    levels_key = LEVELS_KEY if show_levels else ""
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
- Plain numbers only: no units, no ranges, no strings.{levels_rule}

Answer with JSON only, in exactly this shape:
{{"moves": {{"equity": 0, "oil": 0, "gold": 0, "usd": 0, "btc": 0, "us10y": 0}},
 "reasoning": "two or three sentences explaining the main moves",
 "events_used": []{levels_key}}}{evidence}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Scenario: {text.strip()}"},
    ]


FALLBACK_NOTE = ("These are the average real moves of the closest past events, kept only where at least "
                 f"{FALLBACK_MIN_TWINS} of them moved the same way; drivers where they disagreed are set to 0.")


def fallback_moves(twins: Sequence[Twin], limits: dict[str, Limits]) -> dict[str, float] | None:
    """Moves from the strong twins alone (no model), or None if they are too few or never agree.

    Why the agreement rule: similarity cannot tell a comparable event from one that only sounds
    alike (an OPEC output cut is the top match for an OPEC output increase), so a driver gets a
    move only when at least FALLBACK_MIN_TWINS strong twins with data all moved the same way.
    """
    strong = [t for t in twins if t.similarity >= STRONG_MATCH]
    if len(strong) < FALLBACK_MIN_TWINS:
        return None
    avg = twin_average(strong)
    out: dict[str, float] = {}
    for d in DRIVERS:
        vals = [t.moves[d.id] for t in strong if t.moves[d.id] is not None]
        agree = len(vals) >= FALLBACK_MIN_TWINS and (all(v > 0 for v in vals) or all(v < 0 for v in vals))
        out[d.id] = round(min(max(avg[d.id], limits[d.id].low), limits[d.id].high), 2) if agree else 0.0
    return out if any(out.values()) else None


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


def check_directions(moves: dict[str, float], lock: dict[str, str], skip: set[str] = frozenset()) -> list[str]:
    """Problems where an evidence-based move breaks the locked direction (drivers in `skip` are not checked)."""
    errors = []
    for d in DRIVERS:
        if d.id in skip:
            continue
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
    start: StartLevels | None = None, text: str = "",
) -> bool:
    """Ask, check, re-ask with the problems (at most MAX_RETRIES times). Fills `result`; True if valid.

    On the last attempt, if the only problems are moves that do not match a stated level,
    code puts in the computed moves (and ignores levels that are not in the text).
    """
    for attempt in range(1 + MAX_RETRIES):
        reply = client.chat_json(messages)
        result.attempts += 1
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        result.latency_s += reply.latency_s
        result.model = reply.model

        check = check_translation(reply.text, limits, allowed)
        errors = list(check.errors)
        level_errors: list[str] = []
        fixes: dict[str, float] = {}
        used: dict[str, float] = {}
        if check.ok:
            if start is not None:
                lv = check_levels(json.loads(reply.text).get("stated_levels"), check.moves, start, text)
                level_errors, fixes, used = lv.errors, lv.fixes, lv.used
            if lock is not None:
                errors += check_directions(check.moves, lock, skip=set(used))

        moves = check.moves
        if check.ok and not errors and level_errors and attempt == MAX_RETRIES:
            outside = [d for d, v in fixes.items() if not limits[d].low <= v <= limits[d].high]
            if not outside:  # last attempt: code puts in the moves implied by the stated levels
                moves, level_errors = check.moves | fixes, []
                result.level_fixes = sorted(fixes)
        errors += level_errors
        if not errors:
            result.status = "ok"
            result.moves = moves
            result.reasoning = check.reasoning
            result.events_used = check.events_used
            result.stated_levels = used
            return True

        result.errors.append(errors)
        problems = "\n".join(f"- {e}" for e in errors)
        messages = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": f"Your answer has these problems:\n{problems}\nReply again with corrected JSON only."},
        ]
    return False


def translate(
    text: str, client: ChatClient, limits: dict[str, Limits], twins: Sequence[Twin] | None = None,
    start: StartLevels | None = None,
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
    result = TranslationResult(status="failed", moves=None, reasoning="", events_used=[], attempts=0,
                               steps=2 if twins else 1)
    if not twins:
        _ask(client, build_messages(text, limits, start=start), limits, result, start=start, text=text)
        return result

    _ask(client, build_messages(text, limits, start=start), limits, result, start=start, text=text)  # step 1
    base = (result.moves, result.reasoning, result.stated_levels, result.level_fixes) if result.ok else None
    lock = direction_lock(result.moves) if base else None
    result.status, result.moves, result.reasoning, result.events_used = "failed", None, "", []
    result.stated_levels, result.level_fixes = {}, []

    messages = build_messages(text, limits, twins, start)
    if lock is not None:
        messages[0] = {"role": "system", "content": messages[0]["content"] + _lock_text(lock)}
    if _ask(client, messages, limits, result, {t.id for t in twins}, lock, start, text):  # step 2: sizes
        return result

    if base is not None:
        result.status, result.moves, result.stated_levels, result.level_fixes = "ok", base[0], base[2], base[3]
        result.reasoning = "Answer without past-event evidence (the evidence-based answers failed the checks). " + base[1]
        return result
    fb = fallback_moves(twins, limits)
    if fb is not None:
        strong = [t.id for t in twins if t.similarity >= STRONG_MATCH]
        result.status = "fallback"
        result.moves = fb
        result.events_used = strong
        result.reasoning = "The model's answers failed the checks. " + FALLBACK_NOTE
    return result
