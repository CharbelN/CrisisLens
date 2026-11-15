"""Held-out test: do past events make the moves closer to what markets ACTUALLY did?

Usage:  uv run python eval/run_heldout.py

For each event in eval/heldout_events.csv (a neutral headline: no outcome, no firm or
person names, no numbers):
- the event itself, and every event whose 20-day window overlaps it, are hidden from
  retrieval (their real moves would give the answer away);
- four approaches answer the headline:
    no_rag      the model alone (one checked answer)
    rag         the app's translator: retrieved past events, two steps
    twins_only  no model: the fallback from the retrieved events (0 where it gives nothing)
    zero        "no change" for every driver: the floor any approach must beat
- each answer is scored against the event's real 20-day moves (crisis_library.csv):
    direction hit: same up / down / flat as the real move (flat band 2% / 15 bp),
    size error:    |answer - real|, prices in points, the yield in bp / 10.
Bitcoin is scored only where it has data.

Calls are paced to stay under Groq's tokens-per-minute limit, and the smaller fallback model
is switched off, so every answer comes from the same model. Answers are cached in cache/dev/.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from riskdesk.checks.moves import default_limits
from riskdesk.data.library import load_library
from riskdesk.engine.drivers import DRIVER_IDS, DRIVERS_BY_ID
from riskdesk.llm.client import MAIN_MODEL, GroqClient, LLMReply, LLMUnavailableError, Messages
from riskdesk.llm.translator import FLAT_BAND, fallback_moves, translate
from riskdesk.retrieval.index import Retriever

EVAL_DIR = Path(__file__).resolve().parent
TOKENS_PER_MINUTE = 7000  # below Groq's 8,000 free-tier limit
APPROACHES = ["no_rag", "rag", "twins_only", "zero"]


class PacedClient:
    """Main model only; waits after each uncached answer so tokens per minute stay under the limit."""

    def __init__(self) -> None:
        self.inner = GroqClient(fallback_model=None)

    def chat_json(self, messages: Messages) -> LLMReply:
        for wait in (0, 60, 120):  # on a rate limit, wait and try again
            time.sleep(wait)
            try:
                reply = self.inner.chat_json(messages)
            except LLMUnavailableError as exc:
                if "tokens per day" in str(exc).lower():
                    raise SystemExit("Groq's daily token limit is used up. Answers so far are cached: "
                                     "run again later and it resumes where it stopped.") from exc
                if "rate" not in str(exc).lower() and "429" not in str(exc):
                    raise
                print(f"   rate-limited, waiting {wait + 60} s")
                continue
            if not reply.cached:
                time.sleep(60 * (reply.prompt_tokens + reply.completion_tokens) / TOKENS_PER_MINUTE)
            return reply
        raise LLMUnavailableError("still rate-limited after waiting")


def direction(driver: str, move: float) -> str:
    band = FLAT_BAND[DRIVERS_BY_ID[driver].unit]
    return "+" if move > band else "-" if move < -band else "0"


def hidden_ids(library: pd.DataFrame, event_id: str) -> set[str]:
    """The event plus every event whose 20-day window overlaps its window."""
    ev = library.set_index("id").loc[event_id]
    overlap = ~((library["window_end"] < ev["window_start"]) | (library["window_start"] > ev["window_end"]))
    return set(library.loc[overlap, "id"])


def main() -> None:
    library = load_library()
    tests = pd.read_csv(EVAL_DIR / "heldout_events.csv", dtype=str)
    limits, client, retriever = default_limits(), PacedClient(), Retriever.load()
    lib = library.set_index("id")

    rows = []
    for _, t in tests.iterrows():
        hide = hidden_ids(library, t["id"])
        twins = retriever.search(t["headline"], exclude=hide)
        real = {d: lib.at[t["id"], d] for d in DRIVER_IDS}
        answers, meta = {}, {}
        for name, twins_arg in (("no_rag", None), ("rag", twins)):
            res = translate(t["headline"], client, limits, twins_arg)
            answers[name] = res.moves
            meta[name] = (res.status, res.model)
        answers["twins_only"] = fallback_moves(twins, limits) or {d: 0.0 for d in DRIVER_IDS}
        answers["zero"] = {d: 0.0 for d in DRIVER_IDS}

        for name in APPROACHES:
            moves = answers[name]
            for d in DRIVER_IDS:
                if pd.isna(real[d]) or moves is None:
                    continue
                scale = 10.0 if DRIVERS_BY_ID[d].unit == "bp" else 1.0
                rows.append({
                    "id": t["id"], "name": lib.at[t["id"], "name"], "risk_type": lib.at[t["id"], "risk_type"],
                    "approach": name, "driver": d, "answer": moves[d], "real": real[d],
                    "hit": direction(d, moves[d]) == direction(d, real[d]),
                    "error": abs(moves[d] - real[d]) / scale,
                    "status": meta.get(name, ("", ""))[0], "model": meta.get(name, ("", ""))[1],
                    "twins": ";".join(f"{w.id}:{w.similarity:.2f}" for w in twins), "hidden": ";".join(sorted(hide)),
                })
        print(f"{t['id']} {lib.at[t['id'], 'name']}: rag {meta['rag'][0]}, no_rag {meta['no_rag'][0]}, "
              f"twins {[w.id for w in twins]}")

    out = pd.DataFrame(rows)
    out.to_csv(EVAL_DIR / "results" / "heldout.csv", index=False)
    other_models = out[(out["model"] != "") & (out["model"] != MAIN_MODEL)]
    if len(other_models):
        print(f"WARNING: {other_models['id'].nunique()} event(s) answered by another model")

    summary = out.groupby("approach").agg(direction_hits=("hit", "mean"), size_error=("error", "mean"), n=("hit", "size"))
    by_type = out.pivot_table(index="risk_type", columns="approach", values="error", aggfunc="mean")
    by_driver = out.pivot_table(index="driver", columns="approach", values="hit", aggfunc="mean")
    summary = summary.loc[APPROACHES]
    print("\n=== Held-out events: answers vs what markets actually did ===")
    print(summary.assign(direction_hits=summary["direction_hits"].map("{:.0%}".format),
                         size_error=summary["size_error"].map("{:.2f}".format)).to_string())
    print("\nSize error by risk type (lower = closer):\n" + by_type[APPROACHES].round(2).to_string())
    print("\nDirection hits by driver:\n" + by_driver[APPROACHES].map("{:.0%}".format).to_string())
    summary.to_csv(EVAL_DIR / "results" / "heldout_summary.csv")
    print(f"\nSaved: {EVAL_DIR / 'results' / 'heldout.csv'}")


if __name__ == "__main__":
    main()
