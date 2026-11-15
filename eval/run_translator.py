"""Run the scenario translator on eval/scenarios.csv, before or with RAG.

Usage:  uv run python eval/run_translator.py          (before RAG: no evidence)
        uv run python eval/run_translator.py --rag    (with historical twins as evidence)

With --rag the translator works in two steps (see llm/translator.py): an answer without
evidence fixes the directions, then the evidence sizes the moves. So "attempts" is at least 2.

Expected directions in scenarios.csv: "+" up, "-" down, "0" roughly flat, "?" not scored.
"Roughly flat" means within 2% for prices and within 15 bp for the yield.
Answers are cached in cache/dev/, so a second run is instant and free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from riskdesk.checks.levels import latest_levels
from riskdesk.checks.moves import default_limits
from riskdesk.data.market import load_market_data
from riskdesk.engine.drivers import DRIVER_IDS, DRIVERS_BY_ID
from riskdesk.llm.client import GROQ_MAIN, GROQ_SMALL, LLMUnavailableError, pinned_client
from riskdesk.llm.translator import translate

EVAL_DIR = Path(__file__).resolve().parent
FLAT_PCT, FLAT_BP = 2.0, 15.0


def direction(driver: str, move: float) -> str:
    flat = FLAT_BP if DRIVERS_BY_ID[driver].unit == "bp" else FLAT_PCT
    return "+" if move > flat else "-" if move < -flat else "0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", action="store_true", help="give the model the retrieved past events")
    parser.add_argument("--no-levels", action="store_true", help="do not tell the model today's starting levels")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    start = None if args.no_levels else latest_levels(load_market_data())
    out_name = args.out or ("translator_with_rag" if args.rag else "translator_before_rag")
    retriever = None
    if args.rag:
        from riskdesk.retrieval.index import Retriever

        retriever = Retriever.load()

    scenarios = pd.read_csv(EVAL_DIR / "scenarios.csv", dtype=str, keep_default_na=False)
    limits = default_limits()
    client = pinned_client(GROQ_MAIN, GROQ_SMALL)  # Groq only: the saved results use both

    rows = []
    for _, sc in scenarios.iterrows():
        row = {"id": sc["id"], "scenario": sc["scenario"]}
        try:
            twins = retriever.search(sc["scenario"]) if retriever else None
            res = translate(sc["scenario"], client, limits, twins, start)
        except LLMUnavailableError as exc:
            print(f"{sc['id']}: LLM unavailable: {exc}")
            row.update(status="unavailable")
            rows.append(row)
            continue
        row.update(status=res.status, attempts=res.attempts, model=res.model,
                   latency_s=round(res.latency_s, 2), tokens=res.prompt_tokens + res.completion_tokens)
        hits = scored = 0
        for d in DRIVER_IDS:
            expected = sc[d].strip()
            if res.moves is None:
                row[d] = None
                continue
            row[d] = res.moves[d]
            if expected in {"+", "-", "0"}:
                scored += 1
                hits += direction(d, res.moves[d]) == expected
        row["dir_hits"], row["dir_scored"] = hits, scored
        row["events_used"] = ";".join(res.events_used)
        row["reasoning"] = res.reasoning
        row["errors"] = " | ".join("; ".join(e) for e in res.errors)
        rows.append(row)
        print(f"{sc['id']}: {res.status} in {res.attempts} attempt(s), directions {hits}/{scored}")

    out = pd.DataFrame(rows)
    out_path = EVAL_DIR / "results" / f"{out_name}.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    done = out[out["status"].isin(["ok", "failed"])]
    ok = done[done["status"] == "ok"]
    print("\n=== Summary ===")
    print(f"Scenarios run:          {len(done)} / {len(out)}")
    if len(done):
        first = (done["status"] == "ok") & (done["errors"].isna() | done["errors"].astype(str).eq(""))  # no answer needed correcting
        print(f"Valid on first try:     {first.sum()} / {len(done)}")
        print(f"Valid after retries:    {len(ok)} / {len(done)}")
        print(f"Fallback used:          {(done['status'] == 'fallback').sum()}")
        print(f"Failed:                 {(done['status'] == 'failed').sum()}")
    if len(ok):
        print(f"Direction agreement:    {ok['dir_hits'].sum()} / {ok['dir_scored'].sum()} "
              f"= {ok['dir_hits'].sum() / max(ok['dir_scored'].sum(), 1):.0%}")
        print(f"Avg latency (uncached): {done['latency_s'].mean():.2f} s,  avg tokens {done['tokens'].mean():.0f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
