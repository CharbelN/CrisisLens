"""Run the scenario translator on eval/scenarios.csv: the "before RAG" baseline.

Usage:  uv run python eval/run_translator.py [--out NAME]

Expected directions in scenarios.csv: "+" up, "-" down, "0" roughly flat, "?" not scored.
"Roughly flat" means within 2% for prices and within 15 bp for the yield.
Answers are cached in cache/dev/, so a second run is instant and free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from riskdesk.checks.moves import default_limits
from riskdesk.engine.drivers import DRIVER_IDS, DRIVERS_BY_ID
from riskdesk.llm.client import GroqClient, LLMUnavailableError
from riskdesk.llm.translator import translate

EVAL_DIR = Path(__file__).resolve().parent
FLAT_PCT, FLAT_BP = 2.0, 15.0


def direction(driver: str, move: float) -> str:
    flat = FLAT_BP if DRIVERS_BY_ID[driver].unit == "bp" else FLAT_PCT
    return "+" if move > flat else "-" if move < -flat else "0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="translator_before_rag")
    args = parser.parse_args()

    scenarios = pd.read_csv(EVAL_DIR / "scenarios.csv", dtype=str, keep_default_na=False)
    limits = default_limits()
    client = GroqClient()

    rows = []
    for _, sc in scenarios.iterrows():
        row = {"id": sc["id"], "scenario": sc["scenario"]}
        try:
            res = translate(sc["scenario"], client, limits)
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
        row["reasoning"] = res.reasoning
        row["errors"] = " | ".join("; ".join(e) for e in res.errors)
        rows.append(row)
        print(f"{sc['id']}: {res.status} in {res.attempts} attempt(s), directions {hits}/{scored}")

    out = pd.DataFrame(rows)
    out_path = EVAL_DIR / "results" / f"{args.out}.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    done = out[out["status"].isin(["ok", "failed"])]
    ok = done[done["status"] == "ok"]
    print("\n=== Summary ===")
    print(f"Scenarios run:          {len(done)} / {len(out)}")
    if len(done):
        print(f"Valid on first try:     {(done['attempts'] == 1).mul(done['status'] == 'ok').sum()} / {len(done)}")
        print(f"Valid after retries:    {len(ok)} / {len(done)}")
        print(f"Failed:                 {(done['status'] == 'failed').sum()}")
    if len(ok):
        print(f"Direction agreement:    {ok['dir_hits'].sum()} / {ok['dir_scored'].sum()} "
              f"= {ok['dir_hits'].sum() / max(ok['dir_scored'].sum(), 1):.0%}")
        print(f"Avg latency (uncached): {done['latency_s'].mean():.2f} s,  avg tokens {done['tokens'].mean():.0f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
