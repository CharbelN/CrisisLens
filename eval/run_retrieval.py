"""Retrieval evaluation: does the twin search find the past events a person would pick?

Usage:  uv run python eval/run_retrieval.py

Headlines: eval/retrieval_queries.csv (the 20 translator scenarios plus 16 more written
for retrieval). For each headline with expected_events:
- hit@3: at least one expected event is in the top 3 results (the brief's measure),
- hit@4: the same in the top 4 (the app shows and uses 4),
- reciprocal rank: 1 / rank of the first expected event found (0 if none in the top 10).
MRR (mean reciprocal rank) is the average over scenarios.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskdesk.retrieval.index import TOP_K, Retriever

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    scenarios = pd.read_csv(EVAL_DIR / "retrieval_queries.csv", dtype=str, keep_default_na=False)
    retriever = Retriever.load()
    rows = []
    for _, sc in scenarios.iterrows():
        expected = [e for e in sc["expected_events"].split(";") if e]
        found = retriever.search(sc["headline"], k=10)
        ids = [t.id for t in found]
        rank = next((i + 1 for i, e in enumerate(ids) if e in expected), None)
        rows.append({
            "id": sc["id"], "headline": sc["headline"], "expected": ";".join(expected),
            "top4": ";".join(ids[:TOP_K]), "top1_similarity": found[0].similarity,
            "first_hit_rank": rank, "hit_at_3": None if not expected else bool(rank and rank <= 3),
            "hit_at_4": None if not expected else bool(rank and rank <= TOP_K),
            "reciprocal_rank": None if not expected else (1 / rank if rank else 0.0),
        })
        flag = "-" if not expected else ("hit" if rank and rank <= TOP_K else "MISS")
        print(f"{sc['id']}: {flag:4} rank {rank}  top4 {ids[:TOP_K]}")

    out = pd.DataFrame(rows)
    path = EVAL_DIR / "results" / "retrieval.csv"
    out.to_csv(path, index=False)
    scored = out[out["expected"] != ""]
    print("\n=== Retrieval summary ===")
    print(f"Scenarios scored: {len(scored)} (no expected events: {len(out) - len(scored)})")
    print(f"Hit@3:   {scored['hit_at_3'].sum()} / {len(scored)} = {scored['hit_at_3'].mean():.0%}")
    print(f"Hit@{TOP_K}:   {scored['hit_at_4'].sum()} / {len(scored)} = {scored['hit_at_4'].mean():.0%}")
    print(f"MRR@10:  {scored['reciprocal_rank'].mean():.2f}")
    print(f"Top-1 similarity: scored headlines avg {scored['top1_similarity'].mean():.2f}, "
          f"unscored (no real precedent) avg {out.loc[out['expected'] == '', 'top1_similarity'].mean():.2f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
