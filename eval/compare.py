"""Before vs after RAG: one table for the report.

Usage:  uv run python eval/compare.py     (after both translator runs)

"Distance to history" = mean absolute gap between the model's moves and the
similarity-weighted average real moves of the strong twins (prices in %, yield
in bp divided by 10 so both are on a similar scale). Lower = closer to what
markets actually did in comparable past events. Scenarios without a strong twin are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskdesk.engine.drivers import DRIVER_IDS, DRIVERS_BY_ID
from riskdesk.llm.translator import STRONG_MATCH
from riskdesk.retrieval.index import Retriever, twin_average

RESULTS = Path(__file__).resolve().parent / "results"


def distance(row: pd.Series, ref: dict[str, float]) -> float:
    gaps = []
    for d in DRIVER_IDS:
        if pd.isna(row.get(d)):
            return float("nan")
        scale = 10.0 if DRIVERS_BY_ID[d].unit == "bp" else 1.0
        gaps.append(abs(row[d] - ref[d]) / scale)
    return sum(gaps) / len(gaps)


def summarise(df: pd.DataFrame, refs: dict[str, dict[str, float]]) -> dict[str, str]:
    done = df[df["status"].isin(["ok", "fallback", "failed"])]
    ok = done[done["status"] != "failed"]
    dist = [distance(r, refs[r["id"]]) for _, r in ok.iterrows() if r["id"] in refs]
    dist = [x for x in dist if x == x]
    used = df["events_used"].fillna("").astype(str).str.len().gt(0).sum() if "events_used" in df else 0
    return {
        # no answer needed correcting (with RAG there are two steps, so attempts start at 2)
        "Valid on first try": f"{((done['status'] == 'ok') & (done['errors'].isna() | done['errors'].astype(str).eq(''))).sum()} / {len(done)}",
        "Fallback used": str((done["status"] == "fallback").sum()),
        "Failed": str((done["status"] == "failed").sum()),
        "Direction agreement": f"{ok['dir_hits'].sum() / max(ok['dir_scored'].sum(), 1):.0%} ({ok['dir_hits'].sum()}/{ok['dir_scored'].sum()})",
        "Distance to history (lower = closer)": f"{sum(dist) / len(dist):.2f} over {len(dist)} scenarios" if dist else "n/a",
        "Scenarios citing past events": str(used),
        "Avg tokens per scenario": f"{done['tokens'].mean():.0f}",
    }


def main() -> None:
    before = pd.read_csv(RESULTS / "translator_before_rag.csv")
    after = pd.read_csv(RESULTS / "translator_with_rag.csv")
    retriever = Retriever.load()
    refs = {}
    for _, r in before.iterrows():
        strong = [t for t in retriever.search(r["scenario"]) if t.similarity >= STRONG_MATCH]
        if strong:
            refs[r["id"]] = twin_average(strong)
    table = pd.DataFrame({"Before RAG": summarise(before, refs), "With RAG": summarise(after, refs)})
    print(table.to_string())
    table.to_csv(RESULTS / "rag_comparison.csv")
    print(f"\nSaved: {RESULTS / 'rag_comparison.csv'}")


if __name__ == "__main__":
    main()
