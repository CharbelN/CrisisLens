"""Check events.csv, build the crisis library (real moves) and the vector index.

Usage:  uv run python data/build_library.py

Run again whenever events.csv or market_data.csv changes.
"""

from __future__ import annotations

from riskdesk.data.library import (
    EVENTS_PATH, LIBRARY_PATH, build_library, check_events, file_hash, load_events,
)
from riskdesk.data.market import load_market_data
from riskdesk.retrieval.index import INDEX_PATH, FastEmbedder, build_index


def main() -> None:
    levels = load_market_data()
    events = load_events()
    problems = check_events(events, levels)
    if problems:
        print("Problems in events.csv:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)

    unchecked = events[events["checked"] != "yes"]
    print(f"{len(events)} events, {len(unchecked)} not yet checked by hand.")

    library = build_library(events, levels)
    library.to_csv(LIBRARY_PATH, index=False)
    print(f"Saved {LIBRARY_PATH.name}")

    embedder = FastEmbedder()
    build_index(library, embedder, file_hash(EVENTS_PATH))
    print(f"Saved {INDEX_PATH.name} (model {embedder.name})")


if __name__ == "__main__":
    main()
