"""Stated-level evaluation: when a headline names a level ("oil jumps to $150"), is the move right?

Usage:  uv run python eval/run_levels.py               (with today's starting levels: the app)
        uv run python eval/run_levels.py --no-levels   (before: the model is not told today's levels)

Runs the translator the way the app does (with retrieved past events) on eval/level_scenarios.csv.
For each stated level, the exact move is computed from the latest close in market_data.csv.
A move counts as right if it is within LEVEL_TOLERANCE of it (2 points for prices, 10 bp for the yield).
Answers are cached in cache/dev/, so a second run is instant and free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from riskdesk.checks.levels import LEVEL_TOLERANCE, implied_move, latest_levels
from riskdesk.checks.moves import default_limits
from riskdesk.data.market import load_market_data
from riskdesk.engine.drivers import DRIVERS_BY_ID
from riskdesk.llm.client import GroqClient, LLMUnavailableError
from riskdesk.llm.translator import translate
from riskdesk.retrieval.index import Retriever

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-levels", action="store_true", help="do not tell the model today's starting levels")
    args = parser.parse_args()
    out_name = "levels_before" if args.no_levels else "levels_after"

    start = latest_levels(load_market_data())
    scenarios = pd.read_csv(EVAL_DIR / "level_scenarios.csv", dtype=str, keep_default_na=False)
    limits, client, retriever = default_limits(), GroqClient(), Retriever.load()

    rows = []
    for _, sc in scenarios.iterrows():
        try:
            res = translate(sc["scenario"], client, limits, retriever.search(sc["scenario"]),
                            None if args.no_levels else start)
        except LLMUnavailableError as exc:
            print(f"{sc['id']}: LLM unavailable: {exc}")
            continue
        for pair in sc["levels"].split(";"):
            driver, level = pair.split("=")
            target = implied_move(driver, start.levels[driver], float(level))
            move = res.moves[driver] if res.moves else None
            right = move is not None and abs(move - target) <= LEVEL_TOLERANCE[DRIVERS_BY_ID[driver].unit]
            rows.append({"id": sc["id"], "scenario": sc["scenario"], "driver": driver, "level": float(level),
                         "start": start.levels[driver], "exact_move": target, "model_move": move, "right": right,
                         "set_by_code": driver in (res.level_fixes or []), "status": res.status,
                         "attempts": res.attempts})
            print(f"{sc['id']} {driver}: exact {target:+.1f}, got {move}, {'right' if right else 'WRONG'}"
                  f"{' (set by code)' if driver in res.level_fixes else ''}")

    out = pd.DataFrame(rows)
    path = EVAL_DIR / "results" / f"{out_name}.csv"
    out.to_csv(path, index=False)
    print(f"\n=== Stated levels ({'model not told today' + chr(39) + 's levels' if args.no_levels else 'with starting levels'}) ===")
    print(f"Right moves:    {out['right'].sum()} / {len(out)}")
    print(f"Set by code:    {out['set_by_code'].sum()}")
    print(f"Mean miss:      {(out['model_move'] - out['exact_move']).abs().mean():.1f} (points or bp)")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
