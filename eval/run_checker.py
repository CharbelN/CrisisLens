"""Number-checker evaluation: does the checker catch wrong numbers, and does it raise false alarms?

Usage:  uv run python eval/run_checker.py

1. For each of the 20 scenarios in eval/scenarios.csv, run forward mode on the sample portfolio
   and ask the model for ONE memo first draft (cached in cache/dev/, so a second run is free).
2. False alarms: run the checker on every untouched first draft and list what it flags,
   so each flag can be judged by hand (a real model mistake, or a correct number wrongly flagged).
3. Planted errors: in every first draft that passed, replace each number, one at a time, by a wrong
   one, in four ways:
       big        x 1.37 + 113 (the app's demo switch)
       plus10     10% too high
       last_digit the last written digit one higher (a typo: $750 -> $751, 6.4% -> 6.5%)
       swap       the first two digits swapped ($750 -> $570); skipped when they are equal
   and check whether the checker flags the memo. A wrong number that happens to equal another
   calculated figure cannot be caught by design; those cases are counted separately.
Signs are ignored by the checker by design ("a loss of $750" and "-$750" are the same number),
so a flipped sign is not planted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from riskdesk.checks.levels import latest_levels
from riskdesk.checks.moves import HORIZON_DAYS, plausible_limits
from riskdesk.checks.numbers import extract_numbers, matches
from riskdesk.data.market import driver_changes, load_market_data
from riskdesk.forward import run_forward
from riskdesk.llm.client import LLMUnavailableError, default_client
from riskdesk.llm.memo import SECTION_KEYS, build_facts, build_messages, check_memo
from riskdesk.retrieval.index import Retriever
from riskdesk.ui import parse_portfolio, portfolio_table

EVAL_DIR = Path(__file__).resolve().parent


def fmt(value: float, like: str) -> str:
    """Write `value` the way `like` is written ($ sign, thousands commas, number of decimals)."""
    decimals = len(like.split(".")[1]) if "." in like else 0
    body = f"{value:,.{decimals}f}" if "," in like else f"{value:.{decimals}f}"
    return ("$" if like.startswith("$") else "") + body


def wrong_versions(text: str, value: float, step: float) -> dict[str, str]:
    digits = re.sub(r"\D", "", text)
    out = {
        "big": fmt(value * 1.37 + 113, text),
        "plus10": fmt(value * 1.10 if value else 1.0, text),
        "last_digit": fmt(value + step, text),
    }
    if len(digits) >= 2 and digits[0] != digits[1]:
        swapped = digits[1] + digits[0] + digits[2:]
        n_dec = len(text.split(".")[1]) if "." in text else 0
        out["swap"] = fmt(int(swapped) / 10**n_dec, text)
    return {k: v for k, v in out.items() if v != fmt(value, text)}


def replace_once(section: str, old: str, new: str) -> str:
    pattern = r"(?<![\w.,$])" + re.escape(old) + r"(?![\d])"
    return re.sub(pattern, new.replace("\\", "\\\\"), section, count=1)


def main() -> None:
    levels = load_market_data()
    changes = driver_changes(levels, HORIZON_DAYS)
    limits, start = plausible_limits(levels), latest_levels(levels)
    retriever, client = Retriever.load(), default_client()
    portfolio = parse_portfolio(portfolio_table())
    scenarios = pd.read_csv(EVAL_DIR / "scenarios.csv", dtype=str, keep_default_na=False)

    drafts, alarms, planted = [], [], []
    for _, sc in scenarios.iterrows():
        res = run_forward(portfolio, sc["scenario"], client, limits, changes, retriever, start)
        if res.stress is None:
            print(f"{sc['id']}: no result ({res.status}), skipped")
            continue
        facts = build_facts(res, sc["scenario"])
        try:
            reply = client.chat_json(build_messages(facts))
        except LLMUnavailableError as exc:
            print(f"{sc['id']}: model unavailable, skipped ({exc})")
            continue
        sections, problems = check_memo(reply.text, facts)
        number_problems = [p for p in problems if "not in the facts" in p]
        drafts.append({"id": sc["id"], "model": reply.model, "usable": sections is not None,
                       "flagged": bool(number_problems), "problems": " | ".join(number_problems)})
        if number_problems:
            alarms.append((sc["id"], number_problems))
        if sections is None or problems:
            continue
        data = json.loads(reply.text)
        for key in SECTION_KEYS:
            numbers, _ = extract_numbers(sections[key])
            for w in numbers:
                if w.step != 1.0 and w.step >= 1.0:  # "2.5k" style: skip
                    continue
                for kind, new in wrong_versions(w.text, w.value, w.step).items():
                    changed = replace_once(sections[key], w.text, new)
                    if changed == sections[key]:
                        continue
                    _, probs = check_memo(json.dumps(data | {key: changed}), facts)
                    new_written = extract_numbers(new)[0]
                    coincides = bool(new_written) and any(matches(new_written[0], a) for a in facts.numbers)
                    planted.append({"id": sc["id"], "section": key, "kind": kind, "original": w.text, "wrong": new,
                                    "caught": bool(probs), "equals_other_figure": coincides})
        print(f"{sc['id']}: draft {'flagged' if number_problems else 'passed'} ({reply.model})")

    d, p = pd.DataFrame(drafts), pd.DataFrame(planted)
    d.to_csv(EVAL_DIR / "results" / "checker_drafts.csv", index=False)
    p.to_csv(EVAL_DIR / "results" / "checker_planted.csv", index=False)

    print(f"\n=== Untouched first drafts: {len(d)} ===")
    print(f"Flagged by the checker: {int(d['flagged'].sum())}")
    for sid, probs in alarms:
        print(f"  {sid}: {' | '.join(probs)}")
    print(f"\n=== Planted wrong numbers: {len(p)} in {p['id'].nunique()} memos ===")
    catchable = p[~p["equals_other_figure"]]
    print(f"Caught: {int(p['caught'].sum())} / {len(p)}  "
          f"(excluding {len(p) - len(catchable)} that equal another calculated figure: "
          f"{int(catchable['caught'].sum())} / {len(catchable)})")
    print(p.groupby("kind")["caught"].agg(["sum", "size", "mean"]).round(3).to_string())
    missed = catchable[~catchable["caught"]]
    if len(missed):
        print("\nMissed (not equal to any calculated figure):")
        print(missed.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
