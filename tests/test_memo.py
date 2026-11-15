"""Risk memo: the number checker, the facts sheet, retries and the template (fake model)."""

import json

import pandas as pd
import pytest

from riskdesk.checks.moves import Limits
from riskdesk.checks.numbers import check_numbers, extract_numbers
from riskdesk.engine.drivers import DRIVER_IDS
from riskdesk.forward import run_forward, run_manual
from riskdesk.llm.client import LLMReply, LLMUnavailableError
from riskdesk.llm.memo import BRIEFING_MAX_WORDS, SECTION_KEYS, build_facts, check_memo, write_memo
from riskdesk.retrieval.index import Twin
from riskdesk import ui

LIMITS = {d: Limits(-90.0, 500.0) for d in DRIVER_IDS} | {"us10y": Limits(-300.0, 300.0)}
PORTFOLIO = {"us_stocks": 10_000, "us_bonds": 10_000}
CHANGES = pd.DataFrame({d: [0.0] * 4 for d in DRIVER_IDS}, index=pd.date_range("2020-01-01", periods=4))
CHANGES["equity"] = [-20.0, -5.0, 0.0, 5.0]


# --- number checker --------------------------------------------------------
def test_extract_numbers_by_hand():
    nums, dates = extract_numbers("Lost $3,214 (-6.43%) on [E013] in 2008; S&P 500 and 10-year; 2.5k; 1.4x; 2008-09-15")
    assert [w.value for w in nums] == [3214, 6.43, 2500, 1.4]
    assert nums[2].step == 100  # "2.5k" is written to the nearest 100
    assert dates == ["2008-09-15", "2008"]


def test_rounding_rules():
    allowed = [-3213.6, -6.4321]
    assert check_numbers("loss of $3,214, or 6.43%, about 6.4%, about 6%", allowed, []).ok
    bad = check_numbers("loss of $3,200 or 6.5%", allowed, [])
    assert bad.wrong_numbers == ["$3,200", "6.5"]


def test_typographic_dashes_are_plain():
    nums, _ = extract_numbers("US 10" + chr(0x2011) + "year yield " + chr(0x2212) + "3.5%")
    assert [w.value for w in nums] == [3.5]


def test_zero_is_written_plainly():
    facts = build_facts(run_manual({"cash": 1000, "us_stocks": 1000}, {d: 0.0 for d in DRIVER_IDS} | {"equity": -1},
                                   CHANGES), "")
    cash = next(line for line in facts.lines if line.startswith("- Cash"))
    assert cash == "- Cash (USD): $1,000, 0.00%, $0"


def test_dates_must_be_given():
    nc = check_numbers("like 2008-09-15 and 2011", [], {"2008-09-15"})
    assert nc.wrong_dates == ["2011"] and check_numbers("in 2008", [], {"2008-09-15"}).ok


# --- facts and memo --------------------------------------------------------
class FakeClient:
    def __init__(self, *answers, error=None):
        self.answers, self.error, self.seen = list(answers), error, []

    def chat_json(self, messages):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return LLMReply(self.answers.pop(0), "fake", 1, 1, 0.0, False)


def stressed():
    # stocks 10,000 x -10% = -1,000; bonds -7 x 0.0050 x 10,000 = -350; total -1,350 (-6.75%)
    return run_manual(PORTFOLIO, {d: 0.0 for d in DRIVER_IDS} | {"equity": -10, "us10y": 50}, CHANGES)


def memo_json(**over):
    sections = {
        "briefing": "The portfolio would lose $1,350, or 6.75%, mostly from US stocks.",
        "what_happens": "Stocks fall 10% and the yield rises 50 bp over 20 trading days.",
        "biggest_hits": "US stocks lose $1,000; the bond fund loses $350.",
        "history": "The loss is 0.7x the 99% VaR of $2,000.",
        "past_events": "No past events were used.",
    }
    return json.dumps(sections | over)


def test_facts_hold_the_calculated_numbers():
    facts = build_facts(stressed(), "")
    text = "\n".join(facts.lines)
    assert "loss of $1,350 (-6.75% of the portfolio)" in text and "0.7x the 99% VaR" in text
    assert "- US stocks (S&P 500 fund): $10,000, -10.00%, -$1,000" in text
    assert -1350 in facts.numbers and 0.7 in facts.numbers


def test_template_passes_its_own_check():
    facts = build_facts(stressed(), "Oil to $150")
    sections, problems = check_memo(json.dumps(facts.template), facts)
    assert problems == [] and set(sections) == set(SECTION_KEYS)


def test_good_memo_is_kept():
    memo = write_memo(stressed(), "Rates up", FakeClient(memo_json()))
    assert memo.source == "ai" and memo.attempts == 1 and "first try" in memo.note


def test_wrong_number_is_sent_back_then_fixed():
    client = FakeClient(memo_json(biggest_hits="US stocks lose $1,100."), memo_json())
    memo = write_memo(stressed(), "Rates up", client)
    assert memo.source == "ai" and memo.attempts == 2
    assert "$1,100" in memo.problems[0][0]
    assert "$1,100" in client.seen[1][-1]["content"]  # the problem is told to the model


def test_still_wrong_uses_template():
    bad = memo_json(history="Worse than 2011, a 1-in-37 event.")
    memo = write_memo(stressed(), "Rates up", FakeClient(bad, bad))
    assert memo.source == "template" and memo.attempts == 2 and "template" in memo.note
    assert any("2011" in p for p in memo.problems[0]) and any("37" in p for p in memo.problems[0])


def test_other_problems():
    facts = build_facts(stressed(), "")
    long = " ".join(["word"] * (BRIEFING_MAX_WORDS + 1))
    _, problems = check_memo(memo_json(briefing=long, past_events="Like [E999]."), facts)
    assert "words" in problems[0] and "E999" in problems[1]
    assert check_memo("not json", facts)[0] is None
    assert "missing" in check_memo(json.dumps({"briefing": "x"}), facts)[1][0]
    as_list = json.dumps(json.loads(memo_json()) | {"biggest_hits": [{"holding": "US stocks", "amount": "-$1,000"}]})
    assert "not a list" in check_memo(as_list, facts)[1][0]


def test_plant_skips_a_list_section():
    as_list = json.dumps(json.loads(memo_json()) | {"biggest_hits": ["US stocks -$1,000"]})
    memo = write_memo(stressed(), "x", FakeClient(as_list, memo_json()), plant=True)
    assert memo.planted is None and memo.source == "ai" and memo.attempts == 2


def test_model_unavailable_uses_template():
    memo = write_memo(stressed(), "x", FakeClient(error=LLMUnavailableError("down")))
    assert memo.source == "template" and "not available" in memo.note


def test_planted_number_is_caught():
    memo = write_memo(stressed(), "Rates up", FakeClient(memo_json(), memo_json()), plant=True)
    assert memo.planted.startswith("$1,000 changed to")
    assert memo.source == "ai" and memo.attempts == 2 and "1 correction" in memo.note


def test_twin_numbers_and_ids_are_allowed():
    twin = Twin("E013", "Lehman owes $613 billion", "2008-09-15", "d", "credit", 0.8,
                {d: 0.0 for d in DRIVER_IDS} | {"equity": -24.3}, {d: 0.0 for d in DRIVER_IDS} | {"equity": -4.7})
    answer = json.dumps({"moves": {d: 0 for d in DRIVER_IDS} | {"equity": -10, "us10y": 50},
                         "reasoning": "r", "events_used": ["E013"]})

    class OneTwin:
        def search(self, text):
            return [twin]

    res = run_forward(PORTFOLIO, "Bank fails", FakeClient(answer, answer), LIMITS, CHANGES, OneTwin())
    memo = write_memo(res, "Bank fails", FakeClient(memo_json(past_events="Lehman owes $613 billion [E013] on 2008-09-15: stocks "
                                                                           "-4.7% on the first day, -24.3% over 20 days.")))
    assert memo.source == "ai", memo.problems


def test_memo_markdown():
    memo = write_memo(stressed(), "Rates up", FakeClient(memo_json()))
    text = ui.memo_markdown(memo)
    assert "**Briefing.**" in text and "not investment advice" in text and "number check" in text
