"""Stated levels: numbers in the text, level -> move by hand, and the translator putting in the computed move."""

import json

import pandas as pd
import pytest

from riskdesk.checks.levels import StartLevels, check_levels, implied_move, latest_levels, numbers_in_text
from riskdesk.checks.moves import Limits
from riskdesk.engine.drivers import DRIVER_IDS
from riskdesk.llm.client import LLMReply
from riskdesk.llm.translator import build_messages, translate

START = StartLevels("2026-11-13", {"equity": 7500.0, "oil": 92.0, "gold": 4000.0, "usd": 100.0, "btc": 60000.0, "us10y": 5.10})
LIMITS = {d: Limits(-90.0, 500.0) for d in DRIVER_IDS} | {"us10y": Limits(-300.0, 300.0)}
ZERO = {d: 0.0 for d in DRIVER_IDS}


def test_numbers_in_text():
    assert numbers_in_text("Oil jumps to $150 as a Gulf route closes") == [150.0]
    assert numbers_in_text("Bitcoin crashes to $30,000; S&P 500 falls to 6,000") == [30000.0, 500.0, 6000.0]
    assert numbers_in_text("10-year yield hits 6.25%, bitcoin to 40k") == [10.0, 6.25, 40000.0]


def test_implied_move_by_hand():
    assert implied_move("oil", 92.0, 150.0) == pytest.approx(63.04)  # (150 / 92 - 1) x 100
    assert implied_move("btc", 60000.0, 30000.0) == pytest.approx(-50.0)
    assert implied_move("us10y", 5.10, 6.0) == pytest.approx(90.0)  # (6.00 - 5.10) x 100 bp


def test_latest_levels_fills_gaps_forward():
    idx = pd.to_datetime(["2026-01-01", "2026-01-02"])
    df = pd.DataFrame({d: [1.0, 2.0] for d in DRIVER_IDS}, index=idx)
    df.loc["2026-01-02", "oil"] = None  # e.g. a holiday in the oil series
    s = latest_levels(df)
    assert s.date == "2026-01-02" and s.levels["oil"] == 1.0 and s.levels["equity"] == 2.0


def test_check_levels():
    text = "Oil jumps to $150"
    good = check_levels({"oil": 150}, ZERO | {"oil": 62.0}, START, text)  # within 2 points of 63.04
    assert good.errors == [] and good.used == {"oil": 150.0}
    off = check_levels({"oil": 150}, ZERO | {"oil": 30.0}, START, text)
    assert off.fixes == {"oil": pytest.approx(63.04)} and "+63.04%" in off.errors[0]
    invented = check_levels({"gold": 5000}, ZERO, START, text)
    assert invented.used == {} and "not a number stated in the scenario" in invented.errors[0]
    assert check_levels({}, ZERO, START, text).errors == []


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)

    def chat_json(self, messages):
        return LLMReply(self.answers.pop(0), "fake", 1, 1, 0.0, False)


def answer(oil, levels):
    return json.dumps({"moves": ZERO | {"oil": oil, "equity": -5}, "reasoning": "r", "events_used": [], "stated_levels": levels})


def test_prompt_has_starting_levels_only_when_given():
    with_levels = build_messages("Oil jumps to $150", LIMITS, start=START)[0]["content"]
    assert "latest close in our data, 2026-11-13" in with_levels and "oil 92.00" in with_levels
    assert "stated_levels" not in build_messages("Oil jumps to $150", LIMITS)[0]["content"]
    # no number in the text -> no level can be stated -> the prompt stays as it was
    assert build_messages("Oil jumps sharply", LIMITS, start=START) == build_messages("Oil jumps sharply", LIMITS)


def test_model_corrects_its_move_after_feedback():
    res = translate("Oil jumps to $150", FakeClient([answer(30, {"oil": 150}), answer(63, {"oil": 150})]), LIMITS, start=START)
    assert res.ok and res.moves["oil"] == 63 and res.level_fixes == [] and res.stated_levels == {"oil": 150.0}


def test_code_puts_in_the_move_when_the_model_keeps_missing():
    res = translate("Oil jumps to $150", FakeClient([answer(30, {"oil": 150})] * 3), LIMITS, start=START)
    assert res.ok and res.attempts == 3
    assert res.moves["oil"] == pytest.approx(63.04) and res.moves["equity"] == -5
    assert res.level_fixes == ["oil"]


def test_invented_level_is_ignored_on_the_last_attempt():
    res = translate("Oil jumps sharply", FakeClient([answer(30, {"oil": 150})] * 3), LIMITS, start=START)
    assert res.ok and res.moves["oil"] == 30 and res.stated_levels == {} and res.level_fixes == []


def test_no_fix_when_the_stated_level_is_outside_the_limits():
    tight = LIMITS | {"oil": Limits(-50.0, 50.0)}
    res = translate("Oil jumps to $150", FakeClient([answer(30, {"oil": 150})] * 3), tight, start=START)
    assert res.status == "failed"
