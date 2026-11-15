"""Backward mode: the reverse search (by hand), the story checks and the pipeline (fake model)."""

import json

import numpy as np
import pandas as pd
import pytest

from riskdesk.backward import run_backward
from riskdesk.checks.moves import Limits
from riskdesk.engine.drivers import DRIVER_IDS
from riskdesk.engine.reverse import (
    ReverseError, distances, exposures, most_likely_moves, nearest_event, reverse_stress,
)
from riskdesk.llm.client import LLMReply, LLMUnavailableError
from riskdesk.llm.stories import check_story, story_facts, write_stories
from riskdesk import ui

LIMITS = {d: Limits(-90.0, 300.0) for d in DRIVER_IDS} | {"us10y": Limits(-300.0, 300.0)}
ZERO = pd.Series(0.0, index=list(DRIVER_IDS))


def diag_cov(**var):
    v = {d: 1.0 for d in DRIVER_IDS} | var
    return pd.DataFrame(np.diag([v[d] for d in DRIVER_IDS]), index=list(DRIVER_IDS), columns=list(DRIVER_IDS))


# --- the search, by hand --------------------------------------------------------
def test_exposures_by_hand():
    # half stocks, half bonds: +1% stocks -> +0.5%; +1 bp yield -> -7 x 0.0001 x 0.5 = -0.035%
    a = exposures({"us_stocks": 10_000, "us_bonds": 10_000})
    assert a["equity"] == pytest.approx(0.5) and a["us10y"] == pytest.approx(-0.035) and a["gold"] == 0


def test_distance_by_hand():
    # (4 / sqrt 16)^2 + (2 / sqrt 4)^2 = 1 + 1 -> distance sqrt(2)
    row = pd.DataFrame([ZERO.to_dict() | {"equity": 4.0, "gold": 2.0}])
    assert distances(row, ZERO, diag_cov(equity=16.0, gold=4.0)).iloc[0] == pytest.approx(np.sqrt(2))


def test_most_likely_formula_by_hand():
    # a = 0.5 equity + 0.5 gold; variances 16 and 4, no correlation, average 0; target -10%
    # S a = (8, 2); a' S a = 0.5 x 8 + 0.5 x 2 = 5; m = S a x (-10 / 5) -> equity -16, gold -4
    a = ZERO.copy()
    a[["equity", "gold"]] = 0.5
    m, held = most_likely_moves(a, ZERO, diag_cov(equity=16.0, gold=4.0), -10.0, LIMITS)
    assert m["equity"] == pytest.approx(-16) and m["gold"] == pytest.approx(-4) and held == []
    # equity limited to -12: held there, gold must give the rest: 0.5 x g = -10 + 6 -> g = -8
    m, held = most_likely_moves(a, ZERO, diag_cov(equity=16.0, gold=4.0), -10.0, LIMITS | {"equity": Limits(-12.0, 50.0)})
    assert m["equity"] == pytest.approx(-12) and m["gold"] == pytest.approx(-8) and held == ["equity"]


def history():
    """400 made-up 20-day windows: random moves, plus two clear losing months for stocks."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=400)
    df = pd.DataFrame(rng.normal(0, [4, 10, 4, 2, 20, 20], size=(400, 6)), index=idx, columns=list(DRIVER_IDS))
    df.loc[idx[100], "equity"] = -10.0
    df.loc[idx[300], "equity"] = -8.0
    return df


def test_reverse_hits_the_target_and_finds_varied_months():
    res = reverse_stress({"us_stocks": 10_000, "gold": 10_000}, 10.0, history(), LIMITS)
    assert res.scenarios[0].kind == "most_likely"
    months = [s for s in res.scenarios if s.kind == "scaled_month"]
    assert len(months) == 2
    for s in res.scenarios:
        assert s.stress.pnl_pct == pytest.approx(-10.0, abs=0.1)  # moves are rounded to 0.1%
        assert 0 <= s.rarer_than <= 1
    for s in months:
        assert s.scale <= 3.0
    gap = abs((pd.Timestamp(months[0].window_end) - pd.Timestamp(months[1].window_end)).days)
    assert gap >= 180


def test_impossible_questions():
    with pytest.raises(ReverseError, match="only cash"):
        reverse_stress({"cash": 1000}, 10, history(), LIMITS)
    with pytest.raises(ReverseError, match="between"):
        reverse_stress({"us_stocks": 1000}, 120, history(), LIMITS)
    tight = LIMITS | {"equity": Limits(-5.0, 5.0)}
    res = reverse_stress({"us_stocks": 1000}, 50, history(), tight)  # stocks cannot fall 50% within -5%
    assert res.scenarios == [] and "cannot reach" in res.notes[0]


def test_nearest_event_by_hand():
    lib = pd.DataFrame([{"id": "E1", "name": "Crash", "date": "2020-03-01", **ZERO.to_dict(), "equity": -20.0},
                        {"id": "E2", "name": "Rally", "date": "2021-01-01", **ZERO.to_dict(), "equity": 10.0}])
    m = nearest_event(ZERO.to_dict() | {"equity": -15.0}, lib, diag_cov(equity=16.0))
    # gap 5 on equity (sd 4) -> 1.25; averaged over the 6 drivers compared: sqrt(1.25^2 / 6)
    assert m.id == "E1" and m.distance == pytest.approx(np.sqrt(1.25**2 / 6))
    # a portfolio holding bitcoin only matches events with bitcoin data
    lib.loc[0, "btc"] = np.nan
    assert nearest_event(ZERO.to_dict() | {"equity": -15.0}, lib, diag_cov(equity=16.0), ["btc"]).id == "E2"


# --- stories ------------------------------------------------------------------
def scenario():
    return reverse_stress({"us_stocks": 10_000, "gold": 10_000}, 10.0, history(), LIMITS).scenarios[0]


def story_for(sc, extra=""):
    e, g = abs(sc.moves["equity"]), abs(sc.moves["gold"])
    return {"headline": "Markets slide", "story": f"Stocks fall {e:g}% and gold drops {g:g}%.{extra}"}


def test_story_check():
    sc = scenario()
    f = story_facts("A", sc)
    assert set(f.required) == {"equity", "gold"}
    assert check_story(story_for(sc), f) == []
    assert "not moves" in check_story(story_for(sc, " Oil hits $147."), f)[0]
    only_stocks = {"headline": "x", "story": f"Stocks fall {abs(sc.moves['equity']):g}%."}
    assert "must state" in check_story(only_stocks, f)[0]
    assert "2011" in check_story(story_for(sc, " Worst since 2011."), f)[0]


class FakeClient:
    def __init__(self, *answers, error=None):
        self.answers, self.error = list(answers), error

    def chat_json(self, messages):
        if self.error:
            raise self.error
        return LLMReply(self.answers.pop(0), "fake", 1, 1, 0.0, False)


def reply(*stories):
    return json.dumps({"stories": [{"id": letter, **s} for letter, s in stories]})


def test_wrong_story_is_rewritten_then_dropped():
    sc = scenario()
    bad = story_for(sc, " Down 37 points.")
    res = write_stories([sc, sc], FakeClient(reply(("A", story_for(sc)), ("B", bad)), reply(("B", story_for(sc)))))
    assert set(res.stories) == {"A", "B"} and res.attempts == 2 and "37" in res.problems[0][0]
    res = write_stories([sc, sc], FakeClient(reply(("A", story_for(sc)), ("B", bad)), reply(("B", bad))))
    assert set(res.stories) == {"A"} and res.dropped == ["B"] and "dropped" in res.note


def test_stories_without_model():
    res = write_stories([scenario()], FakeClient(error=LLMUnavailableError("down")))
    assert res.stories == {} and "not available" in res.note


def test_backward_pipeline_and_screen():
    pf = {"us_stocks": 10_000, "gold": 10_000}
    sc_all = reverse_stress(pf, 10.0, history(), LIMITS).scenarios
    answer = reply(*[(letter, story_for(sc)) for letter, sc in zip("ABC", sc_all)])
    res = run_backward(pf, 10.0, history(), LIMITS, None, FakeClient(answer))
    assert res.status == "ok" and len(res.stories.stories) == 3
    table = ui.scenarios_table(res)
    assert list(table["Scenario"]) == ["A", "B", "C"] and table["Loss (%)"].iloc[0] == pytest.approx(10.0, abs=0.1)
    text = ui.backward_markdown(res)
    assert "Losing 10% = $2,000 of a $20,000 portfolio" in text and "**A. Markets slide**" in text
    assert run_backward({"cash": 5}, 10.0, history(), LIMITS, None, None).status == "error"
