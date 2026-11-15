"""Vector index, stale-index detection, twin averages and RAG in the translator (fake embedder and model)."""

import json

import numpy as np
import pandas as pd
import pytest

from riskdesk.checks.moves import Limits
from riskdesk.engine.drivers import DRIVER_IDS
from riskdesk.llm.client import LLMReply
from riskdesk.llm.translator import STRONG_MATCH, build_messages, fallback_moves, check_directions, direction_lock, translate, typical_moves
from riskdesk.retrieval.index import Retriever, StaleIndexError, Twin, build_index, twin_average

WORDS = ["oil", "bank", "fed", "crypto"]


class WordEmbedder:
    """Counts keywords: a tiny, predictable stand-in for the real model."""

    name = "fake-words"

    def _vec(self, text):
        t = text.lower()
        return np.array([float(t.count(w)) for w in WORDS]) + 0.01

    def passages(self, texts):
        return np.array([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


def library():
    rows = [
        ("E1", "Oil shock", "oil supply cut, oil soars", 30.0),
        ("E2", "Bank run", "bank fails after bank run", -5.0),
        ("E3", "Crypto crash", "crypto exchange crypto collapse", -2.0),
    ]
    df = pd.DataFrame(
        [{"id": i, "name": n, "date": "2020-01-01", "description": d, "risk_type": "market",
          **{k: 0.0 for k in DRIVER_IDS}, "oil": oil} for i, n, d, oil in rows]
    )
    df.loc[2, "btc"] = np.nan
    return df


@pytest.fixture
def built(tmp_path):
    events = tmp_path / "events.csv"
    events.write_text("dummy", encoding="utf-8")
    index = tmp_path / "index.npz"
    build_index(library(), WordEmbedder(), "will-not-match", index)
    return events, index


def test_search_ranks_by_meaning(tmp_path):
    events = tmp_path / "events.csv"
    events.write_text("x", encoding="utf-8")
    from riskdesk.data.library import file_hash

    index = tmp_path / "index.npz"
    build_index(library(), WordEmbedder(), file_hash(events), index)
    r = Retriever.load(WordEmbedder(), index, events, library())
    twins = r.search("oil prices jump", k=2)
    assert twins[0].id == "E1" and twins[0].moves["oil"] == 30.0
    assert r.search("crypto exchange halts", k=1)[0].moves["btc"] is None
    # hidden events never come back, even when they are the best match
    assert [t.id for t in r.search("oil prices jump", k=3, exclude={"E1"})] in (["E2", "E3"], ["E3", "E2"])


def test_stale_index_is_refused(built):
    events, index = built
    with pytest.raises(StaleIndexError, match="events.csv changed"):
        Retriever.load(WordEmbedder(), index, events, library())


def test_other_model_is_refused(built):
    events, index = built
    other = WordEmbedder()
    other.name = "another-model"
    with pytest.raises(StaleIndexError, match="built with"):
        Retriever.load(other, index, events, library())


def twin(id_, sim, **moves):
    return Twin(id_, id_, "2020-01-01", "d", "market", sim, {d: 0.0 for d in DRIVER_IDS} | moves)


def test_twin_average_by_hand():
    # oil: (0.8 x 30 + 0.2 x -10) / 1.0 = 22
    # btc: only the first twin has data -> 50
    avg = twin_average([twin("A", 0.8, oil=30, btc=50), twin("B", 0.2, oil=-10, btc=None)])
    assert avg["oil"] == pytest.approx(22.0)
    assert avg["btc"] == pytest.approx(50.0)


LIMITS = {d: Limits(-90.0, 500.0) for d in DRIVER_IDS} | {"us10y": Limits(-300.0, 300.0)}


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)

    def chat_json(self, messages):
        return LLMReply(self.answers.pop(0), "fake", 1, 1, 0.0, False)


def answer(events, **moves):
    moves = {d: 0 for d in DRIVER_IDS} | {"oil": 25} | moves
    return json.dumps({"moves": moves, "reasoning": "r", "events_used": events})


def test_prompt_contains_evidence():
    msg = build_messages("Oil up", LIMITS, [twin("E011", 0.7, oil=12.5)])[0]["content"]
    assert "[E011]" in msg and "oil +12.5%" in msg and "strong match" in msg


def test_prompt_shows_first_day_and_typical_moves():
    t = Twin("E011", "Oil peak", "2008-07-11", "d", "market", 0.7, {d: 0.0 for d in DRIVER_IDS} | {"oil": -14.0},
             {d: 0.0 for d in DRIVER_IDS} | {"oil": 5.8})
    msg = build_messages("Oil up", LIMITS, [t])[0]["content"]
    assert "First-day reaction: equity +0%, oil +5.8%" in msg and "oil -14%" in msg


def test_typical_moves_median_by_hand():
    # strong twins only: oil 10, 30, -5 -> median 10, 2 up 1 down; the weak twin (-90) is ignored
    twins = [twin("A", 0.9, oil=10), twin("B", 0.8, oil=30), twin("C", 0.7, oil=-5), twin("D", 0.3, oil=-90)]
    line = typical_moves(twins)
    assert "3 strong match(es)" in line and "oil +10.0% (2 up, 1 down)" in line
    assert "No strong match" in typical_moves([twin("D", 0.3, oil=-90)])


def test_direction_lock_by_hand():
    # band: 2% for prices, 15 bp for the yield
    lock = direction_lock({d: 0.0 for d in DRIVER_IDS} | {"oil": 25, "equity": -1.9, "us10y": -16})
    assert lock["oil"] == "+" and lock["equity"] == "0" and lock["us10y"] == "-"
    ok = {d: 0.0 for d in DRIVER_IDS} | {"oil": 3, "equity": 1.5, "us10y": -40}
    assert check_directions(ok, lock) == []
    bad = ok | {"oil": 1.0, "equity": 5.0, "us10y": 10}
    errs = check_directions(bad, lock)
    assert len(errs) == 3 and errs[0].startswith("equity") and "clear up move" in errs[1]


def test_cited_event_must_be_one_that_was_retrieved():
    twins = [twin("E011", 0.7, oil=10)]
    # step 1 (no evidence) + step 2 twice: first cites an unknown id, second is fine
    res = translate("Oil up", FakeClient([answer([]), answer(["E999"]), answer(["E011"])]), LIMITS, twins)
    assert res.ok and res.attempts == 3 and res.events_used == ["E011"]
    assert "not given as evidence" in res.errors[0][0]


def test_evidence_cannot_flip_a_direction():
    twins = [twin("E011", 0.7, oil=-14)]
    # step 1 says oil +25; the evidence-based answer says -10 (rejected), then +12 (kept)
    client = FakeClient([answer([]), answer(["E011"], oil=-10), answer(["E011"], oil=12)])
    res = translate("Oil up", client, LIMITS, twins)
    assert res.ok and res.moves["oil"] == 12 and "oil" in res.errors[0][0]
    # step 2 keeps flipping -> the answer without evidence is used
    client = FakeClient([answer([])] + [answer(["E011"], oil=-10)] * 3)
    res = translate("Oil up", client, LIMITS, twins)
    assert res.ok and res.moves["oil"] == 25 and res.reasoning.startswith("Answer without past-event evidence")


def test_fallback_uses_strong_twins_only():
    # A and B are strong and agree on oil: (0.8 x 30 + 0.7 x 10) / 1.5 = 20.67; C is weak and ignored
    twins = [twin("A", 0.8, oil=30), twin("B", 0.7, oil=10), twin("C", STRONG_MATCH - 0.1, oil=-50)]
    res = translate("Oil up", FakeClient(["bad"] * 6), LIMITS, twins)
    assert res.status == "fallback" and res.moves["oil"] == pytest.approx(20.67)
    assert res.events_used == ["A", "B"]


def test_fallback_sets_disagreeing_drivers_to_zero():
    # oil: +30 and -10 disagree -> 0; equity: -5 and -3 agree -> (0.8 x -5 + 0.7 x -3) / 1.5 = -4.07
    twins = [twin("A", 0.8, oil=30, equity=-5), twin("B", 0.7, oil=-10, equity=-3)]
    moves = fallback_moves(twins, LIMITS)
    assert moves["oil"] == 0.0 and moves["equity"] == pytest.approx(-4.07)


def test_no_fallback_with_one_strong_twin_or_no_agreement():
    assert fallback_moves([twin("A", 0.9, oil=30), twin("B", 0.4, oil=30)], LIMITS) is None
    assert fallback_moves([twin("A", 0.9, oil=30), twin("B", 0.8, oil=-30)], LIMITS) is None


def test_no_fallback_without_strong_twins():
    res = translate("Aliens", FakeClient(["bad"] * 6), LIMITS, [twin("A", 0.4, oil=30)])
    assert res.status == "failed" and res.moves is None


def test_fallback_is_clipped_to_limits():
    tight = LIMITS | {"oil": Limits(-20.0, 20.0)}
    res = translate("Oil up", FakeClient(["bad"] * 6), tight, [twin("A", 0.9, oil=60), twin("B", 0.8, oil=40)])
    assert res.moves["oil"] == 20.0
