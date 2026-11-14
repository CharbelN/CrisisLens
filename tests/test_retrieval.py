"""Vector index, stale-index detection, twin averages and RAG in the translator (fake embedder and model)."""

import json

import numpy as np
import pandas as pd
import pytest

from riskdesk.checks.moves import Limits
from riskdesk.engine.drivers import DRIVER_IDS
from riskdesk.llm.client import LLMReply
from riskdesk.llm.translator import STRONG_MATCH, build_messages, translate
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


def answer(events):
    return json.dumps({"moves": {d: 0 for d in DRIVER_IDS} | {"oil": 25}, "reasoning": "r", "events_used": events})


def test_prompt_contains_evidence():
    msg = build_messages("Oil up", LIMITS, [twin("E011", 0.7, oil=12.5)])[0]["content"]
    assert "[E011]" in msg and "oil +12.5%" in msg and "strong match" in msg


def test_cited_event_must_be_one_that_was_retrieved():
    twins = [twin("E011", 0.7, oil=10)]
    res = translate("Oil up", FakeClient([answer(["E999"]), answer(["E011"])]), LIMITS, twins)
    assert res.ok and res.attempts == 2 and res.events_used == ["E011"]
    assert "not given as evidence" in res.errors[0][0]


def test_fallback_uses_strong_twins_only():
    twins = [twin("A", 0.8, oil=30), twin("B", STRONG_MATCH - 0.1, oil=-50)]
    res = translate("Oil up", FakeClient(["bad"] * 3), LIMITS, twins)
    assert res.status == "fallback" and res.moves["oil"] == pytest.approx(30.0)
    assert res.events_used == ["A"]


def test_no_fallback_without_strong_twins():
    res = translate("Aliens", FakeClient(["bad"] * 3), LIMITS, [twin("A", 0.4, oil=30)])
    assert res.status == "failed" and res.moves is None


def test_fallback_is_clipped_to_limits():
    tight = LIMITS | {"oil": Limits(-20.0, 20.0)}
    res = translate("Oil up", FakeClient(["bad"] * 3), tight, [twin("A", 0.9, oil=60)])
    assert res.moves["oil"] == 20.0
