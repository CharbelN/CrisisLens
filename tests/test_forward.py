"""Forward mode end to end with a fake model, plus the table converters."""

import json

import pandas as pd
import pytest

from riskdesk import ui
from riskdesk.checks.moves import Limits
from riskdesk.engine.drivers import DRIVER_IDS
from riskdesk.engine.portfolio import InvalidPortfolioError
from riskdesk.forward import run_forward, run_manual
from riskdesk.llm.client import LLMReply, LLMUnavailableError

LIMITS = {d: Limits(-90.0, 500.0) for d in DRIVER_IDS} | {"us10y": Limits(-300.0, 300.0)}
PORTFOLIO = {"us_stocks": 10_000, "us_bonds": 10_000}
# 4 made-up 20-day windows; only equity and us10y matter for this portfolio
CHANGES = pd.DataFrame(
    {d: [0.0] * 4 for d in DRIVER_IDS},
    index=pd.date_range("2020-01-01", periods=4),
)
CHANGES["equity"] = [-20.0, -5.0, 0.0, 5.0]


class FakeClient:
    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    def chat_json(self, messages):
        if self.error:
            raise self.error
        return LLMReply(self.text, "fake", 1, 1, 0.0, False)


def good_answer():
    moves = {d: 0 for d in DRIVER_IDS} | {"equity": -10, "us10y": 50}
    return json.dumps({"moves": moves, "reasoning": "stocks fall, yields rise", "events_used": []})


def test_forward_ok_hand_calculated():
    # stocks: 10,000 x -10% = -1,000
    # bonds:  -7 x 0.0050 x 10,000 = -350
    # total -1,350 on 20,000 = -6.75%
    res = run_forward(PORTFOLIO, "Rates up, stocks down", FakeClient(good_answer()), LIMITS, CHANGES)
    assert res.status == "ok"
    assert res.stress.pnl == pytest.approx(-1350)
    assert res.stress.pnl_pct == pytest.approx(-6.75)
    # worst historical window: equity -20% -> -2,000 loss; with 4 windows 99% VaR = worst
    assert res.var[0.99].var == pytest.approx(2000)
    text = ui.summary_markdown(res)
    assert "Loss: $1,350 (-6.75%)" in text and "0.7×" in text


def test_forward_translation_failed():
    res = run_forward(PORTFOLIO, "x", FakeClient("not json"), LIMITS, CHANGES)
    assert res.status == "translation_failed" and res.stress is None
    assert "3 attempts" in res.message


def test_forward_llm_unavailable():
    res = run_forward(PORTFOLIO, "x", FakeClient(error=LLMUnavailableError("no key")), LIMITS, CHANGES)
    assert res.status == "llm_unavailable" and "by hand" in res.message


def test_manual_recalculation():
    res = run_manual(PORTFOLIO, {"equity": -10, "us10y": 50}, CHANGES)
    assert res.status == "manual" and res.stress.pnl == pytest.approx(-1350)


def test_tables_round_trip():
    assert ui.parse_portfolio(ui.portfolio_table(ui.SAMPLE_PORTFOLIO)) == {
        k: float(v) for k, v in ui.SAMPLE_PORTFOLIO.items()
    }
    moves = {"equity": -8.0, "oil": 40.0, "gold": 5.0, "usd": 1.0, "btc": -12.0, "us10y": 50.0}
    assert ui.parse_moves(ui.moves_table(moves)) == moves


def test_portfolio_parsing_blanks_and_commas():
    df = ui.portfolio_table({}).astype(object)
    df.iloc[0, 1] = "12,500"
    df.iloc[1, 1] = ""
    parsed = ui.parse_portfolio(df)
    assert parsed["us_stocks"] == 12_500 and parsed["us_bonds"] == 0


def test_negative_amount_rejected():
    df = ui.portfolio_table({"us_stocks": -5})
    with pytest.raises(InvalidPortfolioError):
        run_manual(ui.parse_portfolio(df), {}, CHANGES)


def test_text_amount_rejected():
    df = ui.portfolio_table({}).astype(object)
    df.iloc[0, 1] = "lots"
    with pytest.raises(ValueError):
        ui.parse_portfolio(df)


def test_llm_down_uses_strong_twins():
    from riskdesk.retrieval.index import Twin

    class TwoTwins:  # both strong, both equity -10% -> they agree -> average -10%
        def search(self, text):
            moves = {d: 0.0 for d in DRIVER_IDS} | {"equity": -10.0}
            return [Twin("E1", "Oil shock", "2020-01-01", "d", "market", 0.9, moves),
                    Twin("E2", "Oil shock 2", "2021-01-01", "d", "market", 0.8, moves)]

    res = run_forward(PORTFOLIO, "Oil up", FakeClient(error=LLMUnavailableError("down")), LIMITS, CHANGES, TwoTwins())
    assert res.status == "fallback" and res.stress.pnl == pytest.approx(-1000)
    assert ui.twins_table(res).iloc[0]["Used by AI"] == "yes"


def test_twins_table_shows_first_day_and_20_days():
    from riskdesk.retrieval.index import Twin

    zero = {d: 0.0 for d in DRIVER_IDS}
    t = Twin("E1", "Oil shock", "2020-01-01", "d", "market", 0.9, zero | {"oil": 30.0, "btc": None}, zero | {"oil": 5.5, "btc": None})
    res = run_manual(PORTFOLIO, {}, CHANGES)
    res.twins = [t]
    row = ui.twins_table(res).iloc[0]
    assert row["Oil (Brent) (%): first day / 20 days"] == "+5.5 / +30"
    assert row["Bitcoin (%): first day / 20 days"] == "n/a" and row["Strong match"] == "yes"
