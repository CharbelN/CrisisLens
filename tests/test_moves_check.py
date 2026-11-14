"""Plausibility limits and the JSON format check, on tiny made-up data."""

import json

import pandas as pd
import pytest

from riskdesk.checks.moves import Limits, check_translation, plausible_limits
from riskdesk.engine.drivers import DRIVER_IDS

WIDE = {d: Limits(-90.0, 500.0) for d in DRIVER_IDS} | {"us10y": Limits(-300.0, 300.0)}


def test_limits_counted_by_hand():
    # 2-day horizon, 4 days of prices.
    # equity 100, 110, 99, 121 -> 2-day moves: 99/100-1 = -1%, 121/110-1 = +10%
    #   widened x1.5 -> -1.5 and 15.0
    # us10y 4.00, 4.10, 4.50, 3.90 -> 2-day changes: +50 bp, -20 bp -> -30 and 75
    # other drivers constant -> 0 and 0
    idx = pd.date_range("2020-01-01", periods=4)
    levels = pd.DataFrame({d: [1.0] * 4 for d in DRIVER_IDS}, index=idx)
    levels["equity"] = [100, 110, 99, 121]
    levels["us10y"] = [4.00, 4.10, 4.50, 3.90]
    lim = plausible_limits(levels, horizon_days=2, widen=1.5)
    assert lim["equity"] == Limits(-1.5, 15.0)
    assert lim["us10y"].low == pytest.approx(-30.0)
    assert lim["us10y"].high == pytest.approx(75.0)
    assert lim["gold"] == Limits(0.0, 0.0)


def test_price_floor_is_minus_95():
    idx = pd.date_range("2020-01-01", periods=3)
    levels = pd.DataFrame({d: [1.0] * 3 for d in DRIVER_IDS}, index=idx)
    levels["btc"] = [100, 30, 30]  # -70% x 1.5 = -105% -> floored at -95%
    assert plausible_limits(levels, horizon_days=1)["btc"].low == -95.0


def answer(**moves):
    full = {d: 0 for d in DRIVER_IDS} | moves
    return json.dumps({"moves": full, "reasoning": "r", "events_used": ["E1"]})


def test_good_answer_passes():
    res = check_translation(answer(equity=-8, oil=40, us10y=50), WIDE)
    assert res.ok and res.moves["oil"] == 40.0 and res.events_used == ["E1"]


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("not json", "not valid JSON"),
        (json.dumps({"equity": -5}), '"moves"'),
        (json.dumps({"moves": {"equity": -5}}), "Missing driver"),
        (answer(equity="-5%"), "plain number"),
        (answer(equity=-150), "outside the plausible range"),
        (json.dumps({"moves": {d: 0 for d in DRIVER_IDS} | {"silver": 3}}), "Unknown driver"),
    ],
)
def test_bad_answers_are_rejected(text, fragment):
    res = check_translation(text, WIDE)
    assert not res.ok
    assert any(fragment in e for e in res.errors)


def test_real_limits_are_sensible():
    from riskdesk.checks.moves import default_limits

    lim = default_limits()
    assert lim["equity"].low < -30 and lim["equity"].high > 15
    assert lim["btc"].low >= -95
    assert lim["us10y"].low < -50 and lim["us10y"].high > 50
