"""VaR / ES and data-transformation tests on tiny made-up data, countable by hand."""

import numpy as np
import pandas as pd
import pytest

from riskdesk.data.market import driver_changes
from riskdesk.engine.var import historical_var_es, var_es_from_pnl


def test_var_es_counted_by_hand_100_outcomes():
    # outcomes -1, -2, ..., -100 (in shuffled order)
    # 95%: k = 100 x 0.05 = 5 worst -> -100,-99,-98,-97,-96
    #      VaR = 96, ES = mean(100..96) = 98
    # 99%: k = 1 -> VaR = ES = 100
    pnl = np.random.default_rng(0).permutation(-np.arange(1, 101, dtype=float))
    assert var_es_from_pnl(pnl, 0.95) == pytest.approx((96, 98))
    assert var_es_from_pnl(pnl, 0.99) == pytest.approx((100, 100))


def test_var_small_sample_uses_at_least_one():
    # 10 outcomes at 99%: 10 x 0.01 = 0.1 -> k = 1, the single worst
    assert var_es_from_pnl([5, -3, 2, -8, 1, 0, 4, -1, 3, 2], 0.99) == pytest.approx((8, 8))


def _levels() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "equity": [100, 110, 99, 99],
            "oil": [50, 50, 50, 50],
            "gold": [10, 10, 10, 10],
            "usd": [90, 90, 90, 90],
            "btc": [np.nan, np.nan, 200, 100],
            "us10y": [2.00, 2.50, 2.50, 1.50],
        },
        index=idx,
    )


def test_driver_changes_units():
    ch = driver_changes(_levels(), 1)
    # equity 100 -> 110 = +10%, 110 -> 99 = -10%
    assert ch["equity"].tolist() == pytest.approx([10, -10, 0])
    # yield 2.00 -> 2.50 = +50 bp, then 0, then -100 bp
    assert ch["us10y"].tolist() == pytest.approx([50, 0, -100])
    # bitcoin 200 -> 100 = -50%, earlier rows missing
    assert np.isnan(ch["btc"].iloc[0]) and ch["btc"].iloc[2] == pytest.approx(-50)


def test_driver_changes_multi_day():
    ch = driver_changes(_levels(), 2)
    # equity 100 -> 99 = -1%, 110 -> 99 = -10%
    assert ch["equity"].tolist() == pytest.approx([-1, -10])


def test_historical_var_on_tiny_history():
    # 1-day equity moves: +10%, -10%, 0%  -> on $1,000 stocks: +100, -100, 0
    # 99%, 3 outcomes -> k = 1 -> VaR = ES = $100 = 10%
    r = historical_var_es({"us_stocks": 1_000}, driver_changes(_levels(), 1), 1, 0.99)
    assert r.var == pytest.approx(100) and r.es == pytest.approx(100)
    assert r.var_pct == pytest.approx(10)
    assert r.n_scenarios == 3


def test_bitcoin_rows_without_data_dropped():
    # only the last day has a bitcoin move (-50%) -> 1 scenario, loss 500
    r = historical_var_es({"bitcoin": 1_000}, driver_changes(_levels(), 1), 1, 0.95)
    assert r.n_scenarios == 1
    assert r.var == pytest.approx(500)
