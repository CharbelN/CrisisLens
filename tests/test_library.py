"""Crisis library windows and moves, counted by hand."""

import pandas as pd
import pytest

from riskdesk.data.library import check_events, event_moves, event_window, load_events
from riskdesk.data.market import load_market_data
from riskdesk.engine.drivers import DRIVER_IDS


def small_levels():
    # 6 trading days (weekend gap between 3rd and 6th Jan)
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])
    df = pd.DataFrame({d: [100.0] * 6 for d in DRIVER_IDS}, index=idx)
    df["equity"] = [100, 100, 100, 90, 95, 80]
    df["us10y"] = [2.00, 2.00, 2.00, 1.80, 1.75, 1.70]
    df["btc"] = [None, None, None, 50, 60, 70]
    return df


def test_window_starts_at_last_close_before_event():
    # event on Monday 6 Jan -> start = Friday 3 Jan; horizon 2 -> end = Tue 7 Jan
    start, end = event_window(small_levels(), pd.Timestamp("2020-01-06"), horizon=2)
    assert (start, end) == (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-07"))
    # event on a Sunday -> also starts Friday
    assert event_window(small_levels(), pd.Timestamp("2020-01-05"), horizon=2)[0] == pd.Timestamp("2020-01-03")


def test_moves_by_hand():
    # start 3 Jan, end 7 Jan: equity 100 -> 95 = -5%; us10y 2.00 -> 1.75 = -25 bp
    # btc has no price on 3 Jan -> n/a
    m = event_moves(small_levels(), pd.Timestamp("2020-01-06"), horizon=2)
    assert m["equity"] == pytest.approx(-5.0)
    assert m["us10y"] == pytest.approx(-25.0)
    assert m["btc"] is None
    assert m["gold"] == 0.0


def test_window_too_close_to_data_end():
    with pytest.raises(ValueError, match="fewer than"):
        event_window(small_levels(), pd.Timestamp("2020-01-07"), horizon=5)


def test_lehman_matches_csv_prices():
    # E013 Lehman, 2008-09-15 (Monday): start = Fri 2008-09-12, end = 20 trading days later
    levels = load_market_data()
    start, end = event_window(levels, pd.Timestamp("2008-09-15"))
    assert start == pd.Timestamp("2008-09-12")
    expected = (levels.at[end, "equity"] / levels.at[start, "equity"] - 1) * 100
    assert event_moves(levels, pd.Timestamp("2008-09-15"))["equity"] == pytest.approx(expected, abs=0.01)
    assert expected < -20  # the S&P fell more than 20% in the month after Lehman


def test_real_event_list_passes_checks():
    assert check_events(load_events(), load_market_data()) == []
