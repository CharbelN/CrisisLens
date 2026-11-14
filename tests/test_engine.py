"""Risk engine tests. Every expected number is worked out by hand in the comment."""

import pytest

from riskdesk.engine.drivers import InvalidMovesError, validate_moves
from riskdesk.engine.portfolio import InvalidPortfolioError, apply_moves, portfolio_value


def test_brief_example_stocks_and_bonds():
    # $10,000 stocks, stocks -10%           -> -$1,000
    # $5,000 bond fund, duration 7, +50 bp  -> 5,000 x (-7 x 0.005) = -$175
    # total -$1,175 on $15,000              -> -7.8333%
    r = apply_moves({"us_stocks": 10_000, "us_bonds": 5_000}, {"equity": -10, "us10y": 50})
    assert r.start_value == 15_000
    assert r.pnl == pytest.approx(-1_175)
    assert r.loss == pytest.approx(1_175)
    assert r.pnl_pct == pytest.approx(-7.833333, abs=1e-5)
    assert r.end_value == pytest.approx(13_825)
    per = {h.holding_id: h for h in r.holdings}
    assert per["us_stocks"].pnl == pytest.approx(-1_000)
    assert per["us_bonds"].return_pct == pytest.approx(-3.5)


def test_mixed_portfolio_oil_shock():
    # Oil shock: oil +40, stocks -8, gold +5, btc -15, yield +50 bp
    # stocks  20,000 x -0.08         = -1,600
    # gold     5,000 x +0.05         =   +250
    # bonds   10,000 x -7 x 0.005    =   -350
    # bitcoin  2,000 x -0.15         =   -300
    # cash     3,000 x 0             =      0
    # total -2,000 on 40,000 = -5%
    portfolio = {"us_stocks": 20_000, "gold": 5_000, "us_bonds": 10_000, "bitcoin": 2_000, "cash": 3_000}
    moves = {"oil": 40, "equity": -8, "gold": 5, "btc": -15, "us10y": 50}
    r = apply_moves(portfolio, moves)
    assert r.pnl == pytest.approx(-2_000)
    assert r.pnl_pct == pytest.approx(-5.0)


def test_falling_yields_help_bonds():
    # yield -100 bp -> bond fund +7%: 1,000 x 0.07 = +70
    r = apply_moves({"us_bonds": 1_000}, {"us10y": -100})
    assert r.pnl == pytest.approx(70)
    assert r.loss == 0


def test_missing_drivers_mean_no_change():
    r = apply_moves({"gold": 1_000}, {"equity": -50})
    assert r.pnl == 0
    assert r.moves["gold"] == 0


def test_zero_amounts_ignored_and_value():
    assert portfolio_value({"us_stocks": 1_000, "gold": 0, "cash": 500}) == 1_500


@pytest.mark.parametrize(
    "moves",
    [
        {"equity": -150},          # price can't fall more than 100%
        {"equity": -100},          # exactly -100% means price 0, also rejected
        {"oil": 5_000},            # absurd rise
        {"us10y": 3_000},          # +30 percentage points
        {"equity": float("nan")},
        {"equity": "a lot"},
        {"copper": 10},            # unknown driver
    ],
)
def test_bad_moves_rejected(moves):
    with pytest.raises(InvalidMovesError):
        validate_moves(moves)


@pytest.mark.parametrize(
    "portfolio",
    [{}, {"us_stocks": 0}, {"us_stocks": -100}, {"tesla": 1_000}, {"gold": "lots"}],
)
def test_bad_portfolios_rejected(portfolio):
    with pytest.raises(InvalidPortfolioError):
        apply_moves(portfolio, {"equity": -10})
