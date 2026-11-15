"""Converters between the app's tables (pandas DataFrames) and the engine's dicts,
plus the text shown on screen. Kept apart from app.py so they can be tested.
"""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from riskdesk.engine.drivers import DRIVERS, DRIVERS_BY_ID, InvalidMovesError
from riskdesk.engine.holdings import HOLDINGS, HOLDINGS_BY_ID
from riskdesk.engine.portfolio import InvalidPortfolioError
from riskdesk.forward import ForwardResult

SAMPLE_PORTFOLIO = {"us_stocks": 25_000, "us_bonds": 10_000, "gold": 5_000, "oil": 0, "bitcoin": 5_000, "cash": 5_000}

HOLDING_BY_LABEL = {h.label: h.id for h in HOLDINGS}
DRIVER_BY_LABEL = {d.label: d.id for d in DRIVERS}
PORTFOLIO_COLS = ["Holding", "Amount (USD)"]
MOVES_COLS = ["Risk driver", "Move", "Unit"]


def _number(value, what: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    try:
        out = float(str(value).replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        raise ValueError(f"{what}: '{value}' is not a number.") from None
    if not math.isfinite(out):
        raise ValueError(f"{what}: '{value}' is not a finite number.")
    return out


# --- portfolio -------------------------------------------------------------
def portfolio_table(portfolio: Mapping[str, float] = SAMPLE_PORTFOLIO) -> pd.DataFrame:
    return pd.DataFrame([[h.label, float(portfolio.get(h.id, 0))] for h in HOLDINGS], columns=PORTFOLIO_COLS)


def parse_portfolio(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip()
        if label not in HOLDING_BY_LABEL:
            raise InvalidPortfolioError(f"Unknown holding '{label}'.")
        out[HOLDING_BY_LABEL[label]] = _number(row.iloc[1], label)
    return out


# --- moves -----------------------------------------------------------------
def _unit(d) -> str:
    return "%" if d.unit == "pct" else "bp"


def moves_table(moves: Mapping[str, float] | None = None) -> pd.DataFrame:
    moves = moves or {}
    return pd.DataFrame([[d.label, float(moves.get(d.id, 0.0)), _unit(d)] for d in DRIVERS], columns=MOVES_COLS)


def parse_moves(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip()
        if label not in DRIVER_BY_LABEL:
            raise InvalidMovesError(f"Unknown risk driver '{label}'.")
        out[DRIVER_BY_LABEL[label]] = _number(row.iloc[1], label)
    return out


# --- results ---------------------------------------------------------------
def holdings_table(result: ForwardResult) -> pd.DataFrame:
    cols = ["Holding", "Amount (USD)", "Return (%)", "Profit / loss (USD)"]
    if result.stress is None:
        return pd.DataFrame(columns=cols)
    rows = [
        [HOLDINGS_BY_ID[h.holding_id].label, round(h.amount, 2), round(h.return_pct, 2) + 0.0, round(h.pnl, 2) + 0.0]
        for h in result.stress.holdings
    ]
    return pd.DataFrame(rows, columns=cols)


def twins_table(result: ForwardResult) -> pd.DataFrame:
    cols = ["Id", "Past event", "Date", "Similarity", "Used by AI"] + [f"{d.label} ({_unit(d)})" for d in DRIVERS]
    used = set(result.translation.events_used) if result.translation else set()
    rows = [
        [t.id, t.name, t.date, t.similarity, "yes" if t.id in used else ""]
        + ["n/a" if t.moves[d.id] is None else t.moves[d.id] for d in DRIVERS]
        for t in result.twins
    ]
    return pd.DataFrame(rows, columns=cols)


def money(x: float) -> str:
    return f"${x:,.0f}"


def summary_markdown(result: ForwardResult) -> str:
    icon = {"ok": "✅", "manual": "✏️", "fallback": "🔁"}.get(result.status, "⚠️")
    lines = [f"{icon} {result.message}"]
    s = result.stress
    if s is None:
        return "\n\n".join(lines)

    word = "Loss" if s.pnl < 0 else "Gain"
    lines.append(f"### {word}: {money(abs(s.pnl))} ({s.pnl_pct:+.2f}%) on a {money(s.start_value)} portfolio")
    v95, v99 = result.var.get(0.95), result.var.get(0.99)
    if v95 and v99:
        lines.append(
            f"For context, from history ({v99.first_date} to {v99.last_date}, {v99.n_scenarios:,} windows of "
            f"{v99.horizon_days} trading days): a bad month (95% VaR) loses {money(v95.var)} ({v95.var_pct:.2f}%), "
            f"a very bad month (99% VaR) loses {money(v99.var)} ({v99.var_pct:.2f}%), and the average of the worst 1% "
            f"(99% Expected Shortfall) is {money(v99.es)} ({v99.es_pct:.2f}%)."
        )
        if s.pnl < 0 and v99.var > 0:
            lines.append(f"This scenario's loss is **{s.loss / v99.var:.1f}×** the 99% VaR.")
    tr = result.translation
    if tr is not None and tr.events_used:
        names = {t.id: t.name for t in result.twins}
        cited = ", ".join(f"{names.get(e, e)} ({e})" for e in tr.events_used)
        lines.append(f"**Past events relied on:** {cited}")
    if tr is not None and tr.stated_levels and tr.moves:
        parts = []
        for d_id, level in tr.stated_levels.items():
            d = DRIVERS_BY_ID[d_id]
            how = "set by code (the model's move did not match)" if d_id in tr.level_fixes else "checked by code"
            parts.append(f"{d.label}: stated level {level:g} → move {tr.moves[d_id]:+g} {_unit(d)}, {how}")
        lines.append("**Levels stated in the scenario** (moves from the latest close in the data): " + "; ".join(parts))
    if tr is not None and tr.reasoning:
        lines.append(f"**Model's reasoning:** {tr.reasoning}")
        cache_note = "from cache" if tr.latency_s == 0 else f"{tr.latency_s:.1f} s"
        if tr.attempts:
            lines.append(f"<small>Model {tr.model}, {tr.attempts} attempt(s), {cache_note}.</small>")
    return "\n\n".join(lines)
