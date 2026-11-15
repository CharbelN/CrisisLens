"""Reverse stress test: which market moves would make this portfolio lose X%?  (plain code, no AI)

The portfolio's return is a straight-line (linear) function of the moves:
    return (%) = a . m = a_equity x m_equity + ... + a_us10y x m_us10y
where a_j is the portfolio's return for a move of 1 unit in driver j (computed with the risk engine).
So "lose X%" means: find moves m with a . m = -X.

Many sets of moves do that. We want ones that are plausible by history. Plausibility is the
Mahalanobis distance: how far a set of moves is from a normal month, measured in units of
history's own spreads and correlations (stocks -15% with gold +5% is less unusual than
stocks -15% with gold -30%, because history shows the first combination more often).

Two ways to find scenarios:
1. Most likely (a formula): the set of moves with the SMALLEST distance that still gives the loss.
       m* = mu + S a (target - a . mu) / (a' S a)
   (mu = average 20-day move, S = covariance of 20-day moves). It is the textbook
   "most likely scenario" if moves were normally distributed. If a move breaks a sanity limit
   (e.g. bitcoin below -90%), that driver is held at the limit and the formula is solved again
   for the other drivers, given that value.
2. Scaled real months: every past 20-day window in which this portfolio lost money, scaled up or
   down so the loss is exactly the target, but at most MAX_SCALE times. Ranked by how unusual
   they are; we keep the most plausible ones that are clearly different from the scenarios already
   chosen (distance at least MIN_APART), preferably with a different main cause, and from a different
   period. These rules are relaxed one by one only if too few months pass them.
Every scenario must stay inside the sanity limits. Each one is then matched to the past event in
the crisis library whose real moves were closest (same distance).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from riskdesk.checks.moves import Limits
from riskdesk.engine.drivers import DRIVER_IDS, DRIVERS_BY_ID
from riskdesk.engine.portfolio import Portfolio, StressResult, apply_moves, needed_drivers

N_SCALED = 2  # scaled real months shown next to the most likely scenario
MIN_GAP_DAYS = 180  # scaled months must be at least this far apart (different episodes)
MAX_TARGET_PCT = 90.0
MAX_SCALE = 3.0  # a real month may be made at most 3 times as bad (relaxed only if nothing else fits)
MIN_APART = 2.0  # scenarios must differ: Mahalanobis distance between any two at least this (relaxed last)


class ReverseError(ValueError):
    """The question cannot be answered (bad target, portfolio that cannot lose money...)."""


@dataclass(frozen=True)
class Match:
    id: str
    name: str
    date: str
    distance: float


@dataclass
class ReverseScenario:
    kind: str  # "most_likely" or "scaled_month"
    moves: dict[str, float | None]  # None = no data for that driver in that month (bitcoin before 2014)
    stress: StressResult
    distance: float  # Mahalanobis distance: 0 = a normal month; larger = more unusual
    rarer_than: float  # share of past 20-day windows that were LESS unusual (0 to 1)
    main_driver: str  # the driver that causes the largest part of the loss
    window_end: str | None = None  # scaled month: last day of the real window
    held: list[str] = field(default_factory=list)  # most likely: drivers held at a sanity limit
    scale: float | None = None  # scaled month: how much the real moves were multiplied by
    match: Match | None = None


@dataclass
class ReverseResult:
    target_pct: float
    scenarios: list[ReverseScenario]
    notes: list[str] = field(default_factory=list)  # e.g. scenarios dropped for breaking a limit
    n_windows: int = 0
    first_date: str = ""
    last_date: str = ""


# --- building blocks ------------------------------------------------------------
def exposures(portfolio: Portfolio) -> pd.Series:
    """a_j: the portfolio's return (%) for a move of 1 unit (1% or 1 bp) in each driver."""
    base = {d: 0.0 for d in DRIVER_IDS}
    return pd.Series({d: apply_moves(portfolio, base | {d: 1.0}).pnl_pct for d in DRIVER_IDS})


def history_for(portfolio: Portfolio, changes: pd.DataFrame) -> pd.DataFrame:
    """Past windows with data for every driver the portfolio holds (as for VaR)."""
    needed = needed_drivers(portfolio)
    return changes[list(DRIVER_IDS)].dropna(subset=needed) if needed else changes[list(DRIVER_IDS)]


def distances(rows: pd.DataFrame, mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Mahalanobis distance of each row, using only the drivers that row has data for."""
    out = pd.Series(np.nan, index=rows.index)
    pattern = rows.notna()
    for key, idx in pattern.groupby(list(pattern.columns)).groups.items():
        cols = [d for d, has in zip(pattern.columns, key, strict=True) if has]
        if not cols:
            continue
        x = rows.loc[idx, cols].to_numpy() - mu[cols].to_numpy()
        inv = np.linalg.pinv(cov.loc[cols, cols].to_numpy())
        out.loc[idx] = np.sqrt(np.einsum("ij,jk,ik->i", x, inv, x))
    return out


def rarer_than(rows: pd.DataFrame, dist: pd.Series, hist: pd.DataFrame, hist_dist: pd.Series) -> pd.Series:
    """For each row: the share of past windows that were LESS unusual (0 to 1).

    Compared only with past windows that have data for the same drivers, so a month without
    bitcoin data (5 drivers) is not compared with 6-driver distances.
    """
    out = pd.Series(np.nan, index=rows.index)
    hist_key = hist.notna().apply(tuple, axis=1)
    keys = rows.notna().apply(tuple, axis=1)
    for key in keys.unique():
        ref = np.sort(hist_dist[hist_key == key].to_numpy())
        if len(ref):
            sel = keys == key
            out[sel] = np.searchsorted(ref, dist[sel].to_numpy()) / len(ref)
    return out


def _round(moves: Mapping[str, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for d, v in moves.items():
        if v is None or pd.isna(v):
            out[d] = None
        else:
            out[d] = round(float(v), 1) if DRIVERS_BY_ID[d].unit == "pct" else float(round(float(v)))
    return out


def _outside(moves: Mapping[str, float | None], limits: dict[str, Limits]) -> list[str]:
    return [d for d, v in moves.items() if v is not None and not limits[d].low <= v <= limits[d].high]


def main_driver(a: pd.Series, moves: Mapping[str, float | None]) -> str:
    loss = {d: -a[d] * (moves[d] or 0.0) for d in DRIVER_IDS}
    return max(loss, key=loss.get)


def nearest_event(moves: Mapping[str, float | None], library: pd.DataFrame, cov: pd.DataFrame,
                  needed: list[str] = ()) -> Match | None:
    """The crisis-library event whose real 20-day moves are closest.

    Distance: Mahalanobis over the drivers both have data for, divided by the number of those
    drivers (the average gap per driver). Without that, events before 2014 (no bitcoin data) would
    look closer just because one driver is missing from the comparison. Events without data for a
    driver the portfolio holds (`needed`) are skipped, as in the history used for the search.
    """
    best: Match | None = None
    m = pd.Series(moves, dtype=float)
    usable = library.dropna(subset=list(needed)) if needed else library
    for _, ev in usable.iterrows():
        diff = m - ev[list(DRIVER_IDS)].astype(float)
        cols = [d for d in DRIVER_IDS if pd.notna(diff[d])]
        if not cols:
            continue
        x = diff[cols].to_numpy()
        dist = float(np.sqrt(x @ np.linalg.pinv(cov.loc[cols, cols].to_numpy()) @ x / len(cols)))
        if best is None or dist < best.distance:
            best = Match(str(ev["id"]), str(ev["name"]), str(ev["date"]), dist)
    return best


def _inside(limit: Limits, unit: str) -> tuple[float, float]:
    """The limits rounded inwards to the precision moves are shown with (0.1% or 1 bp)."""
    if unit == "pct":
        return np.ceil(limit.low * 10) / 10, np.floor(limit.high * 10) / 10
    return float(np.ceil(limit.low)), float(np.floor(limit.high))


def most_likely_moves(a: pd.Series, mu: pd.Series, cov: pd.DataFrame, target: float,
                      limits: dict[str, Limits]) -> tuple[dict[str, float], list[str]]:
    """The formula, with drivers that break a limit held at the limit. Returns (moves, held drivers)."""
    held: dict[str, float] = {}
    for _ in DRIVER_IDS:
        free = [d for d in DRIVER_IDS if d not in held]
        h = list(held)
        mu_f, cov_ff = mu[free].to_numpy(), cov.loc[free, free].to_numpy()
        t = target
        if h:  # the free drivers, given the held ones (conditional mean and covariance)
            c = np.array([held[d] for d in h])
            gain = cov.loc[free, h].to_numpy() @ np.linalg.pinv(cov.loc[h, h].to_numpy())
            mu_f = mu_f + gain @ (c - mu[h].to_numpy())
            cov_ff = cov_ff - gain @ cov.loc[h, free].to_numpy()
            t = target - float(a[h] @ c)
        a_f = a[free].to_numpy()
        s = cov_ff @ a_f
        if abs(float(a_f @ s)) < 1e-12:
            break  # the free drivers cannot change the loss any more
        m = dict(zip(free, mu_f + s * (t - float(a_f @ mu_f)) / float(a_f @ s), strict=True)) | held
        broken = {d: _inside(limits[d], DRIVERS_BY_ID[d].unit) for d in free
                  if not limits[d].low <= round(m[d], 1) <= limits[d].high}
        if not broken:
            return m, sorted(held)
        for d, (lo, hi) in broken.items():
            held[d] = lo if m[d] < lo else hi
    return {}, sorted(held)


# --- the search -------------------------------------------------------------------
def reverse_stress(
    portfolio: Portfolio, target_pct: float, changes: pd.DataFrame, limits: dict[str, Limits],
    library: pd.DataFrame | None = None, n_scaled: int = N_SCALED,
) -> ReverseResult:
    """Scenarios that make the portfolio lose `target_pct` percent over the 20-day horizon."""
    if not 0 < target_pct <= MAX_TARGET_PCT:
        raise ReverseError(f"The target loss must be between 0% and {MAX_TARGET_PCT:g}%.")
    a = exposures(portfolio)
    if (a.abs() < 1e-12).all():
        raise ReverseError("This portfolio does not move with any market driver (only cash), so it cannot lose money.")

    hist = history_for(portfolio, changes)
    mu, cov = hist.mean(), hist.cov()  # cov uses every pair of drivers where both have data
    hist_dist = distances(hist, mu, cov)
    target = -target_pct
    result = ReverseResult(target_pct, [], n_windows=len(hist),
                           first_date=hist.index[0].date().isoformat(), last_date=hist.index[-1].date().isoformat())

    def make(kind: str, raw: Mapping[str, float | None], **extra) -> ReverseScenario | None:
        moves = _round({d: raw[d] for d in DRIVER_IDS})
        bad = _outside(moves, limits)
        if bad:
            what = "The most likely scenario" if kind == "most_likely" else f"The scaled month to {extra['window_end']}"
            result.notes.append(f"{what} was dropped: {', '.join(bad)} would move beyond the sanity limits.")
            return None
        row = pd.DataFrame([moves], dtype=float)
        d = distances(row, mu, cov)
        sc = ReverseScenario(kind, moves, apply_moves(portfolio, {k: v or 0.0 for k, v in moves.items()}),
                             float(d.iloc[0]), float(rarer_than(row, d, hist, hist_dist).iloc[0]),
                             main_driver(a, moves), **extra)
        if library is not None:
            sc.match = nearest_event(moves, library, cov, needed_drivers(portfolio))
        return sc

    # 1. the most likely moves (formula)
    m_star, held = most_likely_moves(a, mu, cov, target, limits)
    if m_star:
        best = make("most_likely", m_star, held=held)
        if best is not None:
            result.scenarios.append(best)
    else:
        result.notes.append("The most likely scenario was dropped: it cannot reach the loss within the sanity limits.")

    # 2. scaled real months
    loss = hist.fillna(0.0) @ a  # the portfolio's real return in each past window
    k = target / loss[loss < 0]
    scaled = hist.loc[k.index].mul(k, axis=0)
    dist = distances(scaled, mu, cov)
    rank = pd.DataFrame({"rarer": rarer_than(scaled, dist, hist, hist_dist), "dist": dist})
    order = rank.sort_values(["rarer", "dist"]).index  # least unusual first
    picked: list[ReverseScenario] = []
    inv = np.linalg.pinv(cov.to_numpy())

    def apart(moves: Mapping[str, float | None]) -> bool:
        x = np.array([v or 0.0 for v in moves.values()])
        return all(np.sqrt((x - y) @ inv @ (x - y)) >= MIN_APART
                   for y in (np.array([v or 0.0 for v in p.moves.values()]) for p in result.scenarios + picked))

    # rules, strictest first: (largest scale, new main cause, clearly different)
    passes = ((MAX_SCALE, True, True), (MAX_SCALE, False, True), (MAX_SCALE, False, False),
              (np.inf, False, True), (np.inf, False, False))
    for max_scale, strict, differ in passes:
        for end in order:
            if len(picked) == n_scaled:
                break
            if k[end] > max_scale or end in {pd.Timestamp(p.window_end) for p in picked}:
                continue
            if any(abs((end - pd.Timestamp(p.window_end)).days) < MIN_GAP_DAYS for p in picked):
                continue
            moves = _round(scaled.loc[end].to_dict())
            if _outside(moves, limits):
                continue  # silently skip: many scaled months break a limit, only the best are tried
            if strict and main_driver(a, moves) in {p.main_driver for p in result.scenarios + picked}:
                continue
            if differ and not apart(moves):
                continue
            sc = make("scaled_month", moves, window_end=end.date().isoformat(), scale=round(float(k[end]), 2))
            if sc is not None:
                picked.append(sc)
    result.scenarios += picked
    if not result.scenarios:
        result.notes.append(f"No plausible set of moves loses {target_pct:g}% within the sanity limits.")
    return result
