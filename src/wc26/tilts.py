"""Admitted residual-block tilts, centralised.

A team's log-lambda tilt is the sum over admitted blocks of
beta_block * feature_z(team). Currently: football capital (scripts/11-12)
and squad fatigue (scripts/19), both admitted on probation — re-judged on
WC2026 in the final report. Returns None if no block artifact exists, so
callers degrade gracefully.
"""
import json

import pandas as pd


def load_team_tilt(outputs: str = "outputs") -> dict[str, float] | None:
    tilt: dict[str, float] = {}
    try:
        cap = pd.read_csv(f"{outputs}/capital.csv").query("tournament == 'wc2026'")
        beta = json.load(open(f"{outputs}/capital_beta.json"))["beta_capital"]
        for r in cap.itertuples(index=False):
            tilt[r.team] = tilt.get(r.team, 0.0) + beta * r.capital_z
    except FileNotFoundError:
        pass
    try:
        fat = pd.read_csv(f"{outputs}/fatigue.csv", index_col=0)["fatigue_z"]
        beta = json.load(open(f"{outputs}/fatigue_beta.json"))["beta_fatigue"]
        for team, z in fat.items():
            tilt[team] = tilt.get(team, 0.0) + beta * float(z)
    except FileNotFoundError:
        pass
    return tilt or None


def load_city_tilt(outputs: str = "outputs") -> dict | None:
    """Venue-dependent tilts keyed (team, city) from admitted city blocks.

    Currently the altitude block (scripts/22-23): lowland teams are damped at
    the high-altitude 2026 venues (Mexico City, Zapopan). Returns None if the
    artifact is absent, so callers degrade gracefully.
    """
    try:
        a = pd.read_csv(f"{outputs}/altitude_tilt.csv")
    except FileNotFoundError:
        return None
    return {(r.team, r.city): float(r.log_tilt) for r in a.itertuples(index=False)}
