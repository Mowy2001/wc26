"""Admitted residual-block tilts, centralised.

A team's log-lambda tilt is the sum over admitted blocks of
beta_block * feature_z(team). Currently: squad fatigue only (scripts/19).

Football capital was REMOVED from the official model (2026-06-18): its only
predictive value came from clubelo's Europe-only bias (the floor penalising
non-European-league players); on a fair global source it fails the gate, and
'no capital' both predicts better and is unbiased. See METHODOLOGY. It lives
on only as a shadow in the model lab (scripts/25).
"""
import json

import pandas as pd


def load_team_tilt(outputs: str = "outputs") -> dict[str, float] | None:
    tilt: dict[str, float] = {}
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
