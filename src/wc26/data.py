"""Data loading and 2026-tournament structure utilities.

Key trick: the raw results file already contains every 2026 World Cup
group-stage fixture (scores = NA until played). Since each group of 4 is a
round-robin of 6 internal matches, we can reconstruct the 12 groups
*algorithmically* from the fixture graph (connected components) instead of
hardcoding the draw — robust to data updates and typo-free by construction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# 2026 World Cup host nations: their home matches are NOT neutral venues.
# The raw flag is correct today, but upstream has historically marked
# tournament matches neutral=True — the override makes the invariant explicit
# and robust to dataset refreshes (no-op when the raw data is already correct).
WC2026_HOSTS = ("United States", "Mexico", "Canada")


def load_results() -> pd.DataFrame:
    """Full international match history (1872 -> today + future fixtures).

    Enforces the 2026 host-advantage invariant: a match whose home_team is a
    host nation playing on its own soil (country == home_team) is never
    neutral, regardless of how the raw file flags it.
    """
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    host_home = df["home_team"].isin(WC2026_HOSTS) & (df["country"] == df["home_team"])
    df.loc[host_home, "neutral"] = False
    return df


def load_goalscorers() -> pd.DataFrame:
    """Goal-level history: scorer, minute, own-goal and penalty flags.

    Used by the player layer (Golden Boot / most-distinct-scorers): gives
    historical within-team goal share and penalty-taker identification.
    """
    return pd.read_csv(RAW / "goalscorers.csv", parse_dates=["date"])


def load_shootouts() -> pd.DataFrame:
    """Historical penalty shootouts (for knockout shootout calibration)."""
    return pd.read_csv(RAW / "shootouts.csv", parse_dates=["date"])


# --------------------------------------------------------------------------- #
# 2026 tournament structure
# --------------------------------------------------------------------------- #
GROUP_STAGE_END = pd.Timestamp("2026-06-27")  # last group matchday (inclusive)
TOURNAMENT_START = pd.Timestamp("2026-06-11")


def wc2026_fixtures(results: pd.DataFrame) -> pd.DataFrame:
    """All 2026 FIFA World Cup fixtures present in the dataset."""
    m = (
        (results["tournament"] == "FIFA World Cup")
        & (results["date"] >= TOURNAMENT_START)
        & (results["date"] <= pd.Timestamp("2026-07-19"))
    )
    return results[m].copy()


def wc2026_group_fixtures(results: pd.DataFrame) -> pd.DataFrame:
    fx = wc2026_fixtures(results)
    return fx[fx["date"] <= GROUP_STAGE_END].copy()


# One distinctive anchor team per official group, from the FIFA draw
# (Washington DC, 2025-12-05). Group composition is still reconstructed
# algorithmically; the anchors only pin the official letter to each
# component. Kickoff order is NOT a reliable label: it would swap C and D.
OFFICIAL_GROUP_ANCHORS = {
    "A": "Mexico",
    "B": "Canada",
    "C": "Brazil",
    "D": "United States",
    "E": "Germany",
    "F": "Netherlands",
    "G": "Belgium",
    "H": "Spain",
    "I": "France",
    "J": "Argentina",
    "K": "Portugal",
    "L": "England",
}


def reconstruct_groups(group_fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Recover the 12 groups as connected components of the fixture graph.

    Within the group stage, teams only play opponents from their own group,
    so the 'played-against' graph has exactly 12 components of 4 teams.
    Each component gets its official FIFA letter via OFFICIAL_GROUP_ANCHORS.

    Raises if the reconstruction does not yield 12 clean groups of 4, or if
    the anchors do not map 1:1 onto components — canaries against data
    problems.
    """
    parent: dict[str, str] = {}

    def find(t: str) -> str:
        parent.setdefault(t, t)
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in group_fixtures.itertuples(index=False):
        union(r.home_team, r.away_team)

    comps: dict[str, set[str]] = {}
    for team in parent:
        comps.setdefault(find(team), set()).add(team)

    groups = list(comps.values())
    if len(groups) != 12 or any(len(g) != 4 for g in groups):
        raise ValueError(
            f"Group reconstruction failed: {[len(g) for g in groups]} — check raw data."
        )
    labelled: dict[str, list[str]] = {}
    for letter, anchor in OFFICIAL_GROUP_ANCHORS.items():
        hits = [g for g in groups if anchor in g]
        if len(hits) != 1:
            raise ValueError(f"Anchor {anchor!r} (group {letter}) matched {len(hits)} components.")
        labelled[letter] = sorted(hits.pop())
    covered = {t for g in labelled.values() for t in g}
    if len(covered) != 48:
        raise ValueError("Anchors did not cover all 12 components — check raw data.")
    return labelled


# FIFA group-stage tiebreakers (Art. 13): points, goal difference, goals
# scored, head-to-head (points, GD, goals among tied teams), fair play,
# drawing of lots. We implement through head-to-head; fair-play points are
# unobservable pre-match, so ties surviving H2H are broken at random in the
# simulation — equivalent to lots in expectation.
TIEBREAKER_DOC = "points > gd > gf > h2h(points, gd, gf) > random (proxy for fair play/lots)"
