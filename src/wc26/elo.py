"""Elo ratings for national teams, computed from scratch.

Replicates the World Football Elo Ratings methodology (eloratings.net),
so we own the rating pipeline end-to-end instead of scraping a third party.

Update rule
-----------
    R_new = R_old + K * G * (W - W_e)

where:
    K   : match importance weight (World Cup final tournament = 60,
          continental finals = 50, qualifiers = 40, other tournaments = 30,
          friendlies = 20)
    G   : goal-difference multiplier
          (1 if margin <= 1; 1.5 if margin == 2; (11 + margin) / 8 if >= 3)
    W   : actual result (1 win / 0.5 draw / 0 loss)
    W_e : expected result, 1 / (1 + 10 ** (-dr / 400)) with
          dr = R_home + HOME_ADV * (not neutral) - R_away

Design notes
------------
* New teams enter at 1500. Ratings are NOT zero-sum-corrected; this matches
  the reference implementation.
* HOME_ADV = 100 Elo points, the standard value in the reference system.
  This is a *rating-update* convention; the predictive home advantage for
  the match model is estimated separately in `dixon_coles.py`.
* We deliberately compute ratings ourselves: it lets us backtest with
  point-in-time ratings (no look-ahead bias) at any historical date.
"""

from __future__ import annotations

import pandas as pd

HOME_ADV = 100.0
INITIAL_RATING = 1500.0

# Match importance -> K factor. Keys are matched as substrings of the
# `tournament` column (lowercased), first hit wins, ordered by specificity.
K_RULES: list[tuple[str, float]] = [
    ("fifa world cup qualification", 40.0),
    ("fifa world cup", 60.0),
    ("uefa euro qualification", 40.0),
    ("copa américa qualification", 40.0),
    ("uefa euro", 50.0),
    ("copa américa", 50.0),
    ("african cup of nations qualification", 40.0),
    ("african cup of nations", 50.0),
    ("afc asian cup qualification", 40.0),
    ("afc asian cup", 50.0),
    ("concacaf championship qualification", 40.0),
    ("gold cup qualification", 40.0),
    ("concacaf championship", 50.0),
    ("gold cup", 50.0),
    ("confederations cup", 50.0),
    ("uefa nations league", 40.0),
    ("concacaf nations league", 40.0),
    ("friendly", 20.0),
]
K_DEFAULT = 30.0  # any other competitive tournament


def k_factor(tournament: str) -> float:
    """Map a tournament name to its Elo K factor."""
    t = str(tournament).lower()
    for key, k in K_RULES:
        if key in t:
            return k
    return K_DEFAULT


def goal_multiplier(margin: int) -> float:
    """Goal-difference multiplier G (rewards convincing wins, capped growth)."""
    margin = abs(int(margin))
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def expected_score(r_home: float, r_away: float, neutral: bool) -> float:
    """Expected result for the home side (logistic in rating difference)."""
    dr = r_home - r_away + (0.0 if neutral else HOME_ADV)
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def compute_elo_history(results: pd.DataFrame) -> pd.DataFrame:
    """Run Elo over the full match history.

    Parameters
    ----------
    results : DataFrame with columns
        date (datetime), home_team, away_team, home_score, away_score,
        tournament, neutral (bool). Matches with missing scores (future
        fixtures) are skipped.

    Returns
    -------
    DataFrame: one row per played match with pre-match ratings
        (elo_home_pre, elo_away_pre) and post-match ratings. Pre-match
        ratings are the point-in-time features to feed the match model —
        using post-match ratings would leak the result.
    """
    df = results.dropna(subset=["home_score", "away_score"]).sort_values("date")
    ratings: dict[str, float] = {}
    rows = []
    for row in df.itertuples(index=False):
        rh = ratings.get(row.home_team, INITIAL_RATING)
        ra = ratings.get(row.away_team, INITIAL_RATING)
        we = expected_score(rh, ra, bool(row.neutral))
        margin = int(row.home_score) - int(row.away_score)
        w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        delta = k_factor(row.tournament) * goal_multiplier(margin) * (w - we)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
        rows.append(
            {
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "elo_home_pre": rh,
                "elo_away_pre": ra,
                "elo_home_post": rh + delta,
                "elo_away_post": ra - delta,
            }
        )
    return pd.DataFrame(rows)


def ratings_asof(elo_history: pd.DataFrame, date: str | pd.Timestamp) -> pd.Series:
    """Latest rating of every team strictly *before* `date` (no look-ahead)."""
    cutoff = pd.Timestamp(date)
    hist = elo_history[elo_history["date"] < cutoff]
    home = hist[["date", "home_team", "elo_home_post"]].rename(
        columns={"home_team": "team", "elo_home_post": "elo"}
    )
    away = hist[["date", "away_team", "elo_away_post"]].rename(
        columns={"away_team": "team", "elo_away_post": "elo"}
    )
    both = pd.concat([home, away]).sort_values("date")
    return both.groupby("team")["elo"].last()
