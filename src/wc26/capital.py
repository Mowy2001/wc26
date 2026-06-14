"""Football-capital feature: squad club Elo, optionally minutes-weighted.

Shared by the builder (scripts/11) and the minutes-weighting gate
(scripts/21) so both use identical club matching — the v1-vs-v2 comparison
is otherwise not apples-to-apples.

v1: squad MEAN of each player's club Elo (equal weight per player),
    unmatched/non-European clubs imputed at the snapshot's 10th percentile.
v2: the same, but each player's club Elo is weighted by his club-season
    minutes (a fringe player at a giant counts less). Players without a
    FBref minutes match fall back to the squad's mean minutes, so the
    weighting only sharpens what coverage we have rather than dropping them.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches

import pandas as pd

CLUB_ALIASES = {
    "manchester city": "man city", "manchester united": "man united",
    "paris saint germain": "paris sg", "inter milan": "inter", "internazionale": "inter",
    "bayern munich": "bayern", "atletico madrid": "atletico", "sporting cp": "sporting",
    "sporting lisbon": "sporting", "tottenham hotspur": "tottenham",
    "borussia dortmund": "dortmund", "borussia monchengladbach": "gladbach",
    "bayer leverkusen": "leverkusen", "ac milan": "milan", "as roma": "roma",
    "real betis": "betis", "athletic bilbao": "bilbao", "real sociedad": "sociedad",
    "newcastle united": "newcastle", "west ham united": "west ham",
    "wolverhampton wanderers": "wolves", "nottingham forest": "forest",
    "brighton hove albion": "brighton", "psv eindhoven": "psv",
    "rb leipzig": "leipzig", "red bull salzburg": "salzburg", "rangers": "rangers",
    "celtic": "celtic", "olympique lyonnais": "lyon", "olympique de marseille": "marseille",
    "as monaco": "monaco", "dinamo zagreb": "din zagreb", "red star belgrade": "crvena zvezda",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    drop = {"fc", "cf", "sc", "ac", "afc", "cd", "sv", "bk", "fk", "sk", "if", "ks",
            "club", "de", "futbol", "futebol", "calcio", "1", "04", "05", "09", "1899", "1900"}
    return " ".join(w for w in s.split() if w not in drop)


def _club_elo_fn(clubelo: pd.DataFrame):
    elo_by_norm: dict[str, float] = {}
    for r in clubelo.itertuples(index=False):
        elo_by_norm.setdefault(norm(r.Club), float(r.Elo))
    names = list(elo_by_norm)

    def club_elo(club: str):
        n = CLUB_ALIASES.get(norm(club), norm(club))
        if n in elo_by_norm:
            return elo_by_norm[n]
        hit = get_close_matches(n, names, n=1, cutoff=0.88)
        return elo_by_norm[hit[0]] if hit else None

    return club_elo


def capital_table(
    squads: pd.DataFrame, clubelo: pd.DataFrame, minutes: dict[str, float] | None = None
) -> pd.DataFrame:
    """Per-team capital feature. `minutes`: {norm(player): minutes} enables v2."""
    club_elo = _club_elo_fn(clubelo)
    floor = clubelo["Elo"].quantile(0.10)
    out = []
    for team, g in squads.groupby("team"):
        elos = g["club"].map(club_elo)
        matched = elos.notna()
        e = elos.fillna(floor).to_numpy(float)
        if minutes is None:
            raw = float(e.mean())
        else:
            mins = g["player"].map(lambda p: minutes.get(norm(p))).to_numpy(float)
            fallback = pd.Series(mins).mean()
            mins = pd.Series(mins).fillna(fallback).to_numpy(float)
            raw = float((e * mins).sum() / mins.sum()) if mins.sum() > 0 else float(e.mean())
        out.append({"team": team, "capital_raw": raw,
                    "coverage": float(matched.mean()), "n_players": len(g)})
    df = pd.DataFrame(out)
    df["capital_z"] = (df.capital_raw - df.capital_raw.mean()) / df.capital_raw.std(ddof=0)
    return df
