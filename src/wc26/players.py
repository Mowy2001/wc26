"""Player layer v1: allocate simulated team goals to players (Golden Boot).

Model
-----
The tournament Monte Carlo produces, for every simulation s and team t, the
total goals G_{s,t} (groups + knockout incl. extra time). The player layer
distributes them:

    goals of players | G_{s,t} ~ Multinomial(G_{s,t}, p_t)

where p_t are per-player weights from the team's scoring history
(goalscorers.csv, point-in-time as always):

* weight_i = sum over player i's goals (own goals excluded) before the
  prediction date of exp(-ln2 * age_days / HALF_LIFE_DAYS) — recent
  scoring form matters, old goals fade (half-life ~2.5 years);
* only players with at least one goal in the last ACTIVE_WINDOW_DAYS are
  kept (a goal drought of 2.5 years in a ~10-match international year is
  our v1 squad-membership proxy);
* a per-team "new faces" bucket absorbs DEBUTANT_SHARE of the probability
  mass: historically a stable fraction of World Cup goals is scored by
  players with no prior international goal (estimated from past WCs by
  `estimate_debutant_share`). The bucket cannot win the Golden Boot —
  it exists so known stars are not over-credited.

Golden Boot: per simulation, the top scorer across all teams; ties broken
uniformly at random (the real tie-breakers — assists, minutes — are not
observable here).

v1 limitations (by design, see backlog): no xG/expected minutes (FBref
block pending), no penalty-taker bonus, squads proxied by recent scorers.
Each upgrade must earn its place like any other data block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 900
ACTIVE_WINDOW_DAYS = 900
DEBUTANT_KEY = "__new_faces__"


def estimate_debutant_share(
    goalscorers: pd.DataFrame,
    tournaments: list[tuple[str, str]] = (("2014-06-12", "2014-07-13"),
                                          ("2018-06-14", "2018-07-15"),
                                          ("2022-11-20", "2022-12-18")),
) -> float:
    """Share of WC goals scored by players with no prior international goal."""
    gs = goalscorers[~goalscorers["own_goal"].astype(bool)].dropna(subset=["scorer"])
    shares = []
    for start, end in tournaments:
        wc = gs[(gs["date"] >= start) & (gs["date"] <= end)]
        prior = gs[gs["date"] < start]
        seen = set(zip(prior["team"], prior["scorer"]))
        is_new = [(t, s) not in seen for t, s in zip(wc["team"], wc["scorer"])]
        shares.append(float(np.mean(is_new)))
    return float(np.mean(shares))


def scorer_weights(
    goalscorers: pd.DataFrame, team: str, asof: pd.Timestamp, debutant_share: float
) -> dict[str, float]:
    """Per-player allocation probabilities for one team (sums to 1)."""
    g = goalscorers[
        (goalscorers["team"] == team)
        & (goalscorers["date"] < asof)
        & (~goalscorers["own_goal"].astype(bool))
    ].dropna(subset=["scorer"])
    recent = g[g["date"] >= asof - pd.Timedelta(days=ACTIVE_WINDOW_DAYS)]
    active = set(recent["scorer"])
    g = g[g["scorer"].isin(active)]
    age = (asof - g["date"]).dt.days.to_numpy(float)
    w = pd.Series(np.exp(-np.log(2) * age / HALF_LIFE_DAYS), index=g["scorer"]).groupby(level=0).sum()
    if w.empty:  # no scoring history at all: everything to the bucket
        return {DEBUTANT_KEY: 1.0}
    w = w / w.sum() * (1.0 - debutant_share)
    out = w.to_dict()
    out[DEBUTANT_KEY] = debutant_share
    return out


def allocate_goals(
    goal_samples: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    seed: int = 26,
) -> dict[str, pd.DataFrame]:
    """Distribute simulated team goals to players, sim by sim.

    goal_samples: (n_sims x teams) tournament goals from the Monte Carlo.
    weights: {team: {player: prob}} from `scorer_weights`.

    Returns {"players": per-player table (P_golden_boot, E_goals, team),
             "distinct": per-team P(most distinct scorers, known players)}.
    """
    rng = np.random.default_rng(seed)
    n_sims = len(goal_samples)
    best_goals = np.zeros(n_sims, dtype=np.int32)
    best_ties = np.ones(n_sims, dtype=np.int32)  # reservoir counter for fair tie-breaks
    best_player = np.full(n_sims, -1, dtype=np.int32)
    player_names: list[tuple[str, str]] = []  # (player, team), index = id
    exp_goals: list[float] = []
    win_counts: dict[int, int] = {}

    distinct_best = np.zeros(n_sims, dtype=np.int32)
    distinct_ties = np.ones(n_sims, dtype=np.int32)
    distinct_team = np.full(n_sims, -1, dtype=np.int32)
    team_list = list(goal_samples.columns)

    for ti, team in enumerate(team_list):
        wmap = weights[team]
        players = [p for p in wmap if p != DEBUTANT_KEY]
        probs = np.array([wmap[p] for p in players] + [wmap.get(DEBUTANT_KEY, 0.0)])
        probs = probs / probs.sum()
        G = goal_samples[team].to_numpy()
        counts = np.zeros((n_sims, len(probs)), dtype=np.int16)
        # vectorise by grouping simulations with the same goal total
        for g in np.unique(G):
            if g == 0:
                continue
            idx = np.where(G == g)[0]
            counts[idx] = rng.multinomial(int(g), probs, size=len(idx))
        known = counts[:, :-1]

        # ---- Golden Boot reservoir update (uniform among ties) ----
        if players:
            top_g = known.max(axis=1)
            top_i = known.argmax(axis=1)  # first argmax; within-team ties are
            # rare and symmetric in expectation given exchangeable sampling
            better = top_g > best_goals
            tied = (top_g == best_goals) & (top_g > 0)
            best_ties = np.where(better, 1, best_ties + tied)
            take_tie = tied & (rng.random(n_sims) < 1.0 / best_ties)
            take = better | take_tie
            base_id = len(player_names)
            player_names += [(p, team) for p in players]
            best_goals = np.where(take, top_g, best_goals)
            best_player = np.where(take, base_id + top_i, best_player)
            exp_goals += list(known.mean(axis=0))

        # ---- most distinct (known) scorers, same reservoir scheme ----
        ndist = (known > 0).sum(axis=1)
        better = ndist > distinct_best
        tied = ndist == distinct_best
        distinct_ties = np.where(better, 1, distinct_ties + tied)
        take = better | (tied & (rng.random(n_sims) < 1.0 / distinct_ties))
        distinct_best = np.where(take, ndist, distinct_best)
        distinct_team = np.where(take, ti, distinct_team)

    for pid in best_player:
        win_counts[int(pid)] = win_counts.get(int(pid), 0) + 1

    players_tbl = pd.DataFrame(
        {
            "player": [p for p, _ in player_names],
            "team": [t for _, t in player_names],
            "E_goals": exp_goals,
            "P_golden_boot": [win_counts.get(i, 0) / n_sims for i in range(len(player_names))],
        }
    ).sort_values("P_golden_boot", ascending=False)
    distinct_tbl = pd.Series(
        {team_list[i]: float(np.mean(distinct_team == i)) for i in range(len(team_list))},
        name="P_most_distinct_scorers",
    ).sort_values(ascending=False)
    return {"players": players_tbl, "distinct": distinct_tbl.to_frame()}
