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

import unicodedata

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 900
ACTIVE_WINDOW_DAYS = 900
DEBUTANT_KEY = "__new_faces__"


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


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


def squad_weights(
    goalscorers: pd.DataFrame,
    squad: pd.DataFrame,
    team: str,
    asof: pd.Timestamp,
    debutant_share: float,
    age_alpha: float = 0.0,
    drop_to_bucket: bool = True,
) -> dict[str, float]:
    """v2 allocation probabilities: official squad as the membership filter.

    Versus v1 (`scorer_weights`): a scorer keeps his decayed goal share only
    if he is on the official squad list (name-matched within team, accents
    stripped). With drop_to_bucket=True (the backtest winner's setting) the
    mass of non-selected scorers and of the age discount flows into the
    "new faces" bucket — their goals go to replacements, NOT proportionally
    to the surviving stars (renormalising concentrates the stars and scored
    worse on realised splits). Optional age discount
    exp(-age_alpha * max(0, age - 30)); age_alpha is chosen by the
    admission backtest (scripts/16), like every other knob.
    """
    sq = squad[squad["team"] == team]
    roster = {_norm_name(p): pd.Timestamp(b) for p, b in zip(sq["player"], sq["birth"])}
    g = goalscorers[
        (goalscorers["team"] == team)
        & (goalscorers["date"] < asof)
        & (~goalscorers["own_goal"].astype(bool))
    ].dropna(subset=["scorer"])
    age = (asof - g["date"]).dt.days.to_numpy(float)
    w_all = pd.Series(np.exp(-np.log(2) * age / HALF_LIFE_DAYS), index=g["scorer"]).groupby(level=0).sum()
    w = w_all[[_norm_name(p) in roster for p in w_all.index]]
    if age_alpha and len(w):
        yrs = np.array([(asof - roster[_norm_name(p)]).days / 365.25 for p in w.index])
        w = w * np.exp(-age_alpha * np.maximum(0.0, yrs - 30.0))
    if w.empty:
        return {DEBUTANT_KEY: 1.0}
    denom = float(w_all.sum()) if drop_to_bucket else float(w.sum())
    w = w / denom * (1.0 - debutant_share)
    out = w.to_dict()
    out[DEBUTANT_KEY] = 1.0 - float(w.sum())
    return out


def xg_blend_weights(
    goalscorers: pd.DataFrame,
    squad: pd.DataFrame,
    fbref: pd.DataFrame,
    team: str,
    asof: pd.Timestamp,
    debutant_share: float,
    lam: float,
    stat: str = "npg",
    age_alpha: float | None = None,
) -> dict[str, float]:
    """Blend v1 shares with club-season scoring-form shares (FBref).

    Named share_i = (1-lam) * v1_i + lam * stat_i / sum(stat over matched
    squad members); both components live inside the (1 - debutant) mass.
    `stat` is non-penalty club goals by default — the column available for
    every backtest fold (the 2017-18 Big5 page predates xG); npxG exists
    from 2021-22 on but deploying it would mean deploying what was never
    gated. Players outside the Big-5 leagues rely on the v1 component —
    coverage bias declared, same as the capital block. lam is chosen by
    the admission gate (scripts/18).
    """
    if age_alpha is None:
        base = scorer_weights(goalscorers, team, asof, debutant_share)
    else:  # admitted base: official-squad filter + age discount (scripts/16)
        base = squad_weights(goalscorers, squad, team, asof, debutant_share,
                             age_alpha=age_alpha, drop_to_bucket=False)
    deb = base.get(DEBUTANT_KEY, debutant_share)
    v1 = {k: v for k, v in base.items() if k != DEBUTANT_KEY}
    sq_names = {_norm_name(p): p for p in squad[squad["team"] == team]["player"]}
    fb = fbref[[_norm_name(p) in sq_names for p in fbref["player"]]]
    xg = {}
    for r in fb.itertuples(index=False):
        canon = sq_names[_norm_name(r.player)]
        v = getattr(r, stat)
        if pd.notna(v) and v > 0:
            xg[canon] = xg.get(canon, 0.0) + float(v)
    zx = sum(xg.values())
    zv = sum(v1.values())
    named = {}
    keys = set(v1) | set(xg)
    for k in keys:
        a = v1.get(k, 0.0) / zv if zv else 0.0
        # v1 keys are goalscorers names, xg keys squad names: merge on norm
        named[k] = (1 - lam) * a
    if zx:
        for k, v in xg.items():
            hit = next((kk for kk in named if _norm_name(kk) == _norm_name(k)), k)
            named[hit] = named.get(hit, 0.0) + lam * v / zx
    z = sum(named.values())
    if z == 0:
        return {DEBUTANT_KEY: 1.0}
    out = {k: v / z * (1.0 - deb) for k, v in named.items() if v > 0}
    out[DEBUTANT_KEY] = deb
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


def allocate_goals_live(goal_samples, weights, real_by_team, real_team_goals=None, seed=26):
    """Live Golden Boot: real goals already scored + simulated remaining.

    real_by_team: {team: {player_display: goals_scored_so_far}} from the live
    goalscorers feed (2026 WC, own goals excluded). Per simulation a team's
    REMAINING goals = goal_samples - (real so far) are allocated among its
    weighted scorers (frozen pre-tournament shares); each player's tally is
    his real goals + his simulated remaining. Real scorers absent from the
    weights (e.g. tournament debutants) are still credited their real goals,
    with no projected future (no track record -> the model doesn't extrapolate).
    Consistent with frozen beliefs: we condition on what happened, we don't
    re-estimate form.

    real_team_goals: {team: total REAL team goals} from the results table — the
    correct baseline for the remaining-goals subtraction. The scorer-credited
    sum undercounts it whenever a team goal carries no scorer credit (an
    opponent's own goal), leaving a phantom goal to re-distribute in every
    simulation even for an eliminated team (Haaland kept a 5% Golden Boot
    chance after Norway's exit that way). Falls back to the credited sum.
    """
    rng = np.random.default_rng(seed)
    n = len(goal_samples)
    win = np.zeros(n, dtype=np.int32)
    best_g = np.zeros(n, dtype=np.int32)
    best_id = np.full(n, -1, dtype=np.int32)
    ties = np.ones(n, dtype=np.int32)
    ids = {}  # (player, team) -> column id
    names = []
    egoals = []
    win_counts = {}
    for team in goal_samples.columns:
        wmap = weights.get(team, {DEBUTANT_KEY: 1.0})
        players = [p for p in wmap if p != DEBUTANT_KEY]
        probs = np.array([wmap[p] for p in players] + [wmap.get(DEBUTANT_KEY, 0.0)])
        probs = probs / probs.sum() if probs.sum() else probs
        real = {k: v for k, v in (real_by_team.get(team) or {}).items()}
        real_norm = {_norm_name(k): v for k, v in real.items()}
        team_real = max(sum(real.values()), (real_team_goals or {}).get(team, 0))
        G = goal_samples[team].to_numpy()
        rem = np.clip(G - team_real, 0, None)
        counts = np.zeros((n, len(probs)), dtype=np.int16)
        for g in np.unique(rem):
            if g == 0:
                continue
            idx = np.where(rem == g)[0]
            counts[idx] = rng.multinomial(int(g), probs, size=len(idx))
        known = counts[:, :-1]
        # union of weighted players + real scorers
        roster = list(players)
        for rp in real:
            if not any(_norm_name(rp) == _norm_name(p) for p in players):
                roster.append(rp)
        for j, p in enumerate(roster):
            col = ids.setdefault((p, team), len(names))
            if col == len(names):
                names.append((p, team)); egoals.append(0.0)
            sim = known[:, j] if j < known.shape[1] else 0
            real_credit = real_norm.get(_norm_name(p), 0)
            total = sim + real_credit
            egoals[col] = float(np.mean(total))
            better = total > best_g
            tied = (total == best_g) & (total > 0)
            ties = np.where(better, 1, ties + tied)
            take = better | (tied & (rng.random(n) < 1.0 / ties))
            best_g = np.where(take, total, best_g)
            best_id = np.where(take, col, best_id)
    for pid in best_id:
        win_counts[int(pid)] = win_counts.get(int(pid), 0) + 1
    out = pd.DataFrame({
        "player": [p for p, _ in names], "team": [t for _, t in names],
        "E_goals": egoals,
        "P_golden_boot": [win_counts.get(i, 0) / n for i in range(len(names))],
    }).sort_values("P_golden_boot", ascending=False)
    return {"players": out}
