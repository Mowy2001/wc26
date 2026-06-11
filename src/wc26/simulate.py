"""Monte Carlo simulation of the 2026 tournament (groups + knockout).

Group stage: sample all 72 fixtures from the Dixon-Coles score matrix,
build group tables with FIFA tiebreakers, rank third-placed teams across
groups, mark the 24 + 8 qualifiers.

Knockout: official R32 bracket (matches 73-104; the third-place play-off,
match 103, is irrelevant to every reported probability and is skipped).
Third-place slots are filled by a deterministic perfect matching between the
8 qualified thirds and the slot constraints published in the schedule
(3rd A/B/C/D/F etc.). FIFA's regulations fix one assignment per combination
in a 495-row annex; absent that literal table we compute a matching that is
always consistent with the published slot constraints — see METHODOLOGY.md.

Knockout match model: 90' score from the DC matrix (host nations get the
home effect when playing in their own country, e.g. Mexico in matches at
Mexico City); if level, extra time as independent Poisson with lambda/3
(30' at 90' scoring intensity); if still level, a shootout with
P(win) = sigmoid(B_HOME*host + B_ELO*elodiff/400), coefficients fitted on
the 677 historical shootouts in shootouts.csv.

Tiebreakers implemented (FIFA Art. 13): points > GD > GF > head-to-head
(points, GD, GF among tied teams) > random (proxy for fair-play/lots,
unobservable pre-match). Best-thirds ranking: points > GD > GF > random.

Note on already-played matches: pass `fixed_results` to condition the
simulation on real results as the group stage progresses — the simulator
then only samples the remaining fixtures (live-updating forecasts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_group(teams: list[str], stats: dict, h2h: dict, rng: np.random.Generator) -> list[str]:
    """Order 4 teams by FIFA tiebreakers. stats[t] = [pts, gd, gf]."""

    def sort_block(block: list[str]) -> list[str]:
        block = sorted(block, key=lambda t: (stats[t][0], stats[t][1], stats[t][2], rng.random()), reverse=True)
        # head-to-head among fully tied subsets on (pts, gd, gf)
        out, i = [], 0
        while i < len(block):
            j = i + 1
            while j < len(block) and stats[block[j]][:3] == stats[block[i]][:3]:
                j += 1
            tied = block[i:j]
            if len(tied) > 1:
                sub = {t: [0, 0, 0] for t in tied}
                for a in tied:
                    for b in tied:
                        if (a, b) in h2h:
                            ga, gb = h2h[(a, b)]
                            sub[a][1] += ga - gb
                            sub[a][2] += ga
                            sub[a][0] += 3 if ga > gb else (1 if ga == gb else 0)
                tied = sorted(tied, key=lambda t: (sub[t][0], sub[t][1], sub[t][2], rng.random()), reverse=True)
            out.extend(tied)
            i = j
        return out

    return sort_block(list(teams))


def _precompute_group_fixtures(fixtures, model, elo, fixed_results):
    """Per fixture: (home, away, score_matrix or None, fixed score or None)."""
    fx = []
    for r in fixtures.itertuples(index=False):
        pre = fixed_results.get((r.home_team, r.away_team))
        M = None
        if pre is None:
            lh, la = model.predict_lambdas(
                elo[r.home_team], elo[r.away_team], neutral=bool(r.neutral)
            )
            M = model.score_matrix(lh, la)
        fx.append((r.home_team, r.away_team, M, pre))
    return fx


def _play_group_stage(fx, groups, teams, G1, rng):
    """One group-stage realisation.

    Returns (order_by_group, qualified_thirds, stats, goals): final order of
    each group, the 8 best thirds (list of teams), per-team [pts, gd, gf]
    and per-team goals scored.
    """
    stats = {t: [0, 0, 0] for t in teams}
    goals = {t: 0 for t in teams}
    h2h: dict[tuple[str, str], tuple[int, int]] = {}
    for home, away, M, pre in fx:
        if pre is not None:
            hg, ag = pre
        else:
            idx = rng.choice(M.size, p=M.ravel())
            hg, ag = int(idx // G1), int(idx % G1)
        h2h[(home, away)] = (hg, ag)
        h2h[(away, home)] = (ag, hg)
        goals[home] += hg
        goals[away] += ag
        for t, gf, ga in ((home, hg, ag), (away, ag, hg)):
            stats[t][0] += 3 if gf > ga else (1 if gf == ga else 0)
            stats[t][1] += gf - ga
            stats[t][2] += gf

    order_by_group = {g: _rank_group(ts, stats, h2h, rng) for g, ts in groups.items()}
    thirds = sorted(
        (order[2] for order in order_by_group.values()),
        key=lambda t: (stats[t][0], stats[t][1], stats[t][2], rng.random()),
        reverse=True,
    )
    return order_by_group, thirds[:8], stats, goals


def simulate_group_stage(
    groups: dict[str, list[str]],
    fixtures: pd.DataFrame,
    model,
    elo: pd.Series,
    n_sims: int = 20_000,
    seed: int = 26,
    fixed_results: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the group-stage Monte Carlo. Returns aggregated probability tables.

    fixed_results: {(home, away): (hg, ag)} for matches already played.
    """
    rng = np.random.default_rng(seed)
    team_group = {t: g for g, ts in groups.items() for t in ts}
    teams = sorted(team_group)
    fx = _precompute_group_fixtures(fixtures, model, elo, fixed_results or {})
    G1 = len(model.score_matrix(1, 1))  # grid side (MAX_GOALS + 1)

    pos_counts = {t: np.zeros(4) for t in teams}
    qual_counts = {t: 0 for t in teams}
    topscoring_counts = {t: 0 for t in teams}

    for _ in range(n_sims):
        order_by_group, best_thirds, stats, goals = _play_group_stage(fx, groups, teams, G1, rng)
        for order in order_by_group.values():
            for pos, t in enumerate(order):
                pos_counts[t][pos] += 1
                if pos < 2:
                    qual_counts[t] += 1
        for t in best_thirds:
            qual_counts[t] += 1
        topscoring_counts[max(teams, key=lambda t: (goals[t], rng.random()))] += 1

    pos = pd.DataFrame(
        {t: pos_counts[t] / n_sims for t in teams}, index=["P1", "P2", "P3", "P4"]
    ).T
    pos["group"] = pos.index.map(team_group)
    pos["P_qualify"] = pd.Series(qual_counts) / n_sims
    pos["P_top_scoring_team"] = pd.Series(topscoring_counts) / n_sims
    return {"groups": pos.sort_values(["group", "P1"], ascending=[True, False])}


# --------------------------------------------------------------------------- #
# Knockout bracket (official schedule, matches 73-104)
# --------------------------------------------------------------------------- #
# R32 slots: (match, side A spec, side B spec, venue country).
# Specs: ("W", g) = group g winner, ("RU", g) = runner-up,
#        ("3RD", "ABCDF") = best third from one of those groups.
R32_MATCHES: list[tuple[int, tuple, tuple, str]] = [
    (73, ("RU", "A"), ("RU", "B"), "United States"),
    (74, ("W", "E"), ("3RD", "ABCDF"), "United States"),
    (75, ("W", "F"), ("RU", "C"), "Mexico"),
    (76, ("W", "C"), ("RU", "F"), "United States"),
    (77, ("W", "I"), ("3RD", "CDFGH"), "United States"),
    (78, ("RU", "E"), ("RU", "I"), "United States"),
    (79, ("W", "A"), ("3RD", "CEFHI"), "Mexico"),
    (80, ("W", "L"), ("3RD", "EHIJK"), "United States"),
    (81, ("W", "D"), ("3RD", "BEFIJ"), "United States"),
    (82, ("W", "G"), ("3RD", "AEHIJ"), "United States"),
    (83, ("RU", "K"), ("RU", "L"), "Canada"),
    (84, ("W", "H"), ("RU", "J"), "United States"),
    (85, ("W", "B"), ("3RD", "EFGIJ"), "Canada"),
    (86, ("W", "J"), ("RU", "H"), "United States"),
    (87, ("W", "K"), ("3RD", "DEIJL"), "United States"),
    (88, ("RU", "D"), ("RU", "G"), "United States"),
]
# Later rounds: (match, feeder A, feeder B, venue country).
R16_MATCHES = [
    (89, 74, 77, "United States"), (90, 73, 75, "United States"),
    (91, 76, 78, "United States"), (92, 79, 80, "Mexico"),
    (93, 83, 84, "United States"), (94, 81, 82, "United States"),
    (95, 86, 88, "United States"), (96, 85, 87, "Canada"),
]
QF_MATCHES = [
    (97, 89, 90, "United States"), (98, 93, 94, "United States"),
    (99, 91, 92, "United States"), (100, 95, 96, "United States"),
]
SF_MATCHES = [(101, 97, 98, "United States"), (102, 99, 100, "United States")]
FINAL_MATCH = (104, 101, 102, "United States")  # 103 = third-place play-off

# Shootout model, fitted on the 677 shootouts in shootouts.csv merged with
# point-in-time Elo: P(A wins) = sigmoid(B_HOME*host_A + B_ELO*(eloA-eloB)/400).
# LL -462.5 vs -469.3 for a pure coin flip; both terms ~2.5 sigma.
SHOOTOUT_B_HOME = 0.309
SHOOTOUT_B_ELO = 0.677

_THIRDS_SLOTS = [(m, set(spec[1])) for m, _, spec, _ in R32_MATCHES if spec[0] == "3RD"]


def allocate_thirds(qualified: frozenset, _cache: dict = {}) -> dict[int, str]:
    """Assign the 8 qualified third-place groups to R32 slots.

    Deterministic perfect matching against the slot constraints published in
    the official schedule (most-constrained slot first, candidates in
    alphabetical order). FIFA's regulations fix one assignment per 495-row
    annex table; any assignment we produce satisfies the same published
    constraints. Raises if no perfect matching exists (cannot happen if the
    constraints are FIFA's).
    """
    if qualified in _cache:
        return _cache[qualified]

    def solve(slots: list[tuple[int, set]], avail: set) -> dict[int, str] | None:
        if not slots:
            return {}
        slots = sorted(slots, key=lambda s: (len(s[1] & avail), s[0]))
        match, allowed = slots[0]
        for g in sorted(allowed & avail):
            rest = solve(slots[1:], avail - {g})
            if rest is not None:
                rest[match] = g
                return rest
        return None

    sol = solve(_THIRDS_SLOTS, set(qualified))
    if sol is None:
        raise ValueError(f"No third-place allocation for {sorted(qualified)}")
    _cache[qualified] = sol
    return sol


def _ko_match(a: str, b: str, venue: str, model, elo, rng, cache: dict) -> str:
    """Play one knockout tie, return the winner.

    90' from the cached DC matrix (the venue's host nation, if playing,
    takes the home side); extra time = independent Poisson at 1/3 of the
    90' lambdas; then the calibrated shootout model.
    """
    home, away = (b, a) if b == venue else (a, b)
    is_home = home == venue
    key = (home, away, is_home)
    M = cache.get(key)
    if M is None:
        lh, la = model.predict_lambdas(elo[home], elo[away], neutral=not is_home)
        M = (model.score_matrix(lh, la), lh, la)
        cache[key] = M
    Mat, lh, la = M
    G1 = Mat.shape[0]
    idx = rng.choice(Mat.size, p=Mat.ravel())
    hg, ag = idx // G1, idx % G1
    if hg != ag:
        return home if hg > ag else away
    et_h, et_a = rng.poisson(lh / 3.0), rng.poisson(la / 3.0)
    if et_h != et_a:
        return home if et_h > et_a else away
    z = SHOOTOUT_B_HOME * float(is_home) + SHOOTOUT_B_ELO * (elo[home] - elo[away]) / 400.0
    return home if rng.random() < 1.0 / (1.0 + np.exp(-z)) else away


def simulate_tournament(
    groups: dict[str, list[str]],
    fixtures: pd.DataFrame,
    model,
    elo: pd.Series,
    n_sims: int = 20_000,
    seed: int = 26,
    fixed_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    host_advantage: bool = True,
) -> dict[str, pd.DataFrame]:
    """Full-tournament Monte Carlo: groups + R32 bracket through the final.

    Returns {"teams": DataFrame} with group-position probabilities plus
    P_R32, P_R16, P_QF, P_SF, P_final, P_champion (and P_top_scoring_team
    over group-stage goals, kept for continuity with simulate_group_stage).

    host_advantage=False is a counterfactual knob (ablation studies): every
    match, group or knockout, is played as if on neutral ground.
    """
    rng = np.random.default_rng(seed)
    team_group = {t: g for g, ts in groups.items() for t in ts}
    teams = sorted(team_group)
    if not host_advantage:
        fixtures = fixtures.assign(neutral=True)
    fx = _precompute_group_fixtures(fixtures, model, elo, fixed_results or {})
    G1 = len(model.score_matrix(1, 1))
    ko_cache: dict = {}

    pos_counts = {t: np.zeros(4) for t in teams}
    round_counts = {t: np.zeros(6) for t in teams}  # R32, R16, QF, SF, F, champion
    topscoring_counts = {t: 0 for t in teams}

    for _ in range(n_sims):
        order_by_group, best_thirds, stats, goals = _play_group_stage(fx, groups, teams, G1, rng)
        for order in order_by_group.values():
            for pos, t in enumerate(order):
                pos_counts[t][pos] += 1
        topscoring_counts[max(teams, key=lambda t: (goals[t], rng.random()))] += 1

        third_group = {team_group[t]: t for t in best_thirds}
        alloc = allocate_thirds(frozenset(third_group))

        def resolve(spec: tuple, match: int) -> str:
            kind, g = spec
            if kind == "W":
                return order_by_group[g][0]
            if kind == "RU":
                return order_by_group[g][1]
            return third_group[alloc[match]]

        winners: dict[int, str] = {}
        for m, sa, sb, venue in R32_MATCHES:
            if not host_advantage:
                venue = ""
            a, b = resolve(sa, m), resolve(sb, m)
            round_counts[a][0] += 1
            round_counts[b][0] += 1
            winners[m] = _ko_match(a, b, venue, model, elo, rng, ko_cache)
        for depth, matches in enumerate([R16_MATCHES, QF_MATCHES, SF_MATCHES, [FINAL_MATCH]], start=1):
            for m, fa, fb, venue in matches:
                if not host_advantage:
                    venue = ""
                a, b = winners[fa], winners[fb]
                round_counts[a][depth] += 1
                round_counts[b][depth] += 1
                winners[m] = _ko_match(a, b, venue, model, elo, rng, ko_cache)
        round_counts[winners[FINAL_MATCH[0]]][5] += 1

    out = pd.DataFrame(
        {t: pos_counts[t] / n_sims for t in teams}, index=["P1", "P2", "P3", "P4"]
    ).T
    out["group"] = out.index.map(team_group)
    out["P_qualify"] = pd.Series({t: round_counts[t][0] / n_sims for t in teams})
    rounds = pd.DataFrame(
        {t: round_counts[t] / n_sims for t in teams},
        index=["P_R32", "P_R16", "P_QF", "P_SF", "P_final", "P_champion"],
    ).T
    out = out.join(rounds)
    out["P_top_scoring_team"] = pd.Series(topscoring_counts) / n_sims
    return {"teams": out.sort_values(["group", "P1"], ascending=[True, False])}
