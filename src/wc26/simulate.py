"""Monte Carlo simulation of the 2026 group stage.

Every run: sample all 72 group fixtures from the Dixon-Coles score matrix,
build group tables with FIFA tiebreakers, rank third-placed teams across
groups, mark the 24 + 8 qualifiers. Aggregating runs yields empirical
probabilities for: group position distribution, qualification, and
most-goals-in-group-stage.

Knockout bracket (R32 -> final): NOT implemented yet, see TODO at bottom.

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


def simulate_group_stage(
    groups: dict[str, list[str]],
    fixtures: pd.DataFrame,
    model,
    elo: pd.Series,
    n_sims: int = 20_000,
    seed: int = 26,
    fixed_results: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the Monte Carlo. Returns aggregated probability tables.

    fixed_results: {(home, away): (hg, ag)} for matches already played.
    """
    rng = np.random.default_rng(seed)
    fixed_results = fixed_results or {}
    team_group = {t: g for g, ts in groups.items() for t in ts}
    teams = sorted(team_group)

    # Precompute the score matrix per fixture (lambdas are sim-invariant).
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

    pos_counts = {t: np.zeros(4) for t in teams}
    qual_counts = {t: 0 for t in teams}
    topscoring_counts = {t: 0 for t in teams}
    G1 = len(model.score_matrix(1, 1))  # grid side (MAX_GOALS + 1)

    for _ in range(n_sims):
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

        thirds = []
        for g, ts in groups.items():
            order = _rank_group(ts, stats, h2h, rng)
            for pos, t in enumerate(order):
                pos_counts[t][pos] += 1
                if pos < 2:
                    qual_counts[t] += 1
            thirds.append(order[2])
        thirds = sorted(
            thirds, key=lambda t: (stats[t][0], stats[t][1], stats[t][2], rng.random()), reverse=True
        )
        for t in thirds[:8]:
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
# TODO(knockout): implement the R32 bracket.
# The 2026 bracket maps 1st/2nd of each group to fixed slots; the 8 best
# thirds fill slots via FIFA's allocation table, which depends on WHICH
# combination of groups the thirds come from (495 combinations -> lookup
# table published in the official regulations, Annexe C). Steps:
#   1. Transcribe the official allocation table (FIFA regulations PDF).
#   2. Propagate winners R32 -> R16 -> QF -> SF -> F; at each KO node sample
#      90' score; if draw, sample extra time (Poisson with ~1/3 intensity)
#      then shootout ~ Bernoulli(0.5) (calibrate on shootouts.csv).
#   3. Aggregate P(reach round X) and P(champion).
# --------------------------------------------------------------------------- #
