"""Step 38: per-team driver breakdown — strength vs draw luck vs tilts.

Splits each team's chance of going through the group (P of finishing top two)
into three ADDITIVE pieces, by re-running the tournament under counterfactuals:

  A  intrinsic strength : P(top 2) if the team played an AVERAGE group
                          (its three opponents replaced by field-mean-Elo sides),
                          tilts off. "How good are you, draw aside."
  B  + draw luck        : P(top 2) in the ACTUAL group, tilts off, minus A.
                          Positive = an easier-than-average group; negative = a
                          group of death.
  C  + tilts            : P(top 2) in the actual group WITH the deployed tilts
                          (fatigue + altitude), minus B. The venue/fatigue nudge.

  total = A + (B-A) + (C-B) = C  (exactly, by construction)

All three use the tournament-eve information set (no live conditioning), so this
is a clean "what drives each team's pre-tournament odds" decomposition. P(top 2)
is used rather than P_qualify because best-thirds couple groups together and would
muddy a per-team split; the group cut is the legible, decomposable quantity.

We also report each group's difficulty (mean opponent Elo, ranked) for context.
"""
import json
import sys
import time

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from wc26.data import load_results, reconstruct_groups, wc2026_group_fixtures
from wc26.dixon_coles import DixonColes
from wc26.elo import ratings_asof
from wc26.simulate import _tilted, simulate_tournament
from wc26.tilts import load_city_tilt, load_team_tilt

results = load_results()
elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
model = DixonColes().fit(df[df["date"] >= "2005-01-01"], pd.Timestamp("2026-06-11"))
elo = ratings_asof(elo_hist, "2026-06-11")
gfx = wc2026_group_fixtures(results)
groups = reconstruct_groups(gfx)
team_group = {t: g for g, ts in groups.items() for t in ts}
tilt, city_tilt = load_team_tilt(), load_city_tilt()
try:
    draws = json.load(open("outputs/dc_bootstrap.json"))
except FileNotFoundError:
    draws = None

# ---- full tournament, eve, WITH and WITHOUT tilts -> C and B (P top 2) ----
def top2(team_log_tilt, city_log_tilt):
    tbl = simulate_tournament(groups, gfx, model, elo, n_sims=20000, param_draws=draws,
                              team_log_tilt=team_log_tilt, city_log_tilt=city_log_tilt)["teams"]
    return (tbl["P1"] + tbl["P2"]).to_dict()

t0 = time.time()
C = top2(tilt, city_tilt)            # actual group + tilts
B = top2(None, None)                 # actual group, no tilts
print(f"full sims (B,C) done in {time.time()-t0:.0f}s")

# ---- intrinsic strength A: each team vs three field-mean opponents ----
field_mean = float(np.mean([elo[t] for t in team_group]))
G1 = model.score_matrix(1, 1).shape[0]
rng = np.random.default_rng(26)
N = 20000


def avg_group_top2(team, n=N):
    """P(team finishes top 2) of a group vs three identical field-mean sides."""
    # two distinct matchups: team vs avg, and avg vs avg (all neutral, no tilt)
    lh, la = model.predict_lambdas(elo[team], field_mean, neutral=True)
    M_ta = model.score_matrix(lh, la)            # team (home) vs avg
    lh2, la2 = model.predict_lambdas(field_mean, elo[team], neutral=True)
    M_at = model.score_matrix(lh2, la2)          # avg (home) vs team
    M_aa = model.score_matrix(*model.predict_lambdas(field_mean, field_mean, neutral=True))

    def sample(M, k):
        idx = rng.choice(M.size, size=k, p=M.ravel())
        return idx // G1, idx % G1

    # round robin: team plays X1,X2,X3 (alternate home/away); the X's play each other.
    pts = np.zeros((n, 4)); gd = np.zeros((n, 4)); gf = np.zeros((n, 4))
    # team = index 0; opponents 1,2,3
    def add(i, j, gi, gj):
        wi = (gi > gj).astype(int) * 3 + (gi == gj).astype(int)
        wj = (gj > gi).astype(int) * 3 + (gi == gj).astype(int)
        pts[:, i] += wi; pts[:, j] += wj
        gd[:, i] += gi - gj; gd[:, j] += gj - gi
        gf[:, i] += gi; gf[:, j] += gj
    for k, opp in enumerate([1, 2, 3]):
        if k % 2 == 0:
            g0, go = sample(M_ta, n); add(0, opp, g0, go)
        else:
            go, g0 = sample(M_at, n); add(0, opp, g0, go)
    for (i, j) in [(1, 2), (1, 3), (2, 3)]:
        gi, gj = sample(M_aa, n); add(i, j, gi, gj)
    # rank: points, gd, gf, random tiebreak; count team (col 0) in top 2
    keyarr = pts * 1e6 + gd * 1e3 + gf + rng.random((n, 4)) * 1e-3
    rank = (keyarr > keyarr[:, [0]]).sum(axis=1)  # how many teams strictly above team 0
    return float((rank < 2).mean())


t0 = time.time()
A = {t: avg_group_top2(t) for t in team_group}
print(f"intrinsic mini-sims (A) done in {time.time()-t0:.0f}s")

# ---- group difficulty: mean opponent Elo per team, group rank ----
group_mean_opp = {}
for t, g in team_group.items():
    opps = [x for x in groups[g] if x != t]
    group_mean_opp[t] = float(np.mean([elo[x] for x in opps]))
group_strength = {g: float(np.mean([elo[x] for x in ts])) for g, ts in groups.items()}
group_rank = {g: r + 1 for r, (g, _) in enumerate(
    sorted(group_strength.items(), key=lambda kv: -kv[1]))}  # 1 = hardest group

drivers = []
for t in team_group:
    a, b, c = A[t], B[t], C[t]
    drivers.append({
        "team": t, "group": team_group[t], "elo": round(float(elo[t])),
        "strength": round(a, 4),            # intrinsic (avg group, no tilt)
        "draw": round(b - a, 4),            # draw luck (actual vs avg group)
        "tilt": round(c - b, 4),            # deployed tilts
        "total": round(c, 4),               # = P(top 2), eve
        "mean_opp_elo": round(group_mean_opp[t]),
        "group_hardness_rank": group_rank[team_group[t]],  # 1..12, 1 = hardest
    })
drivers.sort(key=lambda d: -d["total"])
json.dump({"field_mean_elo": round(field_mean), "groups_by_hardness": group_rank,
           "teams": drivers}, open("outputs/team_drivers.json", "w"), indent=1)

print(f"\nfield-mean Elo {field_mean:.0f}. Sample (sorted by P top2):")
print(f"{'team':16s}{'P2':>6s} = {'strength':>9s}{'draw':>7s}{'tilt':>7s}  grp(hard#)")
for d in drivers[:8] + drivers[-4:]:
    print(f"{d['team']:16s}{d['total']*100:5.1f}% = {d['strength']*100:8.1f}%"
          f"{d['draw']*100:+6.1f}%{d['tilt']*100:+6.1f}%   {d['group']}(#{d['group_hardness_rank']})")
print("\noutputs/team_drivers.json written")
