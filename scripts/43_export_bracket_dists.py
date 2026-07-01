"""Step 43: per-tie score heatmaps for the knockout bracket.

For each knockout match in the current predicted bracket (outputs/bracket.csv:
the modal team in each slot), compute the model's expected-goals distribution for
that matchup (Dixon-Coles at the frozen 2026-06-11 ratings + the deployed fatigue
tilt, neutral venue) and its 1X2. The site folds these into the heatmap index so
clicking a bracket tie shows its goal heatmap, exactly like a group game.

Output: outputs/bracket_dists.json -> site/data.js.
"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from wc26.data import load_results, wc2026_fixtures, GROUP_STAGE_END
from wc26.dixon_coles import DixonColes
from wc26.elo import ratings_asof
from wc26.tilts import load_team_tilt

CLIP = 5


def clip_matrix(M):
    g = np.zeros((CLIP + 1, CLIP + 1))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            g[min(i, CLIP), min(j, CLIP)] += M[i, j]
    return [[round(float(x), 5) for x in row] for row in g]


results = load_results()
elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
model = DixonColes().fit(df[df["date"] >= "2005-01-01"], pd.Timestamp("2026-06-11"))
elo = ratings_asof(elo_hist, "2026-06-11")
fat = load_team_tilt() or {}

# actual results of played knockout ties, keyed by the pair, so we can grade each tie
# (the modal occupants of a played slot are the real teams once the groups are done).
_ko = wc2026_fixtures(results)
_ko = _ko[(_ko["date"] > GROUP_STAGE_END)].dropna(subset=["home_score", "away_score"])
played_ko = {frozenset((r.home_team, r.away_team)): (r.home_team, int(r.home_score), int(r.away_score))
             for r in _ko.itertuples(index=False)}

bk = pd.read_csv("outputs/bracket.csv")
out = []
for mn, g in bk.groupby("match"):
    slots = {r.slot: r.team for r in g.itertuples(index=False)}
    home, away = slots.get("top"), slots.get("bot")
    if not home or not away or home not in elo or away not in elo or home == "?" or away == "?":
        continue
    lh, la = model.predict_lambdas(elo[home], elo[away], neutral=True)
    d = float(fat.get(home, 0.0) - fat.get(away, 0.0))
    lh, la = lh * np.exp(d), la * np.exp(-d)
    grid = clip_matrix(model.score_matrix(lh, la))
    pH, pD, pA = model.outcome_probs(lh, la)
    flat = sorted(((i, j, grid[i][j]) for i in range(CLIP + 1) for j in range(CLIP + 1)),
                  key=lambda x: -x[2])[:3]
    entry = {"match": int(mn), "home": home, "away": away, "ko": True,
             "lh": round(float(lh), 2), "la": round(float(la), 2),
             "pH": round(pH, 4), "pD": round(pD, 4), "pA": round(pA, 4),
             "grid": grid, "top": [{"h": i, "a": j, "p": round(p, 4)} for i, j, p in flat]}
    pk = played_ko.get(frozenset((home, away)))
    if pk:  # real 90-minute result, oriented to this tie's home/away
        ph, hs, as_ = pk
        entry["actual"] = [hs, as_] if ph == home else [as_, hs]
    out.append(entry)

json.dump(out, open("outputs/bracket_dists.json", "w"), indent=1)
print(f"outputs/bracket_dists.json: {len(out)} bracket-tie heatmaps")
for m in out[:3]:
    print(f"  m{m['match']} {m['home']} v {m['away']}: 1X2 {m['pH']:.0%}/{m['pD']:.0%}/{m['pA']:.0%}")
