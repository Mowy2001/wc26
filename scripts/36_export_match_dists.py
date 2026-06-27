"""Step 36: per-match score distributions for the site heatmaps.

For every group-stage fixture we export the model's full pre-match scoreline
distribution — the Dixon-Coles score matrix at the frozen 2026-06-11 ratings
with the deployed tilts (fatigue + altitude) and the venue city. This is the
"most probable result per match" view: a goals-home x goals-away heatmap, the
modal scoreline, and the W/D/L split. For matches already played we also carry
the actual scoreline so the site can drop a marker on the predicted grid.

Point-in-time: ratings and parameters are the tournament-eve baseline, so the
heatmap is the genuine pre-match forecast, never informed by the result it is
being checked against.
"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from wc26.data import load_results, reconstruct_groups, wc2026_group_fixtures
from wc26.dixon_coles import DixonColes
from wc26.elo import ratings_asof
from wc26.simulate import _tilted
from wc26.tilts import load_city_tilt, load_team_tilt

CLIP = 5  # display grid is 0..5 goals; the last bucket is "5+" (mass-conserving)

results = load_results()
elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
model = DixonColes().fit(df[df["date"] >= "2005-01-01"], pd.Timestamp("2026-06-11"))
elo = ratings_asof(elo_hist, "2026-06-11")
tilt, city_tilt = load_team_tilt(), load_city_tilt()

gfx = wc2026_group_fixtures(results)
team_group = {t: g for g, ts in reconstruct_groups(gfx).items() for t in ts}


def clip_matrix(M: np.ndarray) -> list[list[float]]:
    """Collapse the (MAX_GOALS+1)^2 matrix to (CLIP+1)^2 with a 5+ bucket."""
    g = np.zeros((CLIP + 1, CLIP + 1))
    for i in range(M.shape[0]):
        ci = min(i, CLIP)
        for j in range(M.shape[1]):
            g[ci, min(j, CLIP)] += M[i, j]
    return [[round(float(x), 5) for x in row] for row in g]


matches = []
for r in gfx.itertuples(index=False):
    lh, la = model.predict_lambdas(elo[r.home_team], elo[r.away_team], neutral=bool(r.neutral))
    lh, la = _tilted(lh, la, r.home_team, r.away_team, tilt, getattr(r, "city", None), city_tilt)
    M = model.score_matrix(lh, la)
    pH, pD, pA = model.outcome_probs(lh, la)
    grid = clip_matrix(M)
    flat = [(i, j, grid[i][j]) for i in range(CLIP + 1) for j in range(CLIP + 1)]
    top = sorted(flat, key=lambda x: -x[2])[:3]
    played = not (pd.isna(r.home_score) or pd.isna(r.away_score))
    matches.append({
        "date": str(pd.Timestamp(r.date).date()),
        "home": r.home_team, "away": r.away_team, "group": team_group.get(r.home_team, "?"),
        "city": getattr(r, "city", None), "neutral": bool(r.neutral),
        "lh": round(float(lh), 3), "la": round(float(la), 3),
        "pH": round(pH, 4), "pD": round(pD, 4), "pA": round(pA, 4),
        "grid": grid,  # grid[home_goals][away_goals]
        "top": [{"h": i, "a": j, "p": round(p, 4)} for i, j, p in top],
        "actual": ([int(r.home_score), int(r.away_score)] if played else None),
    })

json.dump(matches, open("outputs/match_dists.json", "w"), indent=1)
played_n = sum(m["actual"] is not None for m in matches)
print(f"outputs/match_dists.json: {len(matches)} group fixtures, {played_n} played")
# Sanity: pick a played match and show modal vs actual
for m in matches:
    if m["actual"] is not None:
        t = m["top"][0]
        print(f"  e.g. {m['home']} v {m['away']}: modal {t['h']}-{t['a']} ({t['p']:.0%}), "
              f"actual {m['actual'][0]}-{m['actual'][1]}, W/D/L {m['pH']:.0%}/{m['pD']:.0%}/{m['pA']:.0%}")
        break
