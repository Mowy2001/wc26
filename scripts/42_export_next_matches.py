"""Step 42: next-matches feed — the upcoming fixtures, model vs market.

Source of fixtures: the latest market-odds snapshot (outputs/odds_history.jsonl),
which knows the REAL upcoming games (our results.csv can lag behind the live
tournament). For each, compute the model's outcome probabilities at the frozen
2026-06-11 ratings + deployed fatigue tilt (neutral venue), and pair them with
the de-vigged market 1X2 from the same snapshot. Knockout ties (teams from
different groups) are flagged so the site can frame them as 1-2 (advance).

Output: outputs/next_matches.json -> site/data.js (the "next matches", shown
first on the site, each with a model-vs-market 1X2 and a click-to-heatmap).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from wc26.data import load_results, reconstruct_groups, wc2026_fixtures, wc2026_group_fixtures
from wc26.dixon_coles import DixonColes
from wc26.elo import ratings_asof
from wc26.tilts import load_city_tilt, load_team_tilt

NAME = {"USA": "United States", "Korea Republic": "South Korea", "IR Iran": "Iran",
        "Czechia": "Czech Republic", "Türkiye": "Turkey"}


def canon(t):
    return NAME.get(t, t)


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
ct = load_city_tilt() or {}
team_group = {t: g for g, ts in reconstruct_groups(wc2026_group_fixtures(results)).items() for t in ts}
# real venue per fixture (schedule): hosts take a true home side, and the
# altitude tilt bites at the high venues (Mexico City / Zapopan) — a neutral
# simplification here would misquote the deployed model (it undersold Mexico
# for the R16 tie at 2,240 m).
VEN = {}
for r in wc2026_fixtures(results).itertuples(index=False):
    VEN[(r.home_team, r.away_team)] = (r.city, bool(r.neutral))

CLIP = 5


def clip_matrix(M):
    g = np.zeros((CLIP + 1, CLIP + 1))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            g[min(i, CLIP), min(j, CLIP)] += M[i, j]
    return [[round(float(x), 5) for x in row] for row in g]


snaps = [json.loads(l) for l in Path("outputs/odds_history.jsonl").read_text().splitlines() if l.strip()]
latest = snaps[-1]

out = []
for k, mk in latest["matches"].items():
    home, away = canon(mk["home"]), canon(mk["away"])
    if home not in elo or away not in elo:
        continue
    # model 1X2 with the deployed tilts at the REAL venue (schedule): home side
    # for hosts, fatigue for both, altitude city tilt where it applies.
    if (home, away) in VEN:
        city, neutral = VEN[(home, away)]
    elif (away, home) in VEN:
        (city, neutral), (home, away) = VEN[(away, home)], (away, home)
    else:
        city, neutral = None, True
    lh, la = model.predict_lambdas(elo[home], elo[away], neutral=bool(neutral))
    d = float(fat.get(home, 0.0) - fat.get(away, 0.0)
              + (ct.get((home, city), 0.0) - ct.get((away, city), 0.0) if city else 0.0))
    lh, la = lh * np.exp(d), la * np.exp(-d)
    pH, pD, pA = model.outcome_probs(lh, la)
    ko = team_group.get(home) != team_group.get(away)  # different groups -> knockout tie
    M = model.score_matrix(lh, la)
    grid = clip_matrix(M)
    flat = sorted(((i, j, grid[i][j]) for i in range(CLIP + 1) for j in range(CLIP + 1)),
                  key=lambda x: -x[2])[:3]
    out.append({
        "home": home, "away": away, "commence": mk["commence"], "ko": bool(ko),
        "model": {"pH": round(pH, 4), "pD": round(pD, 4), "pA": round(pA, 4),
                  "lh": round(float(lh), 2), "la": round(float(la), 2)},
        "market": {"pH": round(mk["pH"], 4), "pD": round(mk["pD"], 4), "pA": round(mk["pA"], 4),
                   "n_books": mk["n_books"]},
        "grid": grid, "top": [{"h": i, "a": j, "p": round(p, 4)} for i, j, p in flat],
    })

out.sort(key=lambda m: m["commence"])
json.dump({"as_of": latest["fetched"], "matches": out},
          open("outputs/next_matches.json", "w"), indent=1)
print(f"outputs/next_matches.json: {len(out)} upcoming matches (as of {latest['fetched']})")
for m in out[:5]:
    md, mb = m["model"], m["market"]
    tag = "KO" if m["ko"] else "grp"
    print(f"  [{tag}] {m['home']} v {m['away']}: model {md['pH']:.0%}/{md['pD']:.0%}/{md['pA']:.0%}"
          f"  market {mb['pH']:.0%}/{mb['pD']:.0%}/{mb['pA']:.0%}")
