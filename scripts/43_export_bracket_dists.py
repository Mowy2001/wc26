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
from wc26.data import load_results, load_shootouts, wc2026_fixtures, wc2026_played_ko
from wc26.dixon_coles import DixonColes
from wc26.elo import ratings_asof
from wc26.simulate import (FINAL_MATCH, KO_CITY, QF_MATCHES, R16_MATCHES,
                           R32_MATCHES, SF_MATCHES)
from wc26.tilts import load_city_tilt, load_team_tilt

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
ct = load_city_tilt() or {}
VENUE = {mn: v for mn, _, _, v in
         [*R32_MATCHES, *R16_MATCHES, *QF_MATCHES, *SF_MATCHES, FINAL_MATCH]}

# The fitted goal-model parameters, exported so the site can compute the SAME
# score grid client-side for any pairing (sandbox picks the bracket export
# can't precompute). Neutral-venue KO convention, fatigue tilt included via
# the team tilt map, exactly like the grids below.
json.dump({k: round(float(v), 10) for k, v in model.params_.items()
           if isinstance(v, float)}, open("outputs/dc_params.json", "w"))

# played knockout ties -> {pair: (winner, {team: 90-min goals})}, shootouts resolved,
# so we can grade each tie (the modal occupants of a played slot are the real teams once
# the groups are done).
played_ko = wc2026_played_ko(results, load_shootouts())

# pre-match market 1X2 per pair, from the archived odds snapshots (scripts/41): the
# LAST fetch strictly before kickoff, so grading the market only uses what it said
# pre-match. Coverage starts 2026-06-29 (when the archive began) — earlier ties
# simply carry no market line.
import os
NAME = {"USA": "United States", "Korea Republic": "South Korea", "IR Iran": "Iran",
        "Czechia": "Czech Republic", "Türkiye": "Turkey"}
canon = lambda t: NAME.get(t, t.replace(" & ", " and "))
market_pre = {}
# The one played tie that predates the archive (RSA-CAN, 2026-06-28): bet365 closing
# line via Sofascore (event 12813000, "Full time" current values at kickoff:
# 15/4, 12/5, 83/100), retrieved 2026-07-03 and de-vigged proportionally like the
# archive. Single book, declared as such.
_inv = [1 / (1 + 15 / 4), 1 / (1 + 12 / 5), 1 / (1 + 83 / 100)]  # fractional -> decimal -> inverse
_s = sum(_inv)
market_pre[frozenset(("South Africa", "Canada"))] = {
    "fetched": "2026-06-28T00:00:00Z", "home": "South Africa",
    "pH": round(_inv[0] / _s, 4), "pD": round(_inv[1] / _s, 4), "pA": round(_inv[2] / _s, 4),
    "n_books": 1}
if os.path.exists("outputs/odds_history.jsonl"):
    for line in open("outputs/odds_history.jsonl"):
        snap = json.loads(line)
        for m in snap.get("matches", {}).values():
            pair = frozenset((canon(m["home"]), canon(m["away"])))
            if m.get("commence") and snap["fetched"] < m["commence"]:
                prev = market_pre.get(pair)
                if prev is None or snap["fetched"] > prev["fetched"]:
                    market_pre[pair] = {"fetched": snap["fetched"], "home": canon(m["home"]),
                                        "pH": m["pH"], "pD": m["pD"], "pA": m["pA"],
                                        "n_books": m.get("n_books")}

bk = pd.read_csv("outputs/bracket.csv")
out = []
for mn, g in bk.groupby("match"):
    slots = {r.slot: r.team for r in g.itertuples(index=False)}
    home, away = slots.get("top"), slots.get("bot")
    if not home or not away or home not in elo or away not in elo or home == "?" or away == "?":
        continue
    # same venue convention as the Monte Carlo (_ko_match_sim): the host nation,
    # if playing, takes a true home side; the altitude city tilt applies at the
    # high venues. λs are then re-oriented to this tie's top/bot for display.
    ven, city = VENUE.get(int(mn), ""), KO_CITY.get(int(mn))
    h, aw = (away, home) if away == ven else (home, away)
    lh, la = model.predict_lambdas(elo[h], elo[aw], neutral=(h != ven))
    d = float(fat.get(h, 0.0) - fat.get(aw, 0.0)
              + (ct.get((h, city), 0.0) - ct.get((aw, city), 0.0) if city else 0.0))
    lh, la = lh * np.exp(d), la * np.exp(-d)
    if h != home:
        lh, la = la, lh
    grid = clip_matrix(model.score_matrix(lh, la))
    pH, pD, pA = model.outcome_probs(lh, la)
    flat = sorted(((i, j, grid[i][j]) for i in range(CLIP + 1) for j in range(CLIP + 1)),
                  key=lambda x: -x[2])[:3]
    entry = {"match": int(mn), "home": home, "away": away, "ko": True,
             "lh": round(float(lh), 2), "la": round(float(la), 2),
             "pH": round(pH, 4), "pD": round(pD, 4), "pA": round(pA, 4),
             "grid": grid, "top": [{"h": i, "a": j, "p": round(p, 4)} for i, j, p in flat]}
    pk = played_ko.get(frozenset((home, away)))
    if pk:  # who advanced (shootouts resolved) + the 90-minute score, oriented to this tie
        winner, goals = pk
        entry["actual"] = [int(goals.get(home, 0)), int(goals.get(away, 0))]
        entry["winner"] = winner
    mkt = market_pre.get(frozenset((home, away)))
    if mkt:  # oriented to this tie's home/away
        flip = mkt["home"] != home
        entry["market"] = {"pH": mkt["pA"] if flip else mkt["pH"], "pD": mkt["pD"],
                           "pA": mkt["pH"] if flip else mkt["pA"], "n_books": mkt["n_books"]}
    out.append(entry)

# Third-place play-off (match 103): not part of the bracket tree, so it is not in
# bracket.csv, but it IS a graded knockout match. Add it from the schedule (the
# second-to-last WC fixture) so it shows up in "Called it" and "Last matches".
_wc = wc2026_fixtures(results).dropna(subset=["home_score"]).sort_values("date")
if len(_wc) >= 104:
    tp = _wc.iloc[-2]
    home, away = tp.home_team, tp.away_team
    if home in elo and away in elo:
        ven, city = VENUE.get(103, ""), KO_CITY.get(103)
        h, aw = (away, home) if away == ven else (home, away)
        lh, la = model.predict_lambdas(elo[h], elo[aw], neutral=(h != ven))
        d = float(fat.get(h, 0.0) - fat.get(aw, 0.0)
                  + (ct.get((h, city), 0.0) - ct.get((aw, city), 0.0) if city else 0.0))
        lh, la = lh * np.exp(d), la * np.exp(-d)
        if h != home:
            lh, la = la, lh
        grid = clip_matrix(model.score_matrix(lh, la))
        pH, pD, pA = model.outcome_probs(lh, la)
        flat = sorted(((i, j, grid[i][j]) for i in range(CLIP + 1) for j in range(CLIP + 1)),
                      key=lambda x: -x[2])[:3]
        e103 = {"match": 103, "home": home, "away": away, "ko": True,
                "lh": round(float(lh), 2), "la": round(float(la), 2),
                "pH": round(pH, 4), "pD": round(pD, 4), "pA": round(pA, 4),
                "grid": grid, "top": [{"h": i, "a": j, "p": round(p, 4)} for i, j, p in flat]}
        pk = played_ko.get(frozenset((home, away)))
        if pk:
            winner, goals = pk
            e103["actual"] = [int(goals.get(home, 0)), int(goals.get(away, 0))]
            e103["winner"] = winner
        mkt = market_pre.get(frozenset((home, away)))
        if mkt:
            flip = mkt["home"] != home
            e103["market"] = {"pH": mkt["pA"] if flip else mkt["pH"], "pD": mkt["pD"],
                              "pA": mkt["pH"] if flip else mkt["pA"], "n_books": mkt["n_books"]}
        out.append(e103)

json.dump(out, open("outputs/bracket_dists.json", "w"), indent=1)
print(f"outputs/bracket_dists.json: {len(out)} bracket-tie heatmaps")
for m in out[:3]:
    print(f"  m{m['match']} {m['home']} v {m['away']}: 1X2 {m['pH']:.0%}/{m['pD']:.0%}/{m['pA']:.0%}")
