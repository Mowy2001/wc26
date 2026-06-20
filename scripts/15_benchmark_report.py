"""Step 15: live track record + real standings + three-way benchmark.

Writes two artifacts for the site:
  outputs/history/scoring.json  — how the frozen eve-of-tournament model is
      doing against real results (running log-loss vs the uniform baseline,
      favourite calibration, per-match list). Uses the DEPLOYED tilts
      (fatigue + altitude); capital was removed from the model.
  outputs/history/standings.json — the REAL group tables so far (played,
      points, GD, GF, rank), the factual record alongside the forecast.

Also prints the champion-market three-way comparison (model vs BetMGM
de-vigged with Shin vs Klement).
"""
import json, os, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes
from wc26.benchmark import shin_probs, implied_raw, log_score
from wc26.tilts import load_team_tilt, load_city_tilt

os.makedirs("outputs/history", exist_ok=True)
BETMGM = {"Spain": 450, "France": 500, "England": 700, "Brazil": 800,
          "Portugal": 900, "Argentina": 900, "Germany": 1400,
          "Netherlands": 2000, "United States": 5000}

results = load_results()
gfx = wc2026_group_fixtures(results)
groups = reconstruct_groups(gfx)
team_group = {t: g for g, ts in groups.items() for t in ts}

# ---- champion-market benchmark ----
tbl = pd.read_csv("outputs/tournament_probs_v1.csv", index_col=0)
field_mass = 1.0 - tbl.loc[list(BETMGM), "P_champion"].sum()
shin = shin_probs(BETMGM, residual_mass=float(field_mass))
print("Champion market vs model (Shin de-vig):")
for t in BETMGM:
    print(f"  {t:14s} model {tbl.loc[t,'P_champion']:5.1%}  Shin {shin[t]:5.1%}")

# ---- real group standings (the factual record) ----
played = gfx.dropna(subset=["home_score", "away_score"])
st = {t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0} for t in team_group}
for r in played.itertuples(index=False):
    hg, ag = int(r.home_score), int(r.away_score)
    for t, gf, ga in ((r.home_team, hg, ag), (r.away_team, ag, hg)):
        s = st[t]; s["P"] += 1; s["GF"] += gf; s["GA"] += ga
        s["W"] += gf > ga; s["D"] += gf == ga; s["L"] += gf < ga
        s["Pts"] += 3 if gf > ga else (1 if gf == ga else 0)
standings = {}
for g, ts in groups.items():
    rows = sorted(ts, key=lambda t: (st[t]["Pts"], st[t]["GF"] - st[t]["GA"], st[t]["GF"]), reverse=True)
    standings[g] = [{"team": t, **st[t], "GD": st[t]["GF"] - st[t]["GA"]} for t in rows]
json.dump(standings, open("outputs/history/standings.json", "w"))
print(f"\nReal standings written ({len(played)} group matches played).")

# ---- live match-level scoring (frozen beliefs, DEPLOYED tilts) ----
if played.empty:
    json.dump({"n": 0}, open("outputs/history/scoring.json", "w"))
    print("No matches played yet.")
    sys.exit(0)

elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
df = df[df["date"] < "2026-06-11"]
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
model = DixonColes().fit(df, pd.Timestamp("2026-06-11"))
elo = ratings_asof(elo_hist, "2026-06-11")
tilt, city_tilt = load_team_tilt() or {}, load_city_tilt() or {}

lls, fav_p, fav_w, matches = [], [], [], []
for r in played.sort_values("date").itertuples(index=False):
    lh, la = model.predict_lambdas(elo[r.home_team], elo[r.away_team], neutral=bool(r.neutral))
    d = tilt.get(r.home_team, 0.0) - tilt.get(r.away_team, 0.0)
    d += city_tilt.get((r.home_team, r.city), 0.0) - city_tilt.get((r.away_team, r.city), 0.0)
    p = model.outcome_probs(lh * np.exp(d), la * np.exp(-d))
    actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
    lls.append(log_score(p[actual]))
    k = int(np.argmax(p)); fav_p.append(p[k]); fav_w.append(int(k == actual))
    matches.append({"home": r.home_team, "away": r.away_team,
                    "score": f"{int(r.home_score)}-{int(r.away_score)}",
                    "p_realised": round(float(p[actual]), 3),
                    "outcome": ["H", "D", "A"][actual]})
json.dump({
    "n": len(lls),
    "log_loss": round(float(np.mean(lls)), 4),
    "uniform": round(float(np.log(3)), 4),
    "fav_predicted": round(float(np.mean(fav_p)), 3),
    "fav_observed": round(float(np.mean(fav_w)), 3),
    "matches": matches[-12:],
}, open("outputs/history/scoring.json", "w"))
print(f"Running log-loss {np.mean(lls):.4f} vs uniform {np.log(3):.4f} | "
      f"favourite {np.mean(fav_p):.0%} predicted vs {np.mean(fav_w):.0%} observed")
