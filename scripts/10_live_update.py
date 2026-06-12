"""Step 10: live forecast update during the group stage.

Workflow during the tournament:
  1. refresh data/raw/results.csv from upstream (played 2026 matches get scores)
  2. run this script: played group matches become `fixed_results`, the
     remaining fixtures re-randomise around them (same seed discipline)
  3. it refreshes outputs + site/data.js — reload the site and you're live.

Limitation (documented): conditioning covers the 72 group fixtures; knockout
conditioning (bracket partially played) is a follow-up.
"""
import json, subprocess, sys
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes
from wc26.simulate import simulate_tournament

results = load_results()
gfx = wc2026_group_fixtures(results)
played = gfx.dropna(subset=["home_score", "away_score"])
fixed = {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
         for r in played.itertuples(index=False)}
print(f"Played group matches found: {len(fixed)} / {len(gfx)}")
if not fixed:
    print("Nothing played yet — the eve-of-tournament forecast stands.")

elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
# Ratings and parameters stay frozen at tournament eve (point-in-time):
# live updating changes WHAT IS KNOWN (results), not what the model believes.
model = DixonColes().fit(df[df["date"] >= "2005-01-01"], pd.Timestamp("2026-06-11"))
elo_now = ratings_asof(elo_hist, "2026-06-11")
groups = reconstruct_groups(gfx)
try:
    draws = json.load(open("outputs/dc_bootstrap.json"))
except FileNotFoundError:
    draws = None
from wc26.tilts import load_team_tilt
tilt = load_team_tilt()

res = simulate_tournament(groups, gfx, model, elo_now, n_sims=20000,
                          fixed_results=fixed, param_draws=draws,
                          collect_goal_samples=True, team_log_tilt=tilt)
res["teams"].round(4).to_csv("outputs/tournament_probs_v1.csv")
res["goal_samples"].to_parquet("outputs/goal_samples.parquet")
print("Tournament probabilities refreshed.")

subprocess.run([sys.executable, "scripts/09_player_layer.py"], check=True)
subprocess.run([sys.executable, "scripts/05_export_site_data.py"], check=True)
print("Site refreshed — reload site/index.html.")
