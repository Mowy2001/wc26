"""Step 10: live forecast update during the group stage.

Workflow during the tournament:
  1. refresh data/raw/results.csv from upstream (played 2026 matches get scores)
  2. run this script: played group matches become `fixed_results`, the
     remaining fixtures re-randomise around them (same seed discipline)
  3. it refreshes outputs + site/data.js — reload the site and you're live.

Knockout conditioning: real KO outcomes are imposed at the pair level
(winner + goals, shootout winners from shootouts.csv) whenever a simulated
tie features the same two teams. In the rare simulation where a random
tie-break sends a different qualifier to that slot, the tie is simulated —
internally consistent, deviation declared. Once FIFA publishes the actual
third-place allocation, pass thirds_override to pin it exactly.
"""
import json, subprocess, sys
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import (load_results, load_shootouts, wc2026_group_fixtures,
                       wc2026_played_ko, reconstruct_groups)
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes
from wc26.simulate import simulate_tournament

results = load_results()
gfx = wc2026_group_fixtures(results)
played = gfx.dropna(subset=["home_score", "away_score"])
fixed = {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
         for r in played.itertuples(index=False)}
fixed_ko = wc2026_played_ko(results, load_shootouts())
print(f"Played: {len(fixed)}/{len(gfx)} group matches, {len(fixed_ko)}/31 knockout ties")
if not fixed and not fixed_ko:
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
from wc26.tilts import load_team_tilt, load_city_tilt
tilt = load_team_tilt()
city_tilt = load_city_tilt()

# Real FIFA third-place allocation (the bracket is fixed positions, not a draw):
# once the groups are complete, pin each 3rd-place slot to the group FIFA assigned
# it to. Only valid with all groups played, so apply it only then.
import os
THIRDS = None
if os.path.exists("outputs/thirds_override.json") and len(fixed) >= len(gfx):
    THIRDS = {int(k): v for k, v in json.load(open("outputs/thirds_override.json")).items() if k.isdigit()}
    print(f"Applying real thirds allocation ({len(THIRDS)} slots)")

res = simulate_tournament(groups, gfx, model, elo_now, n_sims=20000,
                          fixed_results=fixed, param_draws=draws,
                          collect_goal_samples=True, team_log_tilt=tilt,
                          city_log_tilt=city_tilt, thirds_override=THIRDS,
                          fixed_ko_results=fixed_ko, collect_bracket=True)
res["teams"].round(4).to_csv("outputs/tournament_probs_v1.csv")
res["goal_samples"].to_parquet("outputs/goal_samples.parquet")
res["bracket"].to_csv("outputs/bracket.csv", index=False)

print("Tournament probabilities refreshed.")

# Append an immutable snapshot to the forecast timeline (never overwritten;
# the eve-of-tournament baseline lives in outputs/history/baseline_eve.csv).
from datetime import datetime, timezone
snap = res["teams"].round(4).reset_index(names="team")
snap.insert(0, "snapshot_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
snap.insert(1, "played_groups", len(fixed))
snap.insert(2, "played_ko", len(fixed_ko))
tl = "outputs/history/timeline.csv"
import os
snap.to_csv(tl, mode="a", header=not os.path.exists(tl), index=False)
print(f"Timeline snapshot appended ({len(fixed)}+{len(fixed_ko)} played).")

subprocess.run([sys.executable, "scripts/09_player_layer.py"], check=True)

# Extend the match-by-match slider history: one snapshot PER PLAYED MATCH.
# scripts/31 --incremental computes every k missing from replay.json, so even if
# several matches landed since the last refresh the slider gets one tick per
# match, never a single aggregated jump.
subprocess.run([sys.executable, "scripts/31_replay_history.py", "--incremental"], check=True)

subprocess.run([sys.executable, "scripts/25_shadow_scores.py"], check=True)
subprocess.run([sys.executable, "scripts/15_benchmark_report.py"], check=True)
# live market odds: append a snapshot + refresh the next-matches feed. Non-fatal —
# needs ODDS_API_KEY (env or .secrets); skip cleanly if absent or the API is down.
for _step in ("41_fetch_odds", "42_export_next_matches"):
    try:
        subprocess.run([sys.executable, f"scripts/{_step}.py"], check=True)
    except Exception as _e:
        print(f"  odds step {_step} skipped: {_e}")
subprocess.run([sys.executable, "scripts/43_export_bracket_dists.py"], check=True)  # no API needed
subprocess.run([sys.executable, "scripts/05_export_site_data.py"], check=True)
print("Site refreshed — reload site/index.html.")
