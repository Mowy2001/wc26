"""Step 31: per-match replay — simulate the tournament after each played match.

For k = 0,1,...,(matches played), condition on the first k results (in kickoff
order) and run the tournament. Produces a clean per-match history (champion
odds, qualification, modal bracket) for the site's time slider — independent of
the irregular 4-hourly live snapshots. Uses the deployed tilts; point estimate
(no bootstrap) at 10k sims for speed — the bootstrap effect is ~0.07pp, far
below what the slider shows.
"""
import json, sys, time
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import (load_results, load_shootouts, wc2026_fixtures,
                       wc2026_group_fixtures, reconstruct_groups, GROUP_STAGE_END)
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes
from wc26.simulate import simulate_tournament
from wc26.tilts import load_team_tilt, load_city_tilt

results = load_results()
elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
df = df[df["date"] < "2026-06-11"]
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
model = DixonColes().fit(df, pd.Timestamp("2026-06-11"))
elo = ratings_asof(elo_hist, "2026-06-11")
gfx = wc2026_group_fixtures(results)
groups = reconstruct_groups(gfx)
tilt, city_tilt = load_team_tilt(), load_city_tilt()

played = wc2026_fixtures(results).dropna(subset=["home_score", "away_score"]).sort_values("date")
print(f"replaying {len(played)} played matches -> {len(played)+1} snapshots")

snaps = []
t0 = time.time()
for k in range(len(played) + 1):
    sub = played.iloc[:k]
    fixed = {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
             for r in sub[sub.date <= GROUP_STAGE_END].itertuples(index=False)}
    res = simulate_tournament(groups, gfx, model, elo, n_sims=10000, fixed_results=fixed,
                              team_log_tilt=tilt, city_log_tilt=city_tilt)
    t = res["teams"]
    last = "(eve)" if k == 0 else f"{played.iloc[k-1].home_team} {int(played.iloc[k-1].home_score)}-{int(played.iloc[k-1].away_score)} {played.iloc[k-1].away_team}"
    snaps.append({
        "k": k,
        "date": (str(played.iloc[k-1].date.date()) if k > 0 else "2026-06-11"),
        "last_match": last,
        "champion": {tm: round(float(t.loc[tm, "P_champion"]), 4) for tm in t.index},
        "qualify": {tm: round(float(t.loc[tm, "P_qualify"]), 4) for tm in t.index},
    })
    print(f"  k={k:2d} {last[:42]:42s} {time.time()-t0:5.0f}s")

json.dump({"snapshots": snaps, "teams_group": {t: g for g, ts in groups.items() for t in ts}},
          open("outputs/history/replay.json", "w"))
print(f"outputs/history/replay.json written ({len(snaps)} snapshots)")
