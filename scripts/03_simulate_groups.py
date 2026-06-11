"""Step 3: first Monte Carlo run of the 2026 group stage (model v0, Elo-only)."""
import sys, time
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes

results = load_results()
elo_hist = pd.read_parquet("outputs/elo_history.parquet")

# Fit at tournament eve, point-in-time
df = results.dropna(subset=["home_score","away_score"]).copy()
key = ["date","home_team","away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key+["dup","elo_home_pre","elo_away_pre"]], on=key+["dup"], validate="1:1")
model = DixonColes(xi=0.0018).fit(df[df["date"]>="2005-01-01"], pd.Timestamp("2026-06-11"))

elo_now = ratings_asof(elo_hist, "2026-06-11")
gfx = wc2026_group_fixtures(results)
groups = reconstruct_groups(gfx)

t0 = time.time()
out = simres = simulate = None
from wc26.simulate import simulate_group_stage
res = simulate_group_stage(groups, gfx, model, elo_now, n_sims=20000)
tbl = res["groups"]
tbl.round(3).to_csv("outputs/group_stage_probs_v0.csv")
print(f"20k simulations in {time.time()-t0:.0f}s\n")
for g in "ABCDEFGHIJKL":
    sub = tbl[tbl["group"]==g].sort_values("P_qualify", ascending=False)
    line = " | ".join(f"{t} {sub.loc[t,'P_qualify']:.0%}" for t in sub.index)
    print(f"Group {g}: {line}")
print("\nP(top-scoring team of the group stage) — top 6:")
print(tbl["P_top_scoring_team"].sort_values(ascending=False).head(6).round(3).to_string())
