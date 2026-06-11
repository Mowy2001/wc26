"""Step 4: full-tournament Monte Carlo (model v1): groups + R32 bracket -> champion."""
import sys, time
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes
from wc26.simulate import simulate_tournament

results = load_results()
elo_hist = pd.read_parquet("outputs/elo_history.parquet")

# Fit at tournament eve, point-in-time
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
model = DixonColes(xi=0.0018).fit(df[df["date"] >= "2005-01-01"], pd.Timestamp("2026-06-11"))

elo_now = ratings_asof(elo_hist, "2026-06-11")
gfx = wc2026_group_fixtures(results)
groups = reconstruct_groups(gfx)

t0 = time.time()
res = simulate_tournament(groups, gfx, model, elo_now, n_sims=20000)
tbl = res["teams"]
tbl.round(4).to_csv("outputs/tournament_probs_v1.csv")
print(f"20k tournament simulations in {time.time() - t0:.0f}s\n")

top = tbl.sort_values("P_champion", ascending=False).head(12)
print("P(champion) — top 12:")
print(top[["P_champion", "P_final", "P_SF", "P_QF"]].round(3).to_string())

# BetMGM outright implied probabilities (American odds, 2026-06-11, margin NOT removed)
betmgm = {"Spain": 450, "France": 500, "England": 700, "Brazil": 800,
          "Portugal": 900, "Argentina": 900, "Germany": 1400,
          "Netherlands": 2000, "United States": 5000}
print("\nModel vs BetMGM implied (raw, includes vig):")
for t, odds in betmgm.items():
    print(f"  {t:14s} model {tbl.loc[t, 'P_champion']:6.1%}   market {1 / (1 + odds / 100):6.1%}")
