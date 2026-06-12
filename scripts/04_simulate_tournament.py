"""Step 4: full-tournament Monte Carlo (model v2): groups + R32 bracket -> champion.

v2 = v1.1 + parameter bootstrap (backlog #8): each simulated tournament uses
one bootstrap draw of the DC parameters, so the headline probabilities
integrate over epistemic parameter uncertainty. Falls back to the point
estimate if outputs/dc_bootstrap.json is missing (run scripts/08 first).
Also saves per-simulation tournament goals (outputs/goal_samples.parquet),
the conditioning input of the player layer (scripts/09).
"""
import json, sys, time
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
model = DixonColes().fit(df[df["date"] >= "2005-01-01"], pd.Timestamp("2026-06-11"))

try:
    draws = json.load(open("outputs/dc_bootstrap.json"))
    print(f"Using {len(draws)} bootstrap parameter draws")
except FileNotFoundError:
    draws = None
    print("No bootstrap draws found (scripts/08) — using the point estimate")

elo_now = ratings_asof(elo_hist, "2026-06-11")
gfx = wc2026_group_fixtures(results)
groups = reconstruct_groups(gfx)

from wc26.tilts import load_team_tilt
tilt = load_team_tilt()
print(f"Residual tilts active for {len(tilt or {})} teams (capital + fatigue)")

t0 = time.time()
res = simulate_tournament(groups, gfx, model, elo_now, n_sims=20000,
                          param_draws=draws, collect_goal_samples=True,
                          team_log_tilt=tilt)
tbl = res["teams"]
tbl.round(4).to_csv("outputs/tournament_probs_v1.csv")
res["goal_samples"].to_parquet("outputs/goal_samples.parquet")
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
