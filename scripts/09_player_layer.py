"""Step 9: player layer — Golden Boot and most-distinct-scorers distributions.

Conditions on the team-goal samples saved by step 04 (same simulations,
same seed: the player layer never re-rolls the tournament). Allocation
weights are the gate-16 winner under the corrected bucket metric:
official-squad filter + age discount (alpha=0.1); the club-form blend
was tested on top and rejected (scripts/18).
"""
import sys, time
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_goalscorers, load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.players import estimate_debutant_share, squad_weights, allocate_goals

ASOF = pd.Timestamp("2026-06-11")
gs = load_goalscorers()
goal_samples = pd.read_parquet("outputs/goal_samples.parquet")
teams = list(goal_samples.columns)

deb = estimate_debutant_share(gs)
print(f"Debutant share of WC goals (2014/18/22 average): {deb:.1%}")

squads = pd.read_csv("data/external/squads_wc2026.csv", parse_dates=["birth"])
weights = {t: squad_weights(gs, squads, t, ASOF, deb, age_alpha=0.1, drop_to_bucket=False)
           for t in teams}
t0 = time.time()
res = allocate_goals(goal_samples, weights)
print(f"Allocation done in {time.time() - t0:.0f}s\n")

players = res["players"]
players.round(4).to_csv("outputs/golden_boot.csv", index=False)
res["distinct"].round(4).to_csv("outputs/distinct_scorers.csv")

print("Golden Boot — top 15:")
print(players.head(15).round(3).to_string(index=False))

# Golden Boot market (American odds, 2026-06-11, mid of the recorded range)
market = {"Kylian Mbappé": 575, "Harry Kane": 675, "Lionel Messi": 1150, "Erling Haaland": 1350}
print("\nModel vs market (raw implied, includes vig):")
for p, odds in market.items():
    row = players[players.player == p]
    pm = float(row.P_golden_boot.iloc[0]) if len(row) else 0.0
    print(f"  {p:16s} model {pm:6.1%}   market {1 / (1 + odds / 100):6.1%}")

print("\nMost distinct scorers — top 6:")
print(res["distinct"].head(6).round(3).to_string())
