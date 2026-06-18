"""Step 9: player layer — Golden Boot and most-distinct-scorers distributions.

Conditions on the team-goal samples saved by step 04/10 (same simulations,
same seed). Allocation weights are the gate-16 winner: official-squad filter +
age discount (alpha=0.1), frozen at tournament eve.

LIVE: if 2026 matches have been played, the Golden Boot credits the REAL goals
already scored (from the live goalscorers feed) and only simulates each team's
REMAINING goals — conditioning on what happened without re-estimating form
(allocate_goals_live). Pre-tournament it falls back to the pure simulation.
"""
import sys, time
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_goalscorers
from wc26.players import estimate_debutant_share, squad_weights, allocate_goals, allocate_goals_live

ASOF = pd.Timestamp("2026-06-11")
gs = load_goalscorers()
goal_samples = pd.read_parquet("outputs/goal_samples.parquet")
teams = list(goal_samples.columns)

deb = estimate_debutant_share(gs)
print(f"Debutant share of WC goals (2014/18/22 average): {deb:.1%}")

squads = pd.read_csv("data/external/squads_wc2026.csv", parse_dates=["birth"])
weights = {t: squad_weights(gs, squads, t, ASOF, deb, age_alpha=0.1, drop_to_bucket=False)
           for t in teams}

# real goals already scored in the 2026 tournament (own goals excluded)
wc = gs[(gs["date"] >= "2026-06-11") & (gs["date"] <= "2026-07-19")
        & (~gs["own_goal"].astype(bool)) & (gs["team"].isin(teams))].dropna(subset=["scorer"])
real_by_team = {t: g["scorer"].value_counts().to_dict() for t, g in wc.groupby("team")}
n_real = int(wc.shape[0])

t0 = time.time()
if n_real:
    players = allocate_goals_live(goal_samples, weights, real_by_team)["players"]
    print(f"LIVE Golden Boot: {n_real} real goals credited so far")
else:
    players = allocate_goals(goal_samples, weights)["players"]
distinct = allocate_goals(goal_samples, weights)["distinct"]
print(f"Allocation done in {time.time() - t0:.0f}s\n")

players.round(4).to_csv("outputs/golden_boot.csv", index=False)
distinct.round(4).to_csv("outputs/distinct_scorers.csv")

print("Golden Boot — top 15:")
print(players.head(15).round(3).to_string(index=False))

market = {"Kylian Mbappé": 575, "Harry Kane": 675, "Lionel Messi": 1150, "Erling Haaland": 1350}
print("\nModel vs market (raw implied, includes vig):")
for p, odds in market.items():
    row = players[players.player == p]
    pm = float(row.P_golden_boot.iloc[0]) if len(row) else 0.0
    print(f"  {p:16s} model {pm:6.1%}   market {1 / (1 + odds / 100):6.1%}")

print("\nMost distinct scorers — top 6:")
print(distinct.head(6).round(3).to_string())
