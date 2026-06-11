"""Step 8: bootstrap the DC parameters (B refits on resampled training data).

Output: outputs/dc_bootstrap.json — one parameter draw per line item, consumed
by the tournament simulator (one draw per simulated tournament, backlog #8).
"""
import json, sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_results
from wc26.dixon_coles import DixonColes

results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
train = df[df["date"] >= "2005-01-01"]

t0 = time.time()
model = DixonColes()
draws = model.bootstrap_params(train, pd.Timestamp("2026-06-11"), B=100)
print(f"{len(draws)} converged draws in {time.time() - t0:.0f}s")

arr = pd.DataFrame(draws)
print(arr.describe().loc[["mean", "std"]].round(4).to_string())
with open("outputs/dc_bootstrap.json", "w") as f:
    json.dump(draws, f)
print("outputs/dc_bootstrap.json written")
