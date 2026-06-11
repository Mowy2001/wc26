"""Step 1: build Elo from scratch, validate, reconstruct 2026 groups."""
import sys, time
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.elo import compute_elo_history, ratings_asof

t0 = time.time()
results = load_results()
print(f"Matches in dataset: {len(results)} ({results['date'].min().date()} -> {results['date'].max().date()})")

elo_hist = compute_elo_history(results)
elo_hist.to_parquet("outputs/elo_history.parquet")
print(f"Elo computed over {len(elo_hist)} played matches in {time.time()-t0:.1f}s")

today = ratings_asof(elo_hist, "2026-06-11")
print("\nTop 15 Elo, World Cup eve (2026-06-11):")
print(today.sort_values(ascending=False).head(15).round(0).to_string())

groups = reconstruct_groups(wc2026_group_fixtures(results))
print("\n2026 groups reconstructed from the schedule:")
for g, teams in groups.items():
    print(f"  {g}: {', '.join(teams)}")
