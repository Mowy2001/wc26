"""Step 1: build Elo from scratch, validate, reconstruct 2026 groups.

The canonical Elo history is altitude-aware (ALT_GAMMA, scripts/35, deployed
2026-06-25): home wins at altitude earn fewer rating points so altitude
fortresses do not inflate ratings that fail to travel to a sea-level neutral
tournament. Set ALT_GAMMA=0 in elo.py to recover the classic flat advantage.
"""
import json, sys, time
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.elo import ALT_GAMMA, compute_elo_history, habitual_altitudes, ratings_asof

t0 = time.time()
results = load_results()
print(f"Matches in dataset: {len(results)} ({results['date'].min().date()} -> {results['date'].max().date()})")

city_alt = json.load(open("data/external/altitude.json"))
habitual = habitual_altitudes(results, city_alt)
elo_hist = compute_elo_history(results, alt_gamma=ALT_GAMMA, habitual=habitual, city_alt=city_alt)
elo_hist.to_parquet("outputs/elo_history.parquet")
print(f"Elo computed over {len(elo_hist)} played matches in {time.time()-t0:.1f}s "
      f"(altitude-aware, gamma={ALT_GAMMA})")

today = ratings_asof(elo_hist, "2026-06-11")
print("\nTop 15 Elo, World Cup eve (2026-06-11):")
print(today.sort_values(ascending=False).head(15).round(0).to_string())

groups = reconstruct_groups(wc2026_group_fixtures(results))
print("\n2026 groups reconstructed from the schedule:")
for g, teams in groups.items():
    print(f"  {g}: {', '.join(teams)}")
