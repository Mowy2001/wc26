"""Step 24: diaspora feature — DECLARED DOMAIN BET, not gated (shadow only).

There is no past World Cup on US soil to backtest de-facto home support, so
this block can NEVER pass the admission gate. It exists ONLY as a registered
shadow challenger (scripts/25), never in the official model. Construction is
deliberately simple and transparent:

  support_index(team) = sqrt(US foreign-born population from that country)
                        normalised to the max among the 48 teams
  city_log_tilt[(team, US venue)] = DIA_SCALE * support_index(team)

so a team is partly "at home" in US stadiums in proportion to its diaspora.
Applies at US venues only (census is US) and not to the USA itself (already
the host). DIA_SCALE = 0.06 (~60% of the model's home-advantage coefficient
at the maximum) is a declared choice, not a fitted one.

Foreign-born figures (thousands, ~ACS 2021, rounded, approximate — this is a
domain bet, exact values are not load-bearing).
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_results, wc2026_fixtures, wc2026_group_fixtures, reconstruct_groups

FOREIGN_BORN_K = {  # US foreign-born by country of birth, thousands (approx)
    "Mexico": 10700, "South Korea": 1000, "Colombia": 810, "Haiti": 700,
    "Brazil": 500, "Ecuador": 430, "Peru": 430, "Germany": 430, "Iran": 390,
    "Japan": 330, "Iraq": 250, "Egypt": 250, "Argentina": 170, "Ghana": 180,
    "Portugal": 150, "England": 700, "Panama": 120, "South Africa": 100,
    "Morocco": 100, "Australia": 95, "Jordan": 90, "Netherlands": 90,
    "Saudi Arabia": 80, "Uzbekistan": 60, "Uruguay": 50, "Switzerland": 40,
    "Croatia": 30, "Ivory Coast": 30, "Cape Verde": 30, "DR Congo": 30,
    "Algeria": 30, "Tunisia": 25, "Senegal": 25, "Austria": 25,
    "New Zealand": 25, "Paraguay": 20, "Norway": 15, "Qatar": 5, "Curaçao": 10,
    "Scotland": 130, "Belgium": 20, "Turkey": 130, "Tunisia ": 25,
}
DIA_SCALE = 0.06
HOSTS = {"United States", "Mexico", "Canada"}
# US venues = 2026 cities not in Mexico / Canada
NON_US = {"Mexico City", "Zapopan", "Guadalupe", "Toronto", "Vancouver"}

r = load_results()
groups = reconstruct_groups(wc2026_group_fixtures(r))
teams = [t for g in groups.values() for t in g]
venues = sorted(wc2026_fixtures(r).city.dropna().unique())
us_venues = [c for c in venues if c not in NON_US]

idx = {t: np.sqrt(FOREIGN_BORN_K.get(t, 0)) for t in teams}
mx = max(idx.values())
support = {t: idx[t] / mx for t in teams}

rows = []
for t in teams:
    if t == "United States" or support[t] == 0:
        continue
    for c in us_venues:
        rows.append({"team": t, "city": c, "log_tilt": round(DIA_SCALE * support[t], 4)})
out = pd.DataFrame(rows)
out.to_csv("outputs/diaspora_tilt.csv", index=False)
top = sorted(support.items(), key=lambda kv: -kv[1])[:8]
print(f"diaspora tilt: {len(out)} (team, US-venue) entries; DIA_SCALE={DIA_SCALE} (declared)")
print("strongest de-facto support at US venues:")
for t, s in top:
    print(f"  {t:14s} index {s:.2f}  ->  +{DIA_SCALE*s:.3f} log-lambda")
