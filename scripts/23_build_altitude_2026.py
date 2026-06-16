"""Step 23: deploy the admitted altitude block to 2026 (city-dependent tilt).

Habitual altitude per team = mean elevation of its home (non-neutral) match
cities with a known elevation. For each 2026 team x venue, the team's tilt is
    city_log_tilt[(team, city)] = -beta * max(0, venue_alt - habitual)/1000
so a lowland side is damped at Mexico City (2240 m) / Zapopan (1565 m) while
altitude-adapted teams (Mexico, Ecuador) are barely touched. beta from the
CONMEBOL gate (scripts/22). Other venues are low -> tilt 0 there.
"""
import json, sys
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_results, wc2026_fixtures, wc2026_group_fixtures, reconstruct_groups

ALT = json.load(open("data/external/altitude.json"))
BETA = json.load(open("outputs/altitude_beta.json"))["beta_altitude_per_km"]
r = load_results()
home = r[~r.neutral.astype(bool)].dropna(subset=["city"])
home = home[home.city.isin(ALT)]
habitual = home.assign(a=home.city.map(ALT)).groupby("home_team")["a"].mean()
default = float(habitual.median())

groups = reconstruct_groups(wc2026_group_fixtures(r))
teams = [t for g in groups.values() for t in g]
venues = {c: ALT[c] for c in wc2026_fixtures(r).city.dropna().unique() if ALT.get(c) is not None}

rows = []
for t in teams:
    hab = float(habitual.get(t, default))
    for city, va in venues.items():
        suffer = max(0.0, va - hab) / 1000.0
        if suffer > 0:
            rows.append({"team": t, "city": city, "habitual_m": round(hab),
                         "venue_m": round(va), "log_tilt": round(-BETA * suffer, 4)})
out = pd.DataFrame(rows)
out.to_csv("outputs/altitude_tilt.csv", index=False)
print(f"altitude tilt: {len(out)} (team,city) entries, beta={BETA}/km")
mx = out[out.city == "Mexico City"].sort_values("log_tilt")
print("\nMexico City (2240m) — most penalised (lowland) teams:")
print(mx.head(5)[["team", "habitual_m", "log_tilt"]].to_string(index=False))
print("\nMexico City — least penalised (altitude-adapted):")
print(mx.tail(4)[["team", "habitual_m", "log_tilt"]].to_string(index=False))
