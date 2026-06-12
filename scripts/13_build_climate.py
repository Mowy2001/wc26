"""Step 13: build the climate block (backlog #5b).

Heat-mismatch design, point-in-time by construction:
  venue_heat(city, T)  = mean daily Tmax at the venue over tournament T's
                         month-day window, averaged over the 10 PRIOR years
                         (climatology — never the tournament's own weather)
  team_heat(team, T)   = same window at the team's home country (what its
                         fans-and-federation climate is doing at that time
                         of year; club-country refinement left to v2)
  suffering(team,city) = max(0, venue_heat - team_heat)   [°C]

Per-match differential (for the backtest and the simulator):
  d = suffering(home, city) - suffering(away, city)

Declared limits: Qatar 2022 was air-conditioned (noisy fold, kept anyway);
altitude is omitted — no high-altitude venue exists in the backtest window,
so a coefficient for it cannot be admitted under the rule.

APIs: open-meteo geocoding + archive (free, cached in data/external/).
Output: outputs/climate.csv (tournament, team, city, suffering).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests
sys.path.insert(0, "src")
from wc26.data import load_results

EXT = Path("data/external"); EXT.mkdir(exist_ok=True)
TOURNAMENTS = [
    ("wc2014",   "FIFA World Cup", "2014-06-12", "2014-07-13"),
    ("wc2018",   "FIFA World Cup", "2018-06-14", "2018-07-15"),
    ("wc2022",   "FIFA World Cup", "2022-11-20", "2022-12-18"),
    ("euro2016", "UEFA Euro",      "2016-06-10", "2016-07-10"),
    ("euro2020", "UEFA Euro",      "2021-06-11", "2021-07-11"),
    ("euro2024", "UEFA Euro",      "2024-06-14", "2024-07-14"),
    ("wc2026",   "FIFA World Cup", "2026-06-11", "2026-07-19"),
]
# geocoding queries that need disambiguation (else: query = name as-is)
GEO_FIX = {"Guadalupe": "Guadalupe, Nuevo Leon", "Arlington": "Arlington, Texas",
           "East Rutherford": "East Rutherford", "Foxborough": "Foxborough",
           "England": "London", "Scotland": "Glasgow", "Wales": "Cardiff",
           "Northern Ireland": "Belfast", "United States": "Kansas City",
           "South Korea": "Seoul", "North Korea": "Pyongyang",
           "Ivory Coast": "Abidjan", "DR Congo": "Kinshasa", "Cape Verde": "Praia",
           "Bosnia and Herzegovina": "Sarajevo", "Czech Republic": "Prague",
           "Republic of Ireland": "Dublin", "New Zealand": "Wellington"}

GEO_CACHE = EXT / "geocode.json"
geo = json.loads(GEO_CACHE.read_text()) if GEO_CACHE.exists() else {}


def locate(name: str) -> tuple[float, float]:
    if name not in geo:
        q = GEO_FIX.get(name, name)
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": q, "count": 1}, timeout=30).json()
        hit = r["results"][0]
        geo[name] = [hit["latitude"], hit["longitude"]]
        GEO_CACHE.write_text(json.dumps(geo, indent=0))
        time.sleep(0.3)
    return tuple(geo[name])


def daily_tmax(name: str) -> pd.Series:
    """Daily Tmax 2004-2025 at a location, cached."""
    f = EXT / f"tmax_{name.replace(' ', '_').replace(',', '')}.json"
    if not f.exists():
        lat, lon = locate(name)
        for attempt in range(6):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                             params={"latitude": lat, "longitude": lon,
                                     "start_date": "2004-01-01", "end_date": "2025-12-31",
                                     "daily": "temperature_2m_max"}, timeout=120).json()
            if "daily" in r:
                break
            wait = 65 if "limit" in str(r.get("reason", "")).lower() else 5 * (attempt + 1)
            print(f"  archive retry for {name} in {wait}s ({r.get('reason', 'unknown')})")
            time.sleep(wait)
        else:
            raise RuntimeError(f"archive API failed for {name}: {r}")
        f.write_text(json.dumps(r["daily"], indent=0))
        time.sleep(1.0)
    d = json.loads(f.read_text())
    return pd.Series(d["temperature_2m_max"], index=pd.to_datetime(d["time"]), dtype=float)


def climatology(name: str, start: str, end: str) -> float:
    """Mean Tmax over the [start, end] month-day window, 10 years before `start`."""
    s = daily_tmax(name)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    vals = []
    for y in range(1, 11):
        a, b = t0 - pd.DateOffset(years=y), t1 - pd.DateOffset(years=y)
        vals.append(s.loc[a:b].mean())
    return float(np.nanmean(vals))


results = load_results()
rows = []
for slug, comp, start, end in TOURNAMENTS:
    m = results[(results["tournament"] == comp) & (results["date"] >= start) & (results["date"] <= end)]
    cities = sorted(m["city"].dropna().unique())
    teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    vheat = {c: climatology(c, start, end) for c in cities}
    theat = {t: climatology(t, start, end) for t in teams}
    for t in teams:
        for c in cities:
            rows.append({"tournament": slug, "team": t, "city": c,
                         "suffering": max(0.0, vheat[c] - theat[t])})
    hot = max(vheat, key=vheat.get)
    print(f"{slug}: {len(teams)} teams x {len(cities)} cities | hottest venue {hot} {vheat[hot]:.1f}C")

pd.DataFrame(rows).round(2).to_csv("outputs/climate.csv", index=False)
print("outputs/climate.csv written")
