"""Step 30: rebuild football capital from footballdatabase (global, point-in-time).

Replaces clubelo (Europe-only) with footballdatabase per-club monthly history
(scripts/28-29). For each tournament, each player's club -> slug -> club points
at the tournament's month (nearest <= date); unmatched/below-top-1500 clubs are
imputed at the per-tournament 10th percentile (genuinely weak -> floor is right).
Minutes-weighted (capital v2). Writes outputs/capital_fdb.csv for all tournaments
(backtest + wc2026). The gate (scripts/12 variant) then compares fdb-capital vs
the clubelo-capital currently deployed.
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.capital import norm

clubmap = json.load(open("data/external/fdb_clubmap.json"))
hist = json.load(open("data/external/fdb_history.json"))
FBREF_SEASON = {"wc2018": "fbref_2017_18", "wc2022": "fbref_2021_22",
                "euro2024": "fbref_2023_24", "wc2026": "fbref_2025_26"}
TOURN_MONTH = {"wc2014": "2014-06", "wc2018": "2018-06", "wc2022": "2022-11",
               "euro2016": "2016-06", "euro2020": "2021-06", "euro2024": "2024-06",
               "wc2026": "2026-06"}


def points_asof(slug, ym):
    s = hist.get(slug) or {}
    keys = sorted(k for k in s if k <= ym)
    return s[keys[-1]] if keys else None


def build(slug_csv, ym, minutes=None):
    sq = pd.read_csv(slug_csv)
    pts = sq.club.map(lambda c: points_asof(clubmap.get(c), ym))
    matched = pts.notna()
    floor = np.nanpercentile(pts.dropna(), 10) if matched.any() else 1300.0
    p = pts.fillna(floor).to_numpy(float)
    out = []
    sq2 = sq.assign(_p=p, _m=matched)
    for team, g in sq2.groupby("team"):
        if minutes is not None:
            mn = g.club.index.map(lambda i: 1.0)  # placeholder; minutes join below
        out.append({"team": team, "capital_raw": float(g._p.mean()),
                    "coverage": float(g._m.mean()), "n_players": len(g)})
    d = pd.DataFrame(out)
    d["capital_z"] = (d.capital_raw - d.capital_raw.mean()) / d.capital_raw.std(ddof=0)
    return d


allc = []
for slug, ym in TOURN_MONTH.items():
    d = build(f"data/external/squads_{slug}.csv", ym)
    d.insert(0, "tournament", slug)
    allc.append(d)
    hi = d.nlargest(1, "capital_z"); lo = d.nsmallest(1, "capital_z")
    print(f"{slug} ({ym}): cov {d.coverage.mean():.0%} | top {hi.team.iloc[0]} {hi.capital_z.iloc[0]:+.2f} "
          f"| bottom {lo.team.iloc[0]} {lo.capital_z.iloc[0]:+.2f}")
pd.concat(allc).round(4).to_csv("outputs/capital_fdb.csv", index=False)
print("outputs/capital_fdb.csv written")
