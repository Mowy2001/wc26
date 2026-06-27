"""Step 34: hybrid capital — clubelo where available, footballdatabase for the gap.

Removes clubelo's Europe-only bias without discarding its high-quality European
Elo. Per tournament (point-in-time): a club's strength is its clubelo Elo if
matched; else its footballdatabase points mapped onto the clubelo scale via a
linear bridge fitted on clubs rated by BOTH that tournament; else the 10th-pct
floor. z-scored within tournament. Writes outputs/capital_hybrid.csv.
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.capital import _club_elo_fn, norm

clubmap = json.load(open("data/external/fdb_clubmap.json"))
fhist = json.load(open("data/external/fdb_history.json"))
TM = {"wc2014": "2014-06", "wc2018": "2018-06", "wc2022": "2022-11", "euro2016": "2016-06",
      "euro2020": "2021-06", "euro2024": "2024-06", "wc2026": "2026-06"}


def fdb_pts(club, ym):
    s = fhist.get(clubmap.get(club) or "") or {}
    ks = sorted(k for k in s if k <= ym)
    return s[ks[-1]] if ks else None


rows = []
for slug, ym in TM.items():
    sq = pd.read_csv(f"data/external/squads_{slug}.csv")
    clubelo = pd.read_csv(f"data/external/clubelo_{slug}.csv")
    ce = _club_elo_fn(clubelo)
    cev = sq.club.map(ce)
    fdv = sq.club.map(lambda c: fdb_pts(c, ym))
    # bridge fdb -> clubelo on clubs with both
    both = pd.DataFrame({"ce": cev, "fd": fdv}).dropna()
    if len(both) > 10:
        a, b = np.polyfit(both.fd, both.ce, 1)
    else:
        a, b = 1.0, 0.0
    val = cev.copy()
    need = val.isna() & fdv.notna()
    val[need] = a * fdv[need] + b
    floor = np.nanpercentile(val.dropna(), 10)
    sq = sq.assign(_v=val.fillna(floor), _matched=(cev.notna() | fdv.notna()))
    for team, g in sq.groupby("team"):
        rows.append({"tournament": slug, "team": team, "capital_raw": float(g._v.mean()),
                     "coverage": float(g._matched.mean())})
    cov = sq._matched.mean()
    print(f"{slug}: bridge fd->ce a={a:.2f} b={b:.0f} | coverage {cov:.0%}")

d = pd.DataFrame(rows)
d["capital_z"] = d.groupby("tournament").capital_raw.transform(lambda x: (x - x.mean()) / x.std(ddof=0))
d.round(4).to_csv("outputs/capital_hybrid.csv", index=False)
print("outputs/capital_hybrid.csv written")
