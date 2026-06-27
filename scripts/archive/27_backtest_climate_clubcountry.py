"""Step 27: heat block, re-tested with CLUB-country acclimatisation.

The original heat block (scripts/13-14) used the national-team country as a
team's climate and was REJECTED — but on a flawed proxy: a passport is not a
climate (Senegal's squad lives in Ligue 1, not Dakar). This re-test fixes the
proxy at the cheap level Simone approved: a team's acclimatisation climate is
the mean climate of its players' CLUB countries (from clubelo; non-European
clubs fall back to the national country). Same admission gate as the heat
block, on top of capital, LOTO over 6 tournaments. If this is still null, the
honest conclusion is "heat doesn't show in WC/Euro data even done right"
(Qatar 2022 was air-conditioned; Euros are mild).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from scipy.optimize import minimize_scalar
sys.path.insert(0, "src")
from wc26.data import load_results
from wc26.dixon_coles import DixonColes
from wc26.capital import norm

EXT = Path("data/external")
CODE2NAME = {
    "ALB": "Albania", "AUT": "Austria", "BEL": "Belgium", "BHZ": "Bosnia and Herzegovina",
    "BUL": "Bulgaria", "CRO": "Croatia", "CYP": "Cyprus", "CZE": "Czech Republic",
    "DEN": "Denmark", "ENG": "England", "ESP": "Spain", "FIN": "Finland", "FRA": "France",
    "GEO": "Georgia", "GER": "Germany", "GRE": "Greece", "HUN": "Hungary", "IRL": "Ireland",
    "ISL": "Iceland", "ISR": "Israel", "ITA": "Italy", "KAZ": "Kazakhstan", "LAT": "Latvia",
    "LIT": "Lithuania", "MAC": "North Macedonia", "MOL": "Moldova", "MNT": "Montenegro",
    "NED": "Netherlands", "NIR": "Northern Ireland", "NOR": "Norway", "POL": "Poland",
    "POR": "Portugal", "ROM": "Romania", "RUS": "Russia", "SCO": "Scotland", "SLK": "Slovakia",
    "SRB": "Serbia", "SUI": "Switzerland", "SVN": "Slovenia", "SWE": "Sweden", "TUR": "Turkey",
    "UKR": "Ukraine", "WAL": "Wales", "AZE": "Azerbaijan", "ARM": "Armenia", "BLR": "Belarus",
    "EST": "Estonia", "KOS": "Kosovo", "LUX": "Luxembourg", "MLT": "Malta",
}
GEO_FIX = {"England": "London", "Scotland": "Glasgow", "Wales": "Cardiff",
           "Northern Ireland": "Belfast", "Ireland": "Dublin"}
geo = json.loads((EXT / "geocode.json").read_text()) if (EXT / "geocode.json").exists() else {}


def locate(name):
    if name not in geo:
        q = GEO_FIX.get(name, name)
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": q, "count": 1}, timeout=30).json()
        h = r["results"][0]; geo[name] = [h["latitude"], h["longitude"]]
        (EXT / "geocode.json").write_text(json.dumps(geo, indent=0)); time.sleep(0.3)
    return geo[name]


def daily_tmax(name):
    f = EXT / f"tmax_{name.replace(' ', '_').replace(',', '')}.json"
    if not f.exists():
        lat, lon = locate(name)
        for _ in range(6):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                             params={"latitude": lat, "longitude": lon, "start_date": "2004-01-01",
                                     "end_date": "2025-12-31", "daily": "temperature_2m_max"}, timeout=120).json()
            if "daily" in r:
                break
            time.sleep(65)
        else:
            raise RuntimeError(f"archive failed for {name}")
        f.write_text(json.dumps(r["daily"], indent=0)); time.sleep(1.0)
    d = json.loads(f.read_text())
    return pd.Series(d["temperature_2m_max"], index=pd.to_datetime(d["time"]), dtype=float)


def clim(name, start, end):
    s = daily_tmax(name); t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    return float(np.nanmean([s.loc[t0 - pd.DateOffset(years=y):t1 - pd.DateOffset(years=y)].mean()
                             for y in range(1, 11)]))


TOURNAMENTS = [
    ("wc2014", "FIFA World Cup", "2014-06-12", "2014-07-13"), ("wc2018", "FIFA World Cup", "2018-06-14", "2018-07-15"),
    ("wc2022", "FIFA World Cup", "2022-11-20", "2022-12-18"), ("euro2016", "UEFA Euro", "2016-06-10", "2016-07-10"),
    ("euro2020", "UEFA Euro", "2021-06-11", "2021-07-11"), ("euro2024", "UEFA Euro", "2024-06-14", "2024-07-14"),
]
cap = pd.read_csv("outputs/capital.csv")
B_CAP = json.load(open("outputs/capital_beta.json"))["beta_capital"]

results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount(); elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1").sort_values("date")


def club_country_map(clubelo):
    m = {}
    for r in clubelo.itertuples(index=False):
        m.setdefault(norm(r.Club), CODE2NAME.get(r.Country))
    return m


packs = {}
for slug, comp, start, end in TOURNAMENTS:
    fd = pd.Timestamp(start)
    model = DixonColes().fit(df[df["date"] >= fd - pd.DateOffset(years=20)], fd)
    test = df[(df["tournament"] == comp) & (df["date"] >= start) & (df["date"] <= end)]
    sq = pd.read_csv(f"data/external/squads_{slug}.csv")
    cc = club_country_map(pd.read_csv(f"data/external/clubelo_{slug}.csv"))
    # team acclimatisation climate = mean clim over squad's club countries (fallback: national)
    team_clim = {}
    for team, g in sq.groupby("team"):
        countries = [cc.get(norm(c)) or team for c in g.club]
        team_clim[team] = float(np.mean([clim(x, start, end) for x in countries]))
    cz = cap[cap.tournament == slug].set_index("team")["capital_z"]
    rows = []
    for r in test.itertuples(index=False):
        vh = clim(r.city, start, end) if pd.notna(r.city) else None
        lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
        dcap = float(cz.get(r.home_team, 0.0) - cz.get(r.away_team, 0.0))
        lh, la = lh * np.exp(B_CAP * dcap), la * np.exp(-B_CAP * dcap)
        sh = max(0.0, vh - team_clim.get(r.home_team, vh)) if vh else 0.0
        sa = max(0.0, vh - team_clim.get(r.away_team, vh)) if vh else 0.0
        actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        rows.append((lh, la, (sa - sh) / 10.0, actual))  # per 10 C
    packs[slug] = (model, rows)
    print(f"{slug}: built")


def ll_vec(slug, b):
    model, rows = packs[slug]
    return np.array([-np.log(max(model.outcome_probs(lh * np.exp(b * d), la * np.exp(-b * d))[a], 1e-12))
                     for lh, la, d, a in rows])


pool0 = pool1 = n_tot = 0.0
diffs = []
print(f"\n{'held out':9s} {'b*':>9s} {'LL0':>9s} {'LL(b*)':>9s} {'delta':>8s}")
for held, *_ in TOURNAMENTS:
    others = [s for s, *_ in TOURNAMENTS if s != held]
    b = minimize_scalar(lambda b: sum(float(ll_vec(s, b).sum()) for s in others), bounds=(-1, 1), method="bounded").x
    n = len(packs[held][1]); l0 = ll_vec(held, 0.0).mean(); l1 = ll_vec(held, b).mean()
    pool0 += l0 * n; pool1 += l1 * n; n_tot += n
    diffs.append(ll_vec(held, b) - ll_vec(held, 0.0))
    print(f"{held:9s} {b:9.3f} {l0:9.4f} {l1:9.4f} {l1 - l0:+8.4f}")
d = np.concatenate(diffs); t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
verdict = "ADMITTED" if pool1 < pool0 else "REJECTED"
print(f"\nPooled OOS: base {pool0/n_tot:.4f} vs club-country heat {pool1/n_tot:.4f} "
      f"({(pool1-pool0)/n_tot:+.4f}) | paired t={t:.2f} | {verdict}")
