"""Step 21: does minutes-weighting the club-Elo feature help? (capital v2)

Same LOTO admission protocol as scripts/12, restricted to the three folds
with point-in-time FBref minutes (WC2018, WC2022, Euro2024). For each fold
we build TWO features from identical squad/clubelo data and matching:
  v1: equal-weight squad-mean club Elo (deployed)
  v2: minutes-weighted squad-mean club Elo
Each is fitted as a lambda tilt; we compare pooled OOS log-loss. v2 is
adopted only if it beats v1 (the deployed baseline), same bar as any block.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from wc26.data import load_results
from wc26.dixon_coles import DixonColes
from wc26.capital import capital_table, norm

FOLDS = [("wc2018", "FIFA World Cup", "2018-06-14", "2018-07-15", "fbref_2017_18"),
         ("wc2022", "FIFA World Cup", "2022-11-20", "2022-12-18", "fbref_2021_22"),
         ("euro2024", "UEFA Euro", "2024-06-14", "2024-07-14", "fbref_2023_24")]

results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1").sort_values("date")


def features(slug, fbslug):
    squads = pd.read_csv(f"data/external/squads_{slug}.csv")
    clubelo = pd.read_csv(f"data/external/clubelo_{slug}.csv")
    fb = pd.read_csv(f"data/external/{fbslug}.csv")
    mins = {norm(p): m for p, m in zip(fb.player, fb.minutes)}
    v1 = capital_table(squads, clubelo).set_index("team")["capital_z"]
    v2 = capital_table(squads, clubelo, minutes=mins).set_index("team")["capital_z"]
    return v1, v2


packs = {}
for slug, comp, start, end, fbslug in FOLDS:
    fd = pd.Timestamp(start)
    model = DixonColes().fit(df[df["date"] >= fd - pd.DateOffset(years=20)], fd)
    test = df[(df["tournament"] == comp) & (df["date"] >= start) & (df["date"] <= end)]
    v1, v2 = features(slug, fbslug)
    print(f"{slug}: corr(v1, v2) = {np.corrcoef(v1.reindex(v2.index), v2)[0, 1]:.3f}")
    rows = []
    for r in test.itertuples(index=False):
        lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
        d1 = float(v1.get(r.home_team, 0.0) - v1.get(r.away_team, 0.0))
        d2 = float(v2.get(r.home_team, 0.0) - v2.get(r.away_team, 0.0))
        actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        rows.append((lh, la, d1, d2, actual))
    packs[slug] = (model, rows)


def ll(slug, b, which):
    model, rows = packs[slug]
    return sum(-np.log(max(model.outcome_probs(lh * np.exp(b * (d1 if which == 1 else d2)),
                                               la * np.exp(-b * (d1 if which == 1 else d2)))[a], 1e-12))
               for lh, la, d1, d2, a in rows)


def loto(which):
    pool, n_tot = 0.0, 0
    for held, *_ in FOLDS:
        others = [s for s, *_ in FOLDS if s != held]
        b = minimize_scalar(lambda b: sum(ll(s, b, which) for s in others),
                            bounds=(-0.5, 0.5), method="bounded").x
        n = len(packs[held][1])
        pool += ll(held, b, which); n_tot += n
    return pool / n_tot


base = sum(ll(s, 0.0, 1) for s, *_ in FOLDS) / sum(len(packs[s][1]) for s, *_ in FOLDS)
ll_v1, ll_v2 = loto(1), loto(2)
print(f"\nPooled OOS log-loss over {sum(len(packs[s][1]) for s,*_ in FOLDS)} matches (3 folds):")
print(f"  no capital : {base:.4f}")
print(f"  v1 (equal) : {ll_v1:.4f}  ({ll_v1 - base:+.4f})")
print(f"  v2 (minutes): {ll_v2:.4f}  ({ll_v2 - base:+.4f})")
print(f"\nVerdict: {'v2 ADMITTED (replaces v1)' if ll_v2 < ll_v1 else 'v1 KEPT (minutes-weighting does not help)'}")
