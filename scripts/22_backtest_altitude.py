"""Step 22: admission gate for the altitude block, tested where altitude varies.

World Cups avoid altitude, so the 6-tournament backtest is blind to it. The
natural experiment is CONMEBOL World Cup qualifying: a home/away round robin
from La Paz (3782 m) and Quito (2854 m) to sea level. We test whether an
altitude-acclimatisation term adds predictive power beyond Elo + the model's
flat home advantage.

Feature (the user's framing): a team is acclimatised to its HABITUAL altitude
= the weighted-mean elevation of the cities where it plays at home. A side is
hurt in proportion to how far the venue sits ABOVE its habitual altitude:
    suffer(team) = max(0, venue_alt - habitual_alt(team)) / 1000   [km]
    d = suffer(away) - suffer(home)
applied as a tilt  lh *= exp(b*d), la *= exp(-b*d)  (the city-aware tilt
mechanism already in simulate.py). b fitted out-of-sample, 6 time-ordered
folds; admitted only if pooled OOS log-loss beats b = 0.
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from wc26.data import load_results
from wc26.dixon_coles import DixonColes

CONMEBOL = ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
            "Paraguay", "Peru", "Uruguay", "Venezuela"]
ALT = json.load(open("data/external/altitude.json"))

results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1").sort_values("date")

# Habitual altitude per team: frequency-weighted mean elevation of its home
# (non-neutral) cities across ALL history with a known elevation.
home = df[~df["neutral"].astype(bool)].dropna(subset=["city"])
home = home[home["city"].isin(ALT)]
habitual = (home.assign(alt=home["city"].map(ALT))
            .groupby("home_team")["alt"].mean())

q = df[(df["tournament"] == "FIFA World Cup qualification") & (df["date"] >= "2000-01-01")
       & df["home_team"].isin(CONMEBOL) & df["away_team"].isin(CONMEBOL)
       & df["city"].isin(ALT)].copy()
model = DixonColes().fit(df[df["date"] >= "2005-01-01"], pd.Timestamp("2026-06-11"))

rows = []
for r in q.itertuples(index=False):
    va = ALT[r.city]
    sh = max(0.0, va - habitual.get(r.home_team, va)) / 1000.0
    sa = max(0.0, va - habitual.get(r.away_team, va)) / 1000.0
    lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
    actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
    rows.append((r.date, lh, la, sa - sh, actual))
pack = pd.DataFrame(rows, columns=["date", "lh", "la", "d", "actual"]).sort_values("date").reset_index(drop=True)
print(f"CONMEBOL qualifiers with elevation: {len(pack)} | mean |d|: {pack.d.abs().mean():.2f} km")
print(f"matches with real altitude gap (|d|>0.5km): {(pack.d.abs() > 0.5).sum()}")


def ll_vec(sub, b):
    return np.array([-np.log(max(model.outcome_probs(lh * np.exp(b * d), la * np.exp(-b * d))[a], 1e-12))
                     for lh, la, d, a in zip(sub.lh, sub.la, sub.d, sub.actual)])


folds = np.array_split(pack.index.to_numpy(), 6)
pool0 = pool1 = 0.0
diffs = []
print(f"\n{'fold':5s} {'n':>4s} {'b* (others)':>12s} {'LL0':>8s} {'LL(b*)':>8s} {'delta':>8s}")
for i, fold in enumerate(folds):
    test = pack.loc[fold]
    train = pack.drop(fold)
    b = minimize_scalar(lambda b: ll_vec(train, b).sum(), bounds=(-1.5, 1.5), method="bounded").x
    l0, l1 = ll_vec(test, 0.0).mean(), ll_vec(test, b).mean()
    pool0 += l0 * len(test); pool1 += l1 * len(test)
    diffs.append(ll_vec(test, b) - ll_vec(test, 0.0))
    print(f"{i:5d} {len(test):4d} {b:12.3f} {l0:8.4f} {l1:8.4f} {l1 - l0:+8.4f}")

d = np.concatenate(diffs)
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
b_all = minimize_scalar(lambda b: ll_vec(pack, b).sum(), bounds=(-1.5, 1.5), method="bounded").x
n = len(pack)
print(f"\nPooled OOS log-loss: b=0 {pool0/n:.4f} vs altitude {pool1/n:.4f} ({(pool1-pool0)/n:+.4f}) | paired t = {t:.2f}")
print(f"b on all data: {b_all:.3f} per km  | Verdict: {'ADMITTED' if pool1 < pool0 else 'REJECTED'}")
json.dump({"beta_altitude_per_km": round(float(b_all), 4), "oos_delta": round(float((pool1-pool0)/n), 5),
           "t_paired": round(float(t), 2), "n": int(n),
           "verdict": "ADMITTED" if pool1 < pool0 else "REJECTED"},
          open("outputs/altitude_beta.json", "w"), indent=1)
