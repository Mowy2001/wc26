"""Step 12: admission backtest for the football-capital block.

Residual design: the DC lambdas get a multiplicative tilt
    lh' = lh * exp(+b * capdiff),  la' = la * exp(-b * capdiff)
with capdiff = capital_z(home) - capital_z(away). The coefficient b is the
only new parameter.

Protocol (leave-one-tournament-out): for each of the 6 past tournaments,
fit b on the other 5 by predictive log-loss, evaluate on the held-out one.
Admission rule (CLAUDE.md): pooled OOS log-loss must improve vs b = 0.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from wc26.data import load_results
from wc26.dixon_coles import DixonColes

TOURNAMENTS = [
    ("wc2014",   "FIFA World Cup", "2014-06-12", "2014-07-13"),
    ("wc2018",   "FIFA World Cup", "2018-06-14", "2018-07-15"),
    ("wc2022",   "FIFA World Cup", "2022-11-20", "2022-12-18"),
    ("euro2016", "UEFA Euro",      "2016-06-10", "2016-07-10"),
    ("euro2020", "UEFA Euro",      "2021-06-11", "2021-07-11"),
    ("euro2024", "UEFA Euro",      "2024-06-14", "2024-07-14"),
]

cap = pd.read_csv("outputs/capital.csv")
results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1").sort_values("date")

# Per tournament: precompute base lambdas and capdiff per match
packs = {}
for slug, comp, start, end in TOURNAMENTS:
    fd = pd.Timestamp(start)
    model = DixonColes().fit(df[df["date"] >= fd - pd.DateOffset(years=20)], fd)
    test = df[(df["tournament"] == comp) & (df["date"] >= start) & (df["date"] <= end)]
    cz = cap[cap.tournament == slug].set_index("team")["capital_z"]
    missing = (set(test.home_team) | set(test.away_team)) - set(cz.index)
    if missing:
        print(f"WARNING {slug}: no capital for {sorted(missing)}")
    rows = []
    for r in test.itertuples(index=False):
        lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
        d = float(cz.get(r.home_team, 0.0) - cz.get(r.away_team, 0.0))
        actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        rows.append((lh, la, d, actual))
    packs[slug] = (model, rows)


def ll_vec(slug, b):
    model, rows = packs[slug]
    return np.array([-np.log(max(model.outcome_probs(lh * np.exp(b * d), la * np.exp(-b * d))[actual], 1e-12))
                     for lh, la, d, actual in rows])


def ll(slug, b):
    return float(ll_vec(slug, b).sum())


print(f"\n{'held out':10s} {'b* (5 others)':>14s} {'LL(b=0)':>9s} {'LL(b*)':>9s} {'delta':>8s}")
pool0 = pool1 = n_tot = 0.0
for held, *_ in TOURNAMENTS:
    others = [s for s, *_ in TOURNAMENTS if s != held]
    b_star = minimize_scalar(lambda b: sum(ll(s, b) for s in others),
                             bounds=(-0.5, 0.5), method="bounded").x
    n = len(packs[held][1])
    l0, l1 = ll(held, 0.0) / n, ll(held, b_star) / n
    pool0 += l0 * n; pool1 += l1 * n; n_tot += n
    print(f"{held:10s} {b_star:14.4f} {l0:9.4f} {l1:9.4f} {l1 - l0:+8.4f}")

print(f"\nPooled OOS log-loss: b=0 {pool0 / n_tot:.4f} vs capital {pool1 / n_tot:.4f} "
      f"({(pool1 - pool0) / n_tot:+.4f}) over {int(n_tot)} matches")
# paired per-match significance of the OOS gain
diffs = []
for held, *_ in TOURNAMENTS:
    others = [s for s, *_ in TOURNAMENTS if s != held]
    b_star = minimize_scalar(lambda b: sum(ll(s, b) for s in others),
                             bounds=(-0.5, 0.5), method="bounded").x
    diffs.append(ll_vec(held, b_star) - ll_vec(held, 0.0))
d = np.concatenate(diffs)
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print(f"Paired OOS per-match delta: mean {d.mean():+.5f}, t = {t:.2f}")
b_all = minimize_scalar(lambda b: sum(ll(s, b) for s, *_ in TOURNAMENTS),
                        bounds=(-0.5, 0.5), method="bounded").x
print(f"b fitted on all six tournaments: {b_all:.4f}")
verdict = "ADMITTED" if pool1 < pool0 else "REJECTED"
print(f"Verdict per the admission rule: {verdict}")
import json
json.dump({"beta_capital": round(float(b_all), 4), "oos_delta": round(float((pool1 - pool0) / n_tot), 5),
           "t_paired": round(float(t), 2), "n": int(n_tot), "verdict": verdict},
          open("outputs/capital_beta.json", "w"), indent=1)
print("outputs/capital_beta.json written")
