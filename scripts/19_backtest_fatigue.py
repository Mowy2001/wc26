"""Step 19: admission gate for the fatigue block (backlog #5d).

Feature: squad mean club-season minutes (FBref, pre-tournament season),
matched players only (Big-5 coverage bias declared, same as capital),
z-scored within tournament. Tested as a lambda tilt on top of the admitted
capital baseline, LOTO across the two folds with point-in-time minutes
pages (WC2018 <- 2017-18, WC2022 <- 2021-22 captured Aug 2022).

Declared limitation: for the November 2022 fold the load proxy is the
PREVIOUS full season — Aug-Nov 2022 club minutes existed but no
point-in-time snapshot of them survives.
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from wc26.data import load_results
from wc26.dixon_coles import DixonColes
from wc26.players import _norm_name

FOLDS = [("wc2018", "FIFA World Cup", "2018-06-14", "2018-07-15", "fbref_2017_18"),
         ("wc2022", "FIFA World Cup", "2022-11-20", "2022-12-18", "fbref_2021_22")]

cap = pd.read_csv("outputs/capital.csv")
B_CAP = json.load(open("outputs/capital_beta.json"))["beta_capital"]


def fatigue_z(slug, fbslug):
    squad = pd.read_csv(f"data/external/squads_{slug}.csv")
    fb = pd.read_csv(f"data/external/{fbslug}.csv")
    mins = {_norm_name(p): m for p, m in zip(fb.player, fb.minutes)}
    rows = []
    for team, g in squad.groupby("team"):
        matched = [mins[_norm_name(p)] for p in g.player if _norm_name(p) in mins]
        rows.append({"team": team, "load": np.mean(matched) if matched else np.nan,
                     "coverage": len(matched) / len(g)})
    df = pd.DataFrame(rows)
    df["load"] = df["load"].fillna(df["load"].mean())
    df["z"] = (df.load - df.load.mean()) / df.load.std(ddof=0)
    return df.set_index("team")["z"], float(df.coverage.mean())


results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1").sort_values("date")

packs = {}
for slug, comp, start, end, fbslug in FOLDS:
    fd = pd.Timestamp(start)
    model = DixonColes().fit(df[df["date"] >= fd - pd.DateOffset(years=20)], fd)
    test = df[(df["tournament"] == comp) & (df["date"] >= start) & (df["date"] <= end)]
    fz, cov = fatigue_z(slug, fbslug)
    cz = cap[cap.tournament == slug].set_index("team")["capital_z"]
    rows = []
    for r in test.itertuples(index=False):
        lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
        dcap = float(cz.get(r.home_team, 0.0) - cz.get(r.away_team, 0.0))
        lh, la = lh * np.exp(B_CAP * dcap), la * np.exp(-B_CAP * dcap)
        d = float(fz.get(r.home_team, 0.0) - fz.get(r.away_team, 0.0))
        actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        rows.append((lh, la, d, actual))
    packs[slug] = (model, rows)
    print(f"{slug}: minutes coverage {cov:.0%}")


def ll(slug, b):
    model, rows = packs[slug]
    return sum(-np.log(max(model.outcome_probs(lh * np.exp(b * d), la * np.exp(-b * d))[a], 1e-12))
               for lh, la, d, a in rows)


print(f"\n{'held out':8s} {'b* (other)':>11s} {'LL(base)':>9s} {'LL(b*)':>9s} {'delta':>8s}")
deltas = []
for held, *_ in FOLDS:
    other = [s for s, *_ in FOLDS if s != held][0]
    b_star = minimize_scalar(lambda b: ll(other, b), bounds=(-0.5, 0.5), method="bounded").x
    n = len(packs[held][1])
    l0, l1 = ll(held, 0.0) / n, ll(held, b_star) / n
    deltas.append(l1 - l0)
    print(f"{held:8s} {b_star:11.4f} {l0:9.4f} {l1:9.4f} {l1 - l0:+8.4f}")
verdict = "ADMITTED" if np.mean(deltas) < 0 else "REJECTED"
print(f"\nMean OOS delta: {np.mean(deltas):+.4f} -> {verdict}")
b_all = minimize_scalar(lambda b: ll("wc2018", b) + ll("wc2022", b),
                        bounds=(-0.5, 0.5), method="bounded").x
print(f"b on both folds: {b_all:.4f}")
json.dump({"beta_fatigue": round(float(b_all), 4), "oos_delta": round(float(np.mean(deltas)), 5),
           "verdict": verdict, "n_folds": 2}, open("outputs/fatigue_beta.json", "w"), indent=1)

# 2026 feature: 2025-26 minutes (snapshot 2026-03-09, ~3/4 season — declared)
fz26, cov26 = fatigue_z("wc2026", "fbref_2025_26")
fz26.round(4).rename("fatigue_z").to_csv("outputs/fatigue.csv")
print(f"wc2026 fatigue feature written (coverage {cov26:.0%})")
