"""Step 18: admission gate for xG-blended allocation weights (FBref block).

Same metric as scripts/16 (per-goal multinomial LL on realised scorer
splits), folds limited to the xG era: WC2018 (2017-18 season) and WC2022
(2021-22 season, snapshot captured 2022-08 — before the tournament).
lam = weight of the npxG component, chosen leave-one-out across folds.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_goalscorers
from wc26.players import estimate_debutant_share, scorer_weights, xg_blend_weights, DEBUTANT_KEY, _norm_name

FOLDS = [("wc2018", "2018-06-14", "2018-07-15", "fbref_2017_18"),
         ("wc2022", "2022-11-20", "2022-12-18", "fbref_2021_22")]
PRIOR = {"wc2018": [("2006-06-09", "2006-07-09"), ("2010-06-11", "2010-07-11"), ("2014-06-12", "2014-07-13")],
         "wc2022": [("2010-06-11", "2010-07-11"), ("2014-06-12", "2014-07-13"), ("2018-06-14", "2018-07-15")]}
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]

gs = load_goalscorers()


def ll_per_goal(weights_fn, start, end, slug, squad):
    """Per-goal NLL. The bucket mass is SPLIT among squad members without a
    named share (a goal by an unlisted scorer is one of K interchangeable
    new faces, not a single pseudo-player — otherwise an all-bucket model
    would score perfectly)."""
    asof = pd.Timestamp(start)
    deb = estimate_debutant_share(gs, PRIOR[slug])
    wc = gs[(gs["date"] >= start) & (gs["date"] <= end) & (~gs["own_goal"].astype(bool))].dropna(subset=["scorer"])
    out, n = 0.0, 0
    for team, g in wc.groupby("team"):
        w = weights_fn(team, asof, deb)
        norm_w = {_norm_name(k): v for k, v in w.items() if k != DEBUTANT_KEY}
        roster = {_norm_name(p) for p in squad[squad["team"] == team]["player"]}
        k_new = max(1, len(roster - set(norm_w)))
        p_new = w.get(DEBUTANT_KEY, 1e-6) / k_new
        for scorer, goals in g["scorer"].value_counts().items():
            p = norm_w.get(_norm_name(scorer), p_new)
            out -= goals * np.log(max(p, 1e-9))
            n += goals
    return out / n


rows = {}
for slug, start, end, fbslug in FOLDS:
    squad = pd.read_csv(f"data/external/squads_{slug}.csv")
    fbref = pd.read_csv(f"data/external/{fbslug}.csv")
    # base = the gate-16 winner (squad filter + age discount alpha=0.1)
    row = {"base": ll_per_goal(
        lambda t, a, d: xg_blend_weights(gs, squad, fbref, t, a, d, 0.0, age_alpha=0.1),
        start, end, slug, squad)}
    for lam in LAMS:
        row[f"lam{lam}"] = ll_per_goal(
            lambda t, a, d, L=lam: xg_blend_weights(gs, squad, fbref, t, a, d, L, age_alpha=0.1),
            start, end, slug, squad)
    rows[slug] = row
df = pd.DataFrame(rows).T
print(df.round(4).to_string())

print("\nLeave-one-out: held fold | lam from other fold | LL vs v1")
deltas = []
for held in df.index:
    other = df.drop(held)
    best = other[[c for c in df.columns if c.startswith("lam")]].mean().idxmin()
    deltas.append(float(df.loc[held, best] - df.loc[held, "base"]))
    print(f"  {held}: {best}  {df.loc[held, best]:.4f} vs base {df.loc[held, 'base']:.4f} ({deltas[-1]:+.4f})")
verdict = "club-form blend ADMITTED on top" if np.mean(deltas) < 0 else "squad+age base KEPT"
print(f"\nMean OOS delta: {np.mean(deltas):+.4f} -> {verdict}")
