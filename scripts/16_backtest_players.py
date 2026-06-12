"""Step 16: admission gate for player-layer v2 (squad filter + age discount).

Test: for WC2014/18/22, build allocation weights strictly as of each
tournament's eve, then score the REALISED within-team goal split with the
multinomial log-likelihood (per goal): each realised goal by player i
contributes log p_i; goals by scorers outside the weight table contribute
log p_debutant. Lower is better, exactly comparable across variants since
the realised splits are fixed.

Variants: v1 (activity-window proxy) vs v2 (official squad filter) vs
v2 + age discount over a grid (leave-one-tournament-out for the choice).
Debutant share is point-in-time: mean of the three World Cups before the
one being scored.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_goalscorers
from wc26.players import estimate_debutant_share, scorer_weights, squad_weights, DEBUTANT_KEY, _norm_name

WCS = [("wc2014", "2014-06-12", "2014-07-13"),
       ("wc2018", "2018-06-14", "2018-07-15"),
       ("wc2022", "2022-11-20", "2022-12-18")]
PRIOR = {"wc2014": [("2002-05-31", "2002-06-30"), ("2006-06-09", "2006-07-09"), ("2010-06-11", "2010-07-11")],
         "wc2018": [("2006-06-09", "2006-07-09"), ("2010-06-11", "2010-07-11"), ("2014-06-12", "2014-07-13")],
         "wc2022": [("2010-06-11", "2010-07-11"), ("2014-06-12", "2014-07-13"), ("2018-06-14", "2018-07-15")]}

gs = load_goalscorers()
ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.25]
TAUS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]


def temper(w: dict, tau: float) -> dict:
    """Flatten the named shares: p_i ∝ p_i^tau, bucket mass preserved."""
    from wc26.players import DEBUTANT_KEY
    deb = w.get(DEBUTANT_KEY, 0.0)
    named = {k: v for k, v in w.items() if k != DEBUTANT_KEY}
    if not named:
        return w
    t = {k: v ** tau for k, v in named.items()}
    z = sum(t.values())
    out = {k: v / z * (1.0 - deb) for k, v in t.items()}
    out[DEBUTANT_KEY] = deb
    return out


def realised(slug, start, end):
    """{team: {scorer: goals}} in that WC, own goals excluded."""
    wc = gs[(gs["date"] >= start) & (gs["date"] <= end) & (~gs["own_goal"].astype(bool))].dropna(subset=["scorer"])
    return {t: g["scorer"].value_counts().to_dict() for t, g in wc.groupby("team")}


def ll_per_goal(weights_fn, slug, start, end):
    """Mean negative log-likelihood per realised goal under a weights builder."""
    asof = pd.Timestamp(start)
    deb = estimate_debutant_share(gs, PRIOR[slug])
    out, n = 0.0, 0
    for team, scored in realised(slug, start, end).items():
        w = weights_fn(team, asof, deb)
        norm_w = {_norm_name(k): v for k, v in w.items()}
        for scorer, goals in scored.items():
            p = norm_w.get(_norm_name(scorer), w.get(DEBUTANT_KEY, 1e-6))
            out -= goals * np.log(max(p, 1e-9))
            n += goals
    return out / n


rows = []
for slug, start, end in WCS:
    squad = pd.read_csv(f"data/external/squads_{slug}.csv", parse_dates=["birth"])
    v1 = ll_per_goal(lambda t, a, d: scorer_weights(gs, t, a, d), slug, start, end)
    row = {"wc": slug, "v1": v1}
    for alpha in ALPHAS:
        row[f"v2_a{alpha}"] = ll_per_goal(
            lambda t, a, d, al=alpha: squad_weights(gs, squad, t, a, d, age_alpha=al,
                                                    drop_to_bucket=False), slug, start, end)
        row[f"v2b_a{alpha}"] = ll_per_goal(
            lambda t, a, d, al=alpha: squad_weights(gs, squad, t, a, d, age_alpha=al), slug, start, end)
    for tau in TAUS:
        row[f"v3_t{tau}"] = ll_per_goal(
            lambda t, a, d, tu=tau: temper(scorer_weights(gs, t, a, d), tu), slug, start, end)
    rows.append(row)

df = pd.DataFrame(rows).set_index("wc")
print(df.round(4).to_string())
print("\nmean:", df.mean().round(4).to_dict())

# LOTO choice of alpha among v2 variants
picks = []
for held in df.index:
    others = df.drop(held)
    best = others[[c for c in df.columns if not c == "v1"]].mean().idxmin()
    picks.append((held, best, float(df.loc[held, best]), float(df.loc[held, "v1"])))
print("\nLOTO: held-out WC | alpha chosen on others | LL(v2*) vs LL(v1)")
for h, b, l2, l1 in picks:
    print(f"  {h}: {b}  {l2:.4f} vs {l1:.4f}  ({l2 - l1:+.4f})")
mean_v2 = np.mean([p[2] for p in picks]); mean_v1 = np.mean([p[3] for p in picks])
verdict = "challenger ADMITTED" if mean_v2 < mean_v1 else "v1 KEPT"
print(f"\nPooled per-goal LL: challenger* {mean_v2:.4f} vs v1 {mean_v1:.4f} -> {verdict}")
