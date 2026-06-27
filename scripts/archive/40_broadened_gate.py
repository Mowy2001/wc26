"""Step 40: broadened gate — does the model generalise beyond UEFA? (Simone, 2026-06-28)

The admission gate has always been WC + Euro (target-resemblance), but that pool
is UEFA-heavy, so we have never checked whether the model — or its tuned
hyperparameters — hold up across the OTHER confederations. This adds a
"robustness gate" of the four continental cups and re-scores the Elo/DC-level
verdicts on it:

  PRIMARY  (target-like, decides admission) : FIFA World Cup, UEFA Euro
  ROBUST   (cross-confederation veto)        : Copa América (CONMEBOL),
            African Cup of Nations (CAF), AFC Asian Cup (AFC), Gold Cup (CONCACAF)

Editions are detected automatically (a >60-day gap starts a new edition); each
edition is fit point-in-time on the 20 years before its opening day and scored
on its own matches (predictive outcome log-loss). Nations League is deliberately
EXCLUDED (home/away, rotated squads — too noisy for a feature gate).

Outputs: per-confederation base calibration, the xi plateau re-checked on ROBUST,
and the deployed altitude-in-Elo (gamma=75 vs 0) re-checked on ROBUST.
"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from wc26.data import load_results
from wc26.dixon_coles import DixonColes
from wc26.elo import ALT_GAMMA, compute_elo_history, habitual_altitudes

ALT = json.load(open("data/external/altitude.json"))
results = load_results().sort_values("date").reset_index(drop=True)
habitual = habitual_altitudes(results, ALT)

# competition substring -> (label, confederation, set)
COMPS = [
    ("FIFA World Cup", "World Cup", "MIXED", "PRIMARY"),
    ("UEFA Euro", "Euro", "UEFA", "PRIMARY"),
    ("Copa América", "Copa América", "CONMEBOL", "ROBUST"),
    ("African Cup of Nations", "AFCON", "CAF", "ROBUST"),
    ("AFC Asian Cup", "Asian Cup", "AFC", "ROBUST"),
    ("Gold Cup", "Gold Cup", "CONCACAF", "ROBUST"),
]


def editions(sub):
    """List of (start, end, match_index) editions for a competition substring."""
    m = results[results["tournament"].str.contains(sub, case=False, na=False)
                & ~results["tournament"].str.contains("qualif", case=False, na=False)
                & results["home_score"].notna()].copy()
    m = m[(m["date"] >= "2004-01-01") & (m["date"] <= "2026-06-22")]  # exclude the live 2026 WC tail
    if len(m) == 0:
        return []
    m = m.sort_values("date")
    gap = m["date"].diff().dt.days.fillna(999) > 60
    m["ed"] = gap.cumsum()
    out = []
    for _, g in m.groupby("ed"):
        if len(g) >= 8:  # a real finals tournament, not stray matches
            out.append((g["date"].min(), g["date"].max(), g.index))
    return out


def merge_pre(hist):
    key = ["date", "home_team", "away_team"]
    played = results.dropna(subset=["home_score", "away_score"]).copy()
    played["dup"] = played.groupby(key).cumcount()
    hist = hist.copy(); hist["dup"] = hist.groupby(key).cumcount()
    return played.merge(hist, on=key + ["dup"], validate="1:1").sort_values("date")


def edition_ll(df, comp_sub, start, end, xi=0.0027):
    fit = pd.Timestamp(start)
    train = df[(df["date"] >= fit - pd.DateOffset(years=20)) & (df["date"] < fit)]
    test = df[(df["tournament"].str.contains(comp_sub, case=False, na=False))
              & ~df["tournament"].str.contains("qualif", case=False, na=False)
              & (df["date"] >= start) & (df["date"] <= end)]
    model = DixonColes(xi=xi).fit(train, fit)
    lls = []
    for r in test.itertuples(index=False):
        lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
        p = model.outcome_probs(lh, la)
        a = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        lls.append(-np.log(max(p[a], 1e-12)))
    return lls


# ---- base calibration per confederation (deployed altitude-aware Elo) ----
hist_dep = compute_elo_history(results, alt_gamma=ALT_GAMMA, habitual=habitual, city_alt=ALT)
df_dep = merge_pre(hist_dep)
print("=== Base Elo-DC calibration by competition (deployed model) ===")
print(f"{'competition':16s}{'conf':10s}{'eds':>4s}{'n':>5s}{'LL':>8s}{'vs unif':>9s}")
set_pool = {"PRIMARY": [], "ROBUST": []}
conf_pool = {}
for sub, label, conf, which in COMPS:
    eds = editions(sub)
    lls = []
    for (s, e, _idx) in eds:
        lls += edition_ll(df_dep, sub, s, e)
    if not lls:
        continue
    ll = float(np.mean(lls))
    set_pool[which] += lls
    conf_pool.setdefault(conf, []).extend(lls)
    print(f"{label:16s}{conf:10s}{len(eds):>4d}{len(lls):>5d}{ll:>8.4f}{1.0986-ll:>+9.4f}")
print("  (vs unif = how much better than the 1/3-1/3-1/3 baseline; higher = model adds more)")
for s in ["PRIMARY", "ROBUST"]:
    p = set_pool[s]
    print(f"  {s:8s} pooled LL {np.mean(p):.4f}  (n={len(p)}, uniform 1.0986)")

# ---- xi plateau re-checked on each set ----
print("\n=== xi re-checked: pooled OOS LL on PRIMARY vs ROBUST ===")
GRID = [0.0010, 0.0018, 0.0027, 0.0040, 0.0050]
robust_eds = [(sub, s, e) for sub, _l, _c, w in COMPS if w == "ROBUST" for (s, e, _i) in editions(sub)]
prim_eds = [(sub, s, e) for sub, _l, _c, w in COMPS if w == "PRIMARY" for (s, e, _i) in editions(sub)]
for xi in GRID:
    lp = [x for sub, s, e in prim_eds for x in edition_ll(df_dep, sub, s, e, xi=xi)]
    lr = [x for sub, s, e in robust_eds for x in edition_ll(df_dep, sub, s, e, xi=xi)]
    print(f"  xi={xi:.4f}  PRIMARY {np.mean(lp):.4f}   ROBUST {np.mean(lr):.4f}")

# ---- altitude-in-Elo (gamma 75 vs 0) re-checked on ROBUST ----
print("\n=== altitude-in-Elo on ROBUST: deployed gamma=75 vs gamma=0 ===")
hist0 = compute_elo_history(results, alt_gamma=0.0, habitual=habitual, city_alt=ALT)
df0 = merge_pre(hist0)
base = {sub: edition_ll(df0, sub, s, e) for sub, s, e in robust_eds}
dep = {sub: edition_ll(df_dep, sub, s, e) for sub, s, e in robust_eds}
b_all = [x for v in base.values() for x in v]
d_all = [x for v in dep.values() for x in v]
diff = np.array(d_all) - np.array(b_all)
t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
print(f"  gamma=0  {np.mean(b_all):.4f}   gamma=75 {np.mean(d_all):.4f}  "
      f"({np.mean(d_all)-np.mean(b_all):+.4f}, paired t={t:.2f}, n={len(diff)})")
print("  (negative delta = altitude-in-Elo also helps outside Europe)")

out = {"set_pool_ll": {k: round(float(np.mean(v)), 4) for k, v in set_pool.items()},
       "conf_ll": {k: round(float(np.mean(v)), 4) for k, v in conf_pool.items()},
       "altitude_robust_delta": round(float(np.mean(d_all) - np.mean(b_all)), 5),
       "altitude_robust_t": round(float(t), 2), "n_robust": len(d_all)}
json.dump(out, open("outputs/broadened_gate.json", "w"), indent=1)
print("\noutputs/broadened_gate.json written")
