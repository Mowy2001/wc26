"""Step 37: are the Elo weights identified? (Simone, 2026-06-25)

The Elo update has several weights we inherited wholesale from eloratings.net
and never tuned on our own predictive backtest:
  * HOME_ADV         -- the rating-update home advantage (default 100 pts)
  * the K tiers       -- 60/50/40/30/20 by tournament importance
  * the goal-diff multiplier G

We tune them the same way we tuned xi (scripts/06): recompute the WHOLE Elo
history for each weight setting, refit the Dixon-Coles point-in-time before
each of six past tournaments, and score pooled out-of-sample outcome log-loss
on those tournaments. We grid ONE factor at a time around the deployed config
(home_adv=100, global K mult=1.0, friendly K=20), keeping the deployed
altitude-aware home advantage (ALT_GAMMA) fixed so we tune *around* the live
model. A weight is only worth changing if it moves pooled OOS LL beyond noise
(paired t), not just in-sample.

Honest prior (from the xi result): global learning rate trades off against the
DC's refitted beta_elo and is likely unidentified; HOME_ADV and the
friendly-vs-competitive ratio are the plausibly identifiable ones.
"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from wc26.data import load_results
from wc26.dixon_coles import DixonColes
from wc26.elo import (ALT_GAMMA, INITIAL_RATING, K_RULES, K_DEFAULT,
                      expected_score, goal_multiplier, habitual_altitudes)

ALT = json.load(open("data/external/altitude.json"))
results = load_results().sort_values("date").reset_index(drop=True)
habitual = habitual_altitudes(results, ALT)


def k_factor_w(tournament: str, k_mult: float, friendly_k: float) -> float:
    t = str(tournament).lower()
    for key, k in K_RULES:
        if key in t:
            return (friendly_k if key == "friendly" else k) * k_mult
    return K_DEFAULT * k_mult


def elo_history_w(home_adv: float, k_mult: float, friendly_k: float) -> pd.DataFrame:
    """Full-history Elo with tunable weights + the deployed altitude term."""
    played = results.dropna(subset=["home_score", "away_score"])
    ratings: dict[str, float] = {}
    rows = []
    for row in played.itertuples(index=False):
        rh = ratings.get(row.home_team, INITIAL_RATING)
        ra = ratings.get(row.away_team, INITIAL_RATING)
        d_alt = 0.0
        va = ALT.get(getattr(row, "city", None))
        if ALT_GAMMA and va is not None:
            sh = max(0.0, va - habitual.get(row.home_team, 0.0)) / 1000.0
            sa = max(0.0, va - habitual.get(row.away_team, 0.0)) / 1000.0
            d_alt = ALT_GAMMA * (sa - sh)
        # custom home advantage: fold (home_adv - default) is implicit because
        # expected_score adds HOME_ADV; we instead bypass it with neutral=True
        # and add the whole home term ourselves.
        dr_home = (0.0 if bool(row.neutral) else home_adv) + d_alt
        we = expected_score(rh + dr_home, ra, neutral=True)
        margin = int(row.home_score) - int(row.away_score)
        w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        delta = k_factor_w(row.tournament, k_mult, friendly_k) * goal_multiplier(margin) * (w - we)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
        rows.append((row.date, row.home_team, row.away_team, rh, ra))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "elo_home_pre", "elo_away_pre"])


TOURNAMENTS = [
    ("WC2014", "FIFA World Cup", "2014-06-12", "2014-07-13"),
    ("WC2018", "FIFA World Cup", "2018-06-14", "2018-07-15"),
    ("WC2022", "FIFA World Cup", "2022-11-20", "2022-12-18"),
    ("Euro2016", "UEFA Euro", "2016-06-10", "2016-07-10"),
    ("Euro2021", "UEFA Euro", "2021-06-11", "2021-07-11"),
    ("Euro2024", "UEFA Euro", "2024-06-14", "2024-07-14"),
]
played = results.dropna(subset=["home_score", "away_score"]).copy()


def per_match_ll(home_adv, k_mult, friendly_k):
    """Point-in-time OOS outcome log-loss for every tournament match."""
    hist = elo_history_w(home_adv, k_mult, friendly_k)
    key = ["date", "home_team", "away_team"]
    a = played.copy(); a["dup"] = a.groupby(key).cumcount()
    hist["dup"] = hist.groupby(key).cumcount()
    df = a.merge(hist, on=key + ["dup"], validate="1:1").sort_values("date")
    out = {}
    for _n, comp, start, end in TOURNAMENTS:
        fit = pd.Timestamp(start)
        train = df[(df["date"] >= fit - pd.DateOffset(years=20)) & (df["date"] < fit)]
        test = df[(df["tournament"] == comp) & (df["date"] >= start) & (df["date"] <= end)]
        model = DixonColes().fit(train, fit)
        for r in test.itertuples(index=False):
            lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
            p = model.outcome_probs(lh, la)
            act = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            out[(r.date, r.home_team, r.away_team)] = -np.log(max(p[act], 1e-12))
    return out


DEFAULT = dict(home_adv=100.0, k_mult=1.0, friendly_k=20.0)
base = per_match_ll(**DEFAULT)
base_ll = float(np.mean(list(base.values())))
print(f"deployed config {DEFAULT}: pooled OOS LL = {base_ll:.4f}  (n={len(base)})\n")

AXES = {
    "home_adv": [40, 70, 100, 130, 160],
    "k_mult": [0.6, 0.8, 1.0, 1.25, 1.5],
    "friendly_k": [5, 10, 20, 30, 40],
}
summary = {"deployed_ll": round(base_ll, 4), "n": len(base), "axes": {}}
for axis, grid in AXES.items():
    print(f"=== {axis} (others at deployed default) ===")
    rows = []
    for v in grid:
        cfg = dict(DEFAULT); cfg[axis] = float(v)
        if cfg == DEFAULT:
            ll, t = base_ll, 0.0
        else:
            pm = per_match_ll(**cfg)
            keys = set(base) & set(pm)
            d = np.array([pm[k] - base[k] for k in keys])
            ll = float(np.mean(list(pm.values())))
            t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        rows.append({"v": v, "ll": round(ll, 4), "t_vs_default": round(t, 2)})
        star = "  <-- default" if cfg == DEFAULT else ""
        print(f"  {axis}={v:>6}: LL={ll:.4f}  t_vs_default={t:+.2f}{star}")
    best = min(rows, key=lambda r: r["ll"])
    print(f"  best {axis}={best['v']} (LL {best['ll']}, delta {best['ll']-base_ll:+.4f}, "
          f"t={best['t_vs_default']:+.2f})\n")
    summary["axes"][axis] = {"grid": rows, "best": best}

json.dump(summary, open("outputs/elo_weights_grid.json", "w"), indent=1)
print("Reading: an axis is IDENTIFIED only if its best beats the default beyond noise")
print("(|t| >~ 2). Otherwise it is a flat plateau (xi-style) and we keep the convention.")
