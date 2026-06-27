"""Step 35: altitude-AWARE Elo home advantage (the Ecuador residual).

Motivation
----------
The admitted altitude block (scripts/22) is a *match-time venue* effect:
lowland sides are damped at high-altitude venues. It does nothing for the
distinct problem that surfaced live with Ecuador: a habitual-altitude side
accumulates Elo by winning home qualifiers in Quito/Guayaquil, and the flat
100-point Elo home advantage does not know that edge is unusually large
*because* it is an altitude fortress. The rating inflates and then travels
with the team to a neutral, sea-level tournament where the fortress is gone.

Fix under test
--------------
Make the Elo home advantage altitude-aware. For a match at a venue of known
elevation, add to the home side's expected score a term proportional to how
much more the AWAY side suffers from the altitude than the home side:

    suffer(team, venue) = max(0, venue_alt - habitual_alt(team)) / 1000   [km]
    dr = R_home - R_away + HOME_ADV*(not neutral)
            + gamma * ( suffer(away) - suffer(home) )                     [Elo pts]

At a home altitude venue suffer(home)~0 and suffer(away) is large, so the
expected score rises and a home win earns *fewer* rating points -> the
altitude fortress stops inflating the rating. gamma is in Elo points per km
of altitude gap; gamma = 0 recovers the current model.

Gate
----
(A) Canonical, project-wide verdict: the 6-tournament pooled OOS log-loss
    (same protocol as scripts/06), recomputing Elo for each gamma.
(B) Power cut: World Cups, Euros and Copa Américas avoid altitude, so (A) is
    nearly blind to the effect (just as the tilt was). The natural experiment
    is the inflated sides playing AWAY FROM altitude. Test set = matches with
    a known sea-level-ish venue (alt < 1000 m) involving >=1 CONMEBOL team,
    scored on forward folds. This is where deflating altitude-Elo should pay.

Rule (project convention): a feature that does not improve backtest log-loss
is dropped.
"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from wc26.data import load_results
from wc26.dixon_coles import DixonColes
from wc26.elo import HOME_ADV, INITIAL_RATING, expected_score, goal_multiplier, k_factor

ALT = json.load(open("data/external/altitude.json"))
CONMEBOL = ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
            "Paraguay", "Peru", "Uruguay", "Venezuela"]
GRID = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0]  # Elo points per km of altitude gap

results = load_results().sort_values("date").reset_index(drop=True)


def habitual_altitude(df: pd.DataFrame) -> dict[str, float]:
    """Frequency-weighted mean elevation of each team's known home cities."""
    home = df[~df["neutral"].astype(bool)].dropna(subset=["city"])
    home = home[home["city"].isin(ALT)]
    alt = home.assign(a=home["city"].map(ALT)).groupby("home_team")["a"].mean()
    return alt.to_dict()


def suffer(team: str, venue_alt: float, habitual: dict[str, float]) -> float:
    return max(0.0, venue_alt - habitual.get(team, 0.0)) / 1000.0


def elo_history_alt(df: pd.DataFrame, gamma: float, habitual: dict[str, float]) -> pd.DataFrame:
    """Elo over full history with an altitude-aware home advantage."""
    played = df.dropna(subset=["home_score", "away_score"])
    ratings: dict[str, float] = {}
    rows = []
    for row in played.itertuples(index=False):
        rh = ratings.get(row.home_team, INITIAL_RATING)
        ra = ratings.get(row.away_team, INITIAL_RATING)
        va = ALT.get(row.city)
        d_alt = 0.0
        if gamma and va is not None:
            d_alt = gamma * (suffer(row.away_team, va, habitual) - suffer(row.home_team, va, habitual))
        # fold the altitude term into the rating gap (Elo-point space)
        we = expected_score(rh + d_alt, ra, bool(row.neutral))
        margin = int(row.home_score) - int(row.away_score)
        w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        delta = k_factor(row.tournament) * goal_multiplier(margin) * (w - we)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
        rows.append((row.date, row.home_team, row.away_team, rh, ra))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "elo_home_pre", "elo_away_pre"])


habitual = habitual_altitude(results)
print(f"habitual altitude known for {len(habitual)} teams; "
      f"CONMEBOL e.g. Ecuador={habitual.get('Ecuador',0):.0f}m Bolivia={habitual.get('Bolivia',0):.0f}m")

# ---------------------------------------------------------------- gate A
TOURNAMENTS = [
    ("WC2014", "FIFA World Cup", "2014-06-12", "2014-07-13"),
    ("WC2018", "FIFA World Cup", "2018-06-14", "2018-07-15"),
    ("WC2022", "FIFA World Cup", "2022-11-20", "2022-12-18"),
    ("Euro2016", "UEFA Euro", "2016-06-10", "2016-07-10"),
    ("Euro2021", "UEFA Euro", "2021-06-11", "2021-07-11"),
    ("Euro2024", "UEFA Euro", "2024-06-14", "2024-07-14"),
]


def merge_pre(df_played: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    key = ["date", "home_team", "away_team"]
    a = df_played.copy()
    a["dup"] = a.groupby(key).cumcount()
    h = hist.copy()
    h["dup"] = h.groupby(key).cumcount()
    return a.merge(h, on=key + ["dup"], validate="1:1")


def outcome_ll(model, sub) -> list[float]:
    ll = []
    for r in sub.itertuples(index=False):
        lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
        p = model.outcome_probs(lh, la)
        a = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        ll.append(-np.log(max(p[a], 1e-12)))
    return ll


played = results.dropna(subset=["home_score", "away_score"]).copy()
print("\n=== Gate A: canonical 6-tournament pooled OOS log-loss ===")
gateA = {}
for gamma in GRID:
    hist = elo_history_alt(results, gamma, habitual)
    df = merge_pre(played, hist).sort_values("date")
    tot, n = 0.0, 0
    for _name, comp, start, end in TOURNAMENTS:
        fit_date = pd.Timestamp(start)
        train = df[(df["date"] >= fit_date - pd.DateOffset(years=20)) & (df["date"] < fit_date)]
        test = df[(df["tournament"] == comp) & (df["date"] >= start) & (df["date"] <= end)]
        model = DixonColes().fit(train, fit_date)
        ll = outcome_ll(model, test)
        tot += float(np.sum(ll)); n += len(ll)
    gateA[gamma] = tot / n
    print(f"  gamma={gamma:6.0f}  pooled LL={tot/n:.4f}  (n={n})")
bestA = min(gateA, key=gateA.get)
print(f"  -> best gamma {bestA:.0f}; delta vs 0 = {gateA[bestA]-gateA[0.0]:+.4f}")

# ---------------------------------------------------------------- gate B
# CONMEBOL teams playing at sea-level-ish venues (alt < 1000m), forward folds.
print("\n=== Gate B: CONMEBOL sides away from altitude (power cut) ===")
FOLD_DATES = [pd.Timestamp(y) for y in
              ["2010-01-01", "2014-01-01", "2018-01-01", "2022-01-01", "2024-06-01"]]


def is_lowland(city):
    if city is None or (isinstance(city, float) and np.isnan(city)):
        return True  # unknown venue treated as sea level (no altitude help)
    return ALT.get(city, 0.0) < 1000.0


played["lowland"] = played["city"].apply(is_lowland)
played["conmebol"] = played["home_team"].isin(CONMEBOL) | played["away_team"].isin(CONMEBOL)

gateB = {}
diffs_vs0 = {g: [] for g in GRID if g != 0.0}
base_ll_store = None
for gamma in GRID:
    hist = elo_history_alt(results, gamma, habitual)
    df = merge_pre(played, hist).sort_values("date")
    test_all = df[df["conmebol"] & df["lowland"] & (df["date"] >= "2008-01-01")]
    per_match_ll = {}
    for i, fit_date in enumerate(FOLD_DATES):
        nxt = FOLD_DATES[i + 1] if i + 1 < len(FOLD_DATES) else pd.Timestamp("2027-01-01")
        train = df[(df["date"] >= fit_date - pd.DateOffset(years=20)) & (df["date"] < fit_date)]
        test = test_all[(test_all["date"] >= fit_date) & (test_all["date"] < nxt)]
        if len(test) == 0:
            continue
        model = DixonColes().fit(train, fit_date)
        for r, ll in zip(test.itertuples(index=False), outcome_ll(model, test)):
            per_match_ll[(r.date, r.home_team, r.away_team)] = ll
    gateB[gamma] = float(np.mean(list(per_match_ll.values())))
    print(f"  gamma={gamma:6.0f}  OOS LL={gateB[gamma]:.4f}  (n={len(per_match_ll)})")
    if gamma == 0.0:
        base_ll_store = per_match_ll
    else:
        keys = set(base_ll_store) & set(per_match_ll)
        diffs_vs0[gamma] = np.array([per_match_ll[k] - base_ll_store[k] for k in keys])

bestB = min(gateB, key=gateB.get)
dB = diffs_vs0.get(bestB, np.array([0.0]))
tB = dB.mean() / (dB.std(ddof=1) / np.sqrt(len(dB))) if len(dB) > 1 else 0.0
print(f"  -> best gamma {bestB:.0f}; delta vs 0 = {gateB[bestB]-gateB[0.0]:+.4f}  paired t={tB:.2f}")

# ---------------------------------------------------------------- mechanism
print("\n=== Mechanism: how much altitude-Elo deflates key sides at 2026-06-11 ===")
from wc26.elo import ratings_asof
for gamma in [0.0, bestB]:
    hist = elo_history_alt(results, gamma, habitual)
    r = ratings_asof(hist.rename(columns={"elo_home_pre": "elo_home_post",
                                          "elo_away_pre": "elo_away_post"}), "2026-06-11")
    tag = "baseline" if gamma == 0 else f"gamma={gamma:.0f}"
    print(f"  {tag:14s} " + "  ".join(
        f"{t}={r.get(t, float('nan')):.0f}" for t in ["Ecuador", "Bolivia", "Colombia", "Brazil", "Argentina"]))

verdict = "ADMITTED" if (gateB[bestB] < gateB[0.0] and bestB != 0.0) else "REJECTED"
json.dump({"grid": GRID, "gateA_pooled_ll": gateA, "gateB_oos_ll": gateB,
           "best_gamma_A": bestA, "best_gamma_B": bestB,
           "gateB_delta_vs0": round(float(gateB[bestB] - gateB[0.0]), 5),
           "gateB_paired_t": round(float(tB), 2), "verdict": verdict},
          open("outputs/altitude_elo_beta.json", "w"), indent=1)
print(f"\nVerdict (gate B is decisive): {verdict}")
