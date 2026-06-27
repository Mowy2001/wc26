"""Step 6: tune the time-decay hyperparameter xi on a multi-tournament backtest.

Protocol (point-in-time, out-of-sample): for each tournament T in
{WC2014, WC2018, WC2022, Euro2016, Euro2021, Euro2024} and each xi on the
grid, fit the DC model on all matches in the 20 years before T's opening
day, then score predictive outcome log-loss (H/D/A at 90') on T's matches.
xi is chosen by pooled log-loss across all six tournaments. The 2026 model
never sees data later than its own fit date, so tuning on past tournaments
is leakage-free.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_results
from wc26.dixon_coles import DixonColes

TOURNAMENTS = [
    ("WC2014",   "FIFA World Cup", "2014-06-12", "2014-07-13"),
    ("WC2018",   "FIFA World Cup", "2018-06-14", "2018-07-15"),
    ("WC2022",   "FIFA World Cup", "2022-11-20", "2022-12-18"),
    ("Euro2016", "UEFA Euro",      "2016-06-10", "2016-07-10"),
    ("Euro2021", "UEFA Euro",      "2021-06-11", "2021-07-11"),
    ("Euro2024", "UEFA Euro",      "2024-06-14", "2024-07-14"),
]
GRID = [0.0005, 0.0010, 0.0015, 0.0018, 0.0022, 0.0027, 0.0033, 0.0040, 0.0050]

results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1").sort_values("date")

rows = []
for name, comp, start, end in TOURNAMENTS:
    fit_date = pd.Timestamp(start)
    train = df[df["date"] >= fit_date - pd.DateOffset(years=20)]
    test = df[(df["tournament"] == comp) & (df["date"] >= start) & (df["date"] <= end)]
    for xi in GRID:
        model = DixonColes(xi=xi).fit(train, fit_date)
        ll = []
        for r in test.itertuples(index=False):
            lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
            p = model.outcome_probs(lh, la)
            actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            ll.append(-np.log(max(p[actual], 1e-12)))
        rows.append({"tournament": name, "xi": xi, "n": len(test), "ll": float(np.mean(ll))})
        print(f"{name}  xi={xi:.4f}  n={len(test)}  LL={np.mean(ll):.4f}")

res = pd.DataFrame(rows)
res.to_csv("outputs/xi_tuning.csv", index=False)
pooled = res.assign(tot=res.ll * res.n).groupby("xi").apply(
    lambda g: g.tot.sum() / g.n.sum(), include_groups=False
)
print("\nPooled log-loss by xi:")
print(pooled.round(4).to_string())
best = pooled.idxmin()
print(f"\nBest xi: {best}  (pooled LL {pooled[best]:.4f}; current default 0.0018 -> {pooled[0.0018]:.4f})")
