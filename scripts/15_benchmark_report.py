"""Step 15: three-way benchmark report (backlog #9).

Eve-of-tournament: model champion distribution vs BetMGM de-vigged with
Shin's method vs Klement's point forecast. Live: as played 2026 matches
appear in results.csv, score the model's frozen pre-match probabilities
(log-loss + favourite calibration) — the high-n test that actually ranks
forecasters; the champion market is scored once, after the final.

Assumption, declared: BetMGM quotes 9 teams; the unquoted field's true
mass is taken from the model (printed). Klement is a point forecast and
cannot be log-scored pre-emptively; he is tracked on his three calls
(Netherlands champions, final vs Portugal, England & Spain out in semis).
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes
from wc26.benchmark import shin_probs, implied_raw, log_score

BETMGM = {"Spain": 450, "France": 500, "England": 700, "Brazil": 800,
          "Portugal": 900, "Argentina": 900, "Germany": 1400,
          "Netherlands": 2000, "United States": 5000}

tbl = pd.read_csv("outputs/tournament_probs_v1.csv", index_col=0)
field_mass = 1.0 - tbl.loc[list(BETMGM), "P_champion"].sum()
shin = shin_probs(BETMGM, residual_mass=float(field_mass))
raw = implied_raw(BETMGM)
print(f"Champion market (field mass from model: {field_mass:.1%}; "
      f"raw quoted sum {sum(raw.values()):.3f} -> Shin de-vig):")
print(f"{'team':14s} {'model':>7s} {'Shin':>7s} {'raw':>7s}")
for t in BETMGM:
    print(f"{t:14s} {tbl.loc[t, 'P_champion']:7.1%} {shin[t]:7.1%} {raw[t]:7.1%}")

# ---- live match-level scoring (frozen eve-of-tournament beliefs) ----
results = load_results()
gfx = wc2026_group_fixtures(results)
played = gfx.dropna(subset=["home_score", "away_score"])
if played.empty:
    print("\nNo 2026 matches played yet — match-level scoring starts with the data refresh.")
    sys.exit(0)

elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
df = df[df["date"] < "2026-06-11"]
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
model = DixonColes().fit(df, pd.Timestamp("2026-06-11"))
elo_now = ratings_asof(elo_hist, "2026-06-11")
cap = pd.read_csv("outputs/capital.csv").query("tournament == 'wc2026'").set_index("team")["capital_z"]
b_cap = json.load(open("outputs/capital_beta.json"))["beta_capital"]

lls, fav_p, fav_w = [], [], []
print(f"\nMatch-level scoring ({len(played)} played):")
for r in played.itertuples(index=False):
    lh, la = model.predict_lambdas(elo_now[r.home_team], elo_now[r.away_team], neutral=bool(r.neutral))
    d = b_cap * float(cap.get(r.home_team, 0.0) - cap.get(r.away_team, 0.0))
    p = model.outcome_probs(lh * np.exp(d), la * np.exp(-d))
    actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
    lls.append(log_score(p[actual]))
    k = int(np.argmax(p)); fav_p.append(p[k]); fav_w.append(int(k == actual))
    res = f"{int(r.home_score)}-{int(r.away_score)}"
    print(f"  {r.home_team} {res} {r.away_team}: P(realised) {p[actual]:.0%}, LL {lls[-1]:.3f}")
print(f"\nRunning log-loss: {np.mean(lls):.4f} (uniform 1.0986) | "
      f"favourite calibration: {np.mean(fav_p):.0%} predicted vs {np.mean(fav_w):.0%} observed")
