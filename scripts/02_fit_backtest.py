"""Step 2: fit Dixon-Coles (Elo covariates) and backtest on World Cup 2022.

Protocol: point-in-time fit on all matches before the tournament, then
evaluate predictive log-loss on the 64 WC2022 matches (group + KO at 90').
Baselines: uniform (1/3,1/3,1/3) and a draw-aware constant baseline using
historical WC outcome frequencies.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_results
from wc26.elo import compute_elo_history
from wc26.dixon_coles import DixonColes

results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score","away_score"]).copy()
key = ["date","home_team","away_team"]
df["dup"] = df.groupby(key).cumcount(); elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key+["dup","elo_home_pre","elo_away_pre"]], on=key+["dup"], validate="1:1").sort_values("date")

FIT_DATE = pd.Timestamp("2022-11-20")
train = df[df["date"] >= "2005-01-01"]
model = DixonColes().fit(train, FIT_DATE)
print("Estimated parameters:", {k: round(v,4) if isinstance(v,float) else v for k,v in model.params_.items()})

wc22 = df[(df["tournament"]=="FIFA World Cup") & (df["date"]>=FIT_DATE) & (df["date"]<="2022-12-18")]
print(f"\nWC2022 matches evaluated: {len(wc22)}")

ll_model, ll_unif, hits = [], [], 0
for r in wc22.itertuples(index=False):
    lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=True)
    pH, pD, pA = model.outcome_probs(lh, la)
    actual = 0 if r.home_score>r.away_score else (1 if r.home_score==r.away_score else 2)
    p = [pH,pD,pA]
    ll_model.append(-np.log(p[actual])); ll_unif.append(-np.log(1/3))
    hits += int(np.argmax(p)==actual)
print(f"Model log-loss   : {np.mean(ll_model):.4f}")
print(f"Uniform log-loss : {np.mean(ll_unif):.4f}")
print(f"Outcome accuracy : {hits/len(wc22):.1%}  (note: NOT the right metric, colour only)")

# Crude calibration: mean favourite probability vs favourite win frequency
fav_p, fav_win = [], []
for r in wc22.itertuples(index=False):
    lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=True)
    p = model.outcome_probs(lh, la)
    k = int(np.argmax(p)); fav_p.append(p[k])
    actual = 0 if r.home_score>r.away_score else (1 if r.home_score==r.away_score else 2)
    fav_win.append(int(actual==k))
print(f"Favourite calibration: predicted {np.mean(fav_p):.1%} vs observed {np.mean(fav_win):.1%}")
