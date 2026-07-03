"""Step 25: shadow-model scoreboard — score every variant on played matches.

Beliefs are frozen at 2026-06-11; variants differ ONLY by which residual
tilts they apply. For each played 2026 match we take each variant's frozen
pre-match (H/D/A) prediction and accumulate its log-loss. This is the honest
"with vs without" the user asked for — reported live with the explicit
caveat that one tournament cannot adjudicate (see the n and the noise).
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_results, wc2026_fixtures
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes
from wc26.tilts import load_team_tilt, load_city_tilt

results = load_results()
elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
df = df[df["date"] < "2026-06-11"]
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
model = DixonColes().fit(df, pd.Timestamp("2026-06-11"))
elo = ratings_asof(elo_hist, "2026-06-11")

# tilt components
cap = pd.read_csv("outputs/capital.csv").query("tournament=='wc2026'").set_index("team")["capital_z"]
bc = json.load(open("outputs/capital_beta.json"))["beta_capital"]
fz = pd.read_csv("outputs/fatigue.csv", index_col=0)["fatigue_z"]
bf = json.load(open("outputs/fatigue_beta.json"))["beta_fatigue"]
cap_only = {t: bc * z for t, z in cap.items()}
fat_only = {t: bf * float(z) for t, z in fz.items()}
kz = pd.read_csv("outputs/cohesion.csv", index_col=0)["cohesion_z"]
bk = json.load(open("outputs/cohesion_beta.json"))["beta_cohesion"]
coh_only = {t: bk * float(z) for t, z in kz.items()}
combined = load_team_tilt()  # fatigue only (capital removed from official model)
def _merge(*ds):
    out = {}
    for d in ds:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0.0) + v
    return out
cap_fat = _merge(cap_only, fat_only)
alt = load_city_tilt() or {}
dia = {(r.team, r.city): r.log_tilt for r in pd.read_csv("outputs/diaspora_tilt.csv").itertuples(index=False)}
alt_dia = dict(alt)
for k, v in dia.items():
    alt_dia[k] = alt_dia.get(k, 0.0) + v

VARIANTS = {
    "Elo only": (None, None),
    "Full model": (combined, alt),
    "no altitude": (combined, None),
    "no fatigue": (None, alt),
    "+ capital (shadow)": (_merge(combined, cap_only), alt),
    "+ cohesion (shadow)": (_merge(combined, coh_only), alt),
    "+ diaspora (shadow)": (combined, alt_dia),
    # the kitchen sink: deployed tilts + every shadow bet at once — the "what if we
    # had admitted everything" counterfactual.
    "everything on (all shadows in)": (_merge(combined, cap_only, coh_only), alt_dia),
}

played = wc2026_fixtures(results).dropna(subset=["home_score", "away_score"]).sort_values("date")


def predict(home, away, neutral, city, tt, ct):
    lh, la = model.predict_lambdas(elo[home], elo[away], neutral=neutral)
    d = 0.0
    if tt:
        d += tt.get(home, 0.0) - tt.get(away, 0.0)
    if ct:
        d += ct.get((home, city), 0.0) - ct.get((away, city), 0.0)
    return model.outcome_probs(lh * np.exp(d), la * np.exp(-d))


rows = []
for name, (tt, ct) in VARIANTS.items():
    lls = []
    for r in played.itertuples(index=False):
        p = predict(r.home_team, r.away_team, bool(r.neutral), r.city, tt, ct)
        actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        lls.append(-np.log(max(p[actual], 1e-12)))
    rows.append({"variant": name, "n": len(lls), "log_loss": round(float(np.mean(lls)), 4)})

board = pd.DataFrame(rows).sort_values("log_loss")
board.to_csv("outputs/shadow_scores.csv", index=False)
print(f"Shadow scoreboard over {len(played)} played matches (uniform = 1.0986):")
print(board.to_string(index=False))
print(f"\nspread best-worst: {board.log_loss.max()-board.log_loss.min():.4f} "
      f"(n={len(played)} — far too few to be significant; this accrues over time)")
