"""Diagnostic: Elo-provenance outliers vs live 2026 performance.

Question (Simone, 2026-07-04): can we identify teams whose RATING WAS BUILT in an
unusual way (how they accumulated Elo, not just its level) and should they be
handled differently?

Profile per WC26 team, computed point-in-time from our own files (no external
source): rating points gained in the trailing 4 years (momentum), share and net
points from cross-confederation opponents, friendly share, mean opponent Elo, and
ELITE EXPOSURE = matches against 1800+ opponents. Confederations are inferred from
continental-championship participation. Each profile is compared with the team's
LIVE residual: actual points per match minus the frozen model's expectation,
over every played 2026 match.

Findings (recorded 2026-07-04, 88 matches, n=48 teams — one tournament, so
suggestive, not a verdict):
  - ELITE EXPOSURE is the strongest axis: r = +0.29 (t = 2.09) with the live
    residual. Bottom-quartile exposure teams run -0.11 pts/match vs +0.20 for the
    top quartile.
  - The "suspicious profile" (big 4-year climb + low elite exposure) captures the
    flops almost by name - Turkey (+192 Elo/4y, resid -0.55), Uzbekistan (+163,
    -0.98), Jordan (+118, -0.70) - mean residual -0.32 vs field -0.07, with honest
    counterexamples (DR Congo +0.28, Morocco +0.31) that forbid hand-waving.
  - Momentum alone is weak (r = -0.14); it needs the exposure conditioning.

Implication: a GATEABLE candidate for future cycles - an elite-exposure shrinkage
on ratings (or an exposure x elodiff interaction in the goal model), testable on
WC 1994-2022 exactly like scripts/archive/44. Unlike the cross-confederation
player-Elo (blocked by the data wall), this needs nothing but results.csv.
Backlog item 10. The 2026 deployed model stays frozen regardless.

Standalone; run from the repo root:  python scripts/archive/46_elo_provenance.py
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_results, wc2026_fixtures, reconstruct_groups, wc2026_group_fixtures

results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
gfx = wc2026_group_fixtures(results)
groups = reconstruct_groups(gfx)
WC = sorted({t for g in groups.values() for t in g})

# --- confederation map from continental-championship participation ---
CUPS = {"UEFA Euro": "UEFA", "Copa América": "CONMEBOL", "African Cup of Nations": "CAF",
        "AFC Asian Cup": "AFC", "Gold Cup": "CONCACAF", "CONCACAF Championship": "CONCACAF",
        "Oceania Nations Cup": "OFC", "OFC Nations Cup": "OFC"}
conf = {}
for tname, c in CUPS.items():
    sub = results[results["tournament"].str.contains(tname, na=False, regex=False)]
    for col in ("home_team", "away_team"):
        for t in sub[col].unique():
            conf.setdefault(t, {}).setdefault(c, 0)
            conf[t][c] += 1
CONF = {t: max(d, key=d.get) for t, d in conf.items()}
print(f"confed mapped: {len(CONF)} teams; WC26 unmapped: {[t for t in WC if t not in CONF]}")

# --- per-team accumulation profile up to 2026-06-11 ---
e = elo.merge(results[["date","home_team","away_team","tournament","neutral"]],
              on=["date","home_team","away_team"], how="left")
e = e[e["date"] < "2026-06-11"]
rows = []
for t in WC:
    h = e[e.home_team == t].assign(d=lambda x: x.elo_home_post - x.elo_home_pre,
                                   opp=lambda x: x.away_team, opp_elo=lambda x: x.elo_away_pre)
    a = e[e.away_team == t].assign(d=lambda x: x.elo_away_post - x.elo_away_pre,
                                   opp=lambda x: x.home_team, opp_elo=lambda x: x.elo_home_pre)
    m = pd.concat([h[["date","tournament","d","opp","opp_elo"]], a[["date","tournament","d","opp","opp_elo"]]]).sort_values("date")
    m["cross"] = m["opp"].map(lambda o: CONF.get(o, "?") != CONF.get(t, "!"))
    m["friendly"] = m["tournament"].eq("Friendly")
    recent = m[m.date >= "2022-06-11"]  # last 4y (the rating-relevant window)
    tot = m.d.sum()
    rows.append({
        "team": t, "conf": CONF.get(t, "?"), "elo_now": 1500 + tot,
        # net rating points from cross-confederation opponents, ever and last 4y
        "cross_pts_4y": recent[recent.cross].d.sum(),
        "cross_share_matches_4y": recent.cross.mean(),
        "n_cross_4y": int(recent.cross.sum()),
        "friendly_pts_4y": recent[recent.friendly].d.sum(),
        "pts_4y": recent.d.sum(),                      # momentum: rating gained in 4y
        "mean_opp_4y": recent.opp_elo.mean(),          # schedule strength
        "n_top1800_4y": int((recent.opp_elo >= 1800).sum()),  # elite exposure
        "n_4y": len(recent),
    })
prof = pd.DataFrame(rows).set_index("team")

# --- live residual: actual points vs model-expected, over all played WC matches ---
from wc26.dixon_coles import DixonColes
from wc26.elo import ratings_asof
from wc26.tilts import load_team_tilt, load_city_tilt
df = results.dropna(subset=["home_score","away_score"]).copy()
df = df[df["date"] < "2026-06-11"]
key = ["date","home_team","away_team"]
df["dup"]=df.groupby(key).cumcount()
e2=elo.copy(); e2["dup"]=e2.groupby(key).cumcount()
df=df.merge(e2[key+["dup","elo_home_pre","elo_away_pre"]],on=key+["dup"],validate="1:1")
model=DixonColes().fit(df, pd.Timestamp("2026-06-11"))
R=ratings_asof(elo,"2026-06-11"); tilt=load_team_tilt() or {}; ct=load_city_tilt() or {}
played=wc2026_fixtures(results).dropna(subset=["home_score","away_score"])
res={t:{"exp":0.0,"act":0.0,"n":0} for t in WC}
for r in played.itertuples(index=False):
    lh,la=model.predict_lambdas(R[r.home_team],R[r.away_team],neutral=bool(r.neutral))
    d=tilt.get(r.home_team,0)-tilt.get(r.away_team,0)+ct.get((r.home_team,r.city),0)-ct.get((r.away_team,r.city),0)
    pH,pD,pA=model.outcome_probs(lh*np.exp(d),la*np.exp(-d))
    hp=3 if r.home_score>r.away_score else (1 if r.home_score==r.away_score else 0)
    for t,exp,act in ((r.home_team,3*pH+pD,hp),(r.away_team,3*pA+pD,3-hp if hp!=1 else 1)):
        if t in res: res[t]["exp"]+=exp; res[t]["act"]+=act; res[t]["n"]+=1
prof["live_resid"]=[ (res[t]["act"]-res[t]["exp"])/max(res[t]["n"],1) for t in prof.index]
prof["n_live"]=[res[t]["n"] for t in prof.index]

# --- outliers & correlations ---
prof["cross_pts_z"]=(prof.cross_pts_4y-prof.cross_pts_4y.mean())/prof.cross_pts_4y.std()
prof["momentum_z"]=(prof.pts_4y-prof.pts_4y.mean())/prof.pts_4y.std()
print("\n=== correlation of profile vs live residual (points per match vs expectation) ===")
for c in ["cross_pts_4y","cross_share_matches_4y","n_cross_4y","friendly_pts_4y","pts_4y","mean_opp_4y","n_top1800_4y"]:
    print(f"  {c:26s} r = {prof[c].corr(prof.live_resid):+.3f}")
print("\n=== top-10 most negative live residuals (flops) ===")
cols=["conf","elo_now","pts_4y","cross_pts_4y","n_cross_4y","n_top1800_4y","live_resid"]
print(prof.sort_values("live_resid")[cols].head(10).round(2).to_string())
print("\n=== top-10 overachievers ===")
print(prof.sort_values("live_resid",ascending=False)[cols].head(10).round(2).to_string())
prof.round(3).to_csv("/private/tmp/claude-501/-Users-mowy01-wc26/abe958e1-65f9-4eb5-9699-7286742936c8/scratchpad/elo_provenance.csv")
